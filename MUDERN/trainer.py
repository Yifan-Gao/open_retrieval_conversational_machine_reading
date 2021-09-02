# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy
from dataclasses import dataclass
import torch
import torch.nn as nn


import collections
import inspect
import math
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    hp_params,
    is_azureml_available,
    is_comet_available,
    is_fairscale_available,
    is_mlflow_available,
    is_optuna_available,
    is_ray_tune_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    init_deepspeed,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import WEIGHTS_NAME, is_apex_available, is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    LabelSmoother,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging


_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_mlflow_available():
    from transformers.integrations import MLflowCallback

    DEFAULT_CALLBACKS.append(MLflowCallback)

if is_azureml_available():
    from transformers.integrations import AzureMLCallback

    DEFAULT_CALLBACKS.append(AzureMLCallback)

if is_fairscale_available():
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


@dataclass
class DiscernDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_keys = ["input_ids", "attention_mask", "label"]
        remaining_keys = [k for k in features[0].keys() if k not in batch_keys]
        features_for_padding = [{k: fea[k] for k in batch_keys} for fea in features]
        batch = self.tokenizer.pad(
            features_for_padding,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        for _k in remaining_keys:
            batch[_k] = [torch.tensor(fea[_k], dtype=torch.long) for fea in features]

        batch["gold_rule_idx_mask"] = nn.utils.rnn.pad_sequence(batch["gold_rule_idx_mask"], batch_first=True, padding_value=0)
        batch["label_entail"] = nn.utils.rnn.pad_sequence(batch["label_entail"], batch_first=True, padding_value=-100)
        return batch


class DiscernTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, model_args=None, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        # self.eval_examples = eval_examples
        # self.post_process_function = post_process_function
        self.model_args = model_args
        self.config = config

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
            if isinstance(v, List) and isinstance(v[0], torch.Tensor):
                inputs[k] = [_v.to(self.args.device) for _v in v]

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
            """
            Main training entry point.

            Args:
                model_path (:obj:`str`, `optional`):
                    Local path to the model if the model to train has been instantiated from a local path. If present,
                    training will resume from the optimizer/scheduler states loaded here.
                trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                    The trial run or the hyperparameter dictionary for hyperparameter search.
            """
            # This might change the seed so needs to run first.
            self._hp_search_setup(trial)

            # Model re-init
            if self.model_init is not None:
                # Seed must be set before instantiating the model when using model_init.
                set_seed(self.args.seed)

                model = self.call_model_init(trial)
                if not self.is_model_parallel:
                    model = model.to(self.args.device)

                self.model = model
                self.model_wrapped = model

                # Reinitializes optimizer and scheduler
                self.optimizer, self.lr_scheduler = None, None

            # Keeping track whether we can can len() on the dataset or not
            train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            if train_dataset_is_sized:
                num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                if self.args.max_steps > 0:
                    max_steps = self.args.max_steps
                    num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                        self.args.max_steps % num_update_steps_per_epoch > 0
                    )
                else:
                    max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(self.args.num_train_epochs)
            else:
                # see __init__. max_steps is set when the dataset has no __len__
                max_steps = self.args.max_steps
                num_train_epochs = 1
                num_update_steps_per_epoch = max_steps

            if self.args.deepspeed:
                model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
                self.model = model.module
                self.model_wrapped = model  # will get further wrapped in DDP
                self.deepspeed = model  # DeepSpeedEngine object
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
            else:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState()
            self.state.is_hyper_param_search = trial is not None

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(model_path)

            model = self.model_wrapped

            # Mixed precision training with apex (torch < 1.6)
            if self.use_apex:
                model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

            # Multi-gpu training (should be after apex fp16 initialization)
            if self.args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Distributed training (should be after apex fp16 initialization)
            if self.sharded_dpp:
                model = ShardedDDP(model, self.optimizer)
            elif self.args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                    find_unused_parameters=(
                        not getattr(model.config, "gradient_checkpointing", False)
                        if isinstance(model, PreTrainedModel)
                        else True
                    ),
                )
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

            # Train!
            if is_torch_tpu_available():
                total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
            else:
                total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
                )

            num_examples = (
                self.num_examples(train_dataloader)
                if train_dataset_is_sized
                else total_train_batch_size * self.args.max_steps
            )

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0

            # Check if continuing training from a checkpoint
            if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
                self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
                epochs_trained = self.state.global_step // num_update_steps_per_epoch
                if not self.args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not self.args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                        "batches in the first epoch."
                    )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
            self.state.trial_params = hp_params(trial) if trial is not None else None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(self.args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = 0
            self._total_flos = self.state.total_flos
            model.zero_grad()

            self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

            # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
            if not self.args.ignore_data_skip:
                for epoch in range(epochs_trained):
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break

            for epoch in range(epochs_trained, num_train_epochs):
                if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)

                if is_torch_tpu_available():
                    parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                        self.args.device
                    )
                    epoch_iterator = parallel_loader
                else:
                    epoch_iterator = train_dataloader

                # Reset the past mems state at the beginning of each epoch if necessary.
                if self.args.past_index >= 0:
                    self._past = None

                steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
                self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

                for step, inputs in enumerate(epoch_iterator):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                    if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss += self.training_step(model, inputs)
                    else:
                        tr_loss += self.training_step(model, inputs)
                    self._total_flos += self.floating_point_ops(inputs)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= self.args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # Gradient clipping
                        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.use_amp:
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    self.args.max_grad_norm,
                                )

                        # Optimizer step
                        if is_torch_tpu_available():
                            xm.optimizer_step(self.optimizer)
                        elif self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        self.lr_scheduler.step()
                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.args.tpu_metrics_debug or self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            # logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            # if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            #     logger.info(
            #         f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            #     )
            #     if isinstance(self.model, PreTrainedModel):
            #         self.model = self.model.from_pretrained(
            #             self.state.best_model_checkpoint,
            #             config=self.config,
            #             sequence_transformer_layer=self.model_args.sequence_transformer_layer,
            #             lambda_entailment=self.model_args.lambda_entailment,
            #         )
            #         if not self.is_model_parallel:
            #             self.model = self.model.to(self.args.device)
            #     else:
            #         state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
            #         self.model.load_state_dict(state_dict)

            #     if self.deepspeed:
            #         self.deepspeed.load_checkpoint(
            #             self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
            #         )

            metrics = speed_metrics("train", start_time, self.state.max_steps)
            if self._total_flos is not None:
                self.store_flos()
                metrics["total_flos"] = self.state.total_flos
            self.log(metrics)

            self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()

            return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)