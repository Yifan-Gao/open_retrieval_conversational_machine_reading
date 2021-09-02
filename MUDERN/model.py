import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn import LayerNorm, TransformerEncoderLayer, TransformerEncoder
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, SequenceClassifierOutput


class DiscernForMultipleSequence(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, lambda_entailment=3.0, sequence_transformer_layer=3,):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lambda_entailment = lambda_entailment
        self.sequence_transformer_layer = sequence_transformer_layer

        if sequence_transformer_layer > 0:
            encoder_layer = TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, 4 * config.hidden_size)
            encoder_norm = LayerNorm(config.hidden_size)
            self.transformer_encoder = TransformerEncoder(encoder_layer, sequence_transformer_layer, encoder_norm)

        self.w_entail = nn.Linear(config.hidden_size, 3)

        self.w_selfattn = nn.Linear(config.hidden_size, 1)
        self.w_output = nn.Linear(config.hidden_size, 3)

        self.init_weights()

        if sequence_transformer_layer > 0:
            self._reset_transformer_parameters()

        # self.entail_emb = nn.Parameter(torch.rand(3, config.hidden_size))
        # nn.init.normal_(self.entail_emb)

    def _reset_transformer_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name, param in self.named_parameters():
            if 'transformer' in name and param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def logic_op(self, input, input_mask, retrieval_score=None):
        selfattn_unmask = self.w_selfattn(self.dropout(input))
        selfattn_unmask.masked_fill_(~input_mask, -float('inf'))
        selfattn_weight = F.softmax(selfattn_unmask, dim=1)
        selfattn = torch.sum(selfattn_weight * input, dim=1)
        score = self.w_output(self.dropout(selfattn))
        return score

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rule_idx=None,
        user_idx=None,
        gold_rule_idx_mask=None,
        label_entail=None,
        retrieval_score=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # v1: use user + rule cls
        # tenc_input, tenc_mask = [], []
        # for idx, output in enumerate(outputs[0]):
        #     tenc_idx = torch.cat([user_idx[idx], rule_idx[idx]], dim=-1)
        #     tenc_input.append(torch.index_select(output, 0, tenc_idx))
        #     tenc_mask.append(torch.tensor([1] * tenc_idx.shape[0], dtype=torch.bool, device=self.device))
        # tenc_out = torch.nn.utils.rnn.pad_sequence(tenc_input, batch_first=True)
        # tenc_mask = nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]
        # logits = self.logic_op(tenc_out, tenc_mask)  # [bz, 3]

        # v2: use rule cls only
        # tenc_input, tenc_mask = [], []
        # for idx, output in enumerate(outputs[0]):
        #     tenc_input.append(torch.index_select(output, 0, rule_idx[idx]))
        #     tenc_mask.append(torch.tensor([1] * rule_idx[idx].shape[0], dtype=torch.bool, device=self.device))
        # tenc_out = torch.nn.utils.rnn.pad_sequence(tenc_input, batch_first=True)
        # tenc_mask = nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]
        # logits = self.logic_op(tenc_out, tenc_mask)  # [bz, 3]

        # v3/4: rule cls + entailment
        # tenc_input, tenc_mask = [], []
        # for idx, output in enumerate(outputs[0]):
        #     tenc_input.append(torch.index_select(output, 0, rule_idx[idx]))
        #     tenc_mask.append(torch.tensor([1] * rule_idx[idx].shape[0], dtype=torch.bool, device=self.device))
        # tenc_out = torch.nn.utils.rnn.pad_sequence(tenc_input, batch_first=True)
        # tenc_mask = nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]
        # logits = self.logic_op(tenc_out, tenc_mask, retrieval_score)  # [bz, 3]
        # entail_logits = self.w_entail(self.dropout(tenc_out))  # [bz * seqlen * 3]

        # v5: use user + rule cls + entailment
        # tenc_input, tenc_mask, trule_input = [], [], []
        # for idx, output in enumerate(outputs[0]):
        #     tenc_idx = torch.cat([user_idx[idx], rule_idx[idx]], dim=-1)
        #     tenc_input.append(torch.index_select(output, 0, tenc_idx))
        #     tenc_mask.append(torch.tensor([1] * tenc_idx.shape[0], dtype=torch.bool, device=self.device))
        #     trule_input.append(torch.index_select(output, 0, rule_idx[idx]))
        # tenc_out = torch.nn.utils.rnn.pad_sequence(tenc_input, batch_first=True)
        # trule_out = torch.nn.utils.rnn.pad_sequence(trule_input, batch_first=True)
        # tenc_mask = nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]
        # logits = self.logic_op(tenc_out, tenc_mask)  # [bz, 3]
        # entail_logits = self.w_entail(self.dropout(trule_out))  # [bz * seqlen * 3]

        # v6 (from v3): add transformer layers
        # user_rule_input, user_rule_mask = [], []
        # for idx, output in enumerate(outputs[0]):
        #     user_rule_idx = torch.cat([user_idx[idx], rule_idx[idx]], dim=-1)
        #     user_rule_input.append(torch.index_select(output, 0, user_rule_idx))
        #     user_rule_mask.append(torch.tensor([False] * user_rule_idx.shape[0], dtype=torch.uint8, device=self.device))
        #
        # user_rule_ptm_out = torch.nn.utils.rnn.pad_sequence(user_rule_input)
        # user_rule_mask = torch.nn.utils.rnn.pad_sequence(user_rule_mask, batch_first=True, padding_value=True)
        # user_rule_transformer_out = self.transformer_encoder(user_rule_ptm_out, src_key_padding_mask=user_rule_mask)
        # user_rule_transformer_out = torch.transpose(user_rule_transformer_out, 0, 1).contiguous()  # [bz * seqlen * dim]
        #
        # # select rule encoded
        # rule_input, rule_mask = [], []
        # for idx, user_rule_transformer_out_i in enumerate(user_rule_transformer_out):
        #     seq_rule_idx = torch.tensor([i for i in range(len(user_idx[idx]), len(user_idx[idx]) + len(rule_idx[idx]))], dtype=torch.long, device=self.device)
        #     rule_input.append(torch.index_select(user_rule_transformer_out_i, 0, seq_rule_idx))
        #     rule_mask.append(torch.tensor([1] * len(rule_idx[idx]), dtype=torch.bool, device=self.device))
        # rule_out = torch.nn.utils.rnn.pad_sequence(rule_input, batch_first=True)
        # rule_mask = nn.utils.rnn.pad_sequence(rule_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]
        #
        # logits = self.logic_op(rule_out, rule_mask)  # [bz, 3]
        # entail_logits = self.w_entail(self.dropout(rule_out))  # [bz * seqlen * 3]

        # v7: from v3: add entailment vector
        # tenc_input, tenc_mask = [], []
        # for idx, output in enumerate(outputs[0]):
        #     tenc_input.append(torch.index_select(output, 0, rule_idx[idx]))
        #     tenc_mask.append(torch.tensor([1] * rule_idx[idx].shape[0], dtype=torch.bool, device=self.device))
        # tenc_out = torch.nn.utils.rnn.pad_sequence(tenc_input, batch_first=True)
        # tenc_mask = nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]
        #
        # entail_logits = self.w_entail(self.dropout(tenc_out))  # [bz * seqlen * 3]
        # entail_state = torch.matmul(entail_logits, self.entail_emb)
        # # cat_state = torch.cat([entail_state, tenc_out], dim=-1)
        # cat_state = entail_state + tenc_out
        # logits = self.logic_op(cat_state, tenc_mask)  # [bz, 3]

        # v9 (from v5): add transformer layers + use user + rule cls + entailment
        user_rule_input, user_rule_mask, user_rule_transformer_mask = [], [], []
        for idx, output in enumerate(outputs[0]):
            user_rule_idx = torch.cat([user_idx[idx], rule_idx[idx]], dim=-1)
            user_rule_input.append(torch.index_select(output, 0, user_rule_idx))
            user_rule_mask.append(torch.tensor([False] * user_rule_idx.shape[0], dtype=torch.uint8, device=self.device))
            user_rule_transformer_mask.append(torch.tensor([1] * user_rule_idx.shape[0], dtype=torch.bool, device=self.device))

        user_rule_transformer_mask = nn.utils.rnn.pad_sequence(user_rule_transformer_mask, batch_first=True).unsqueeze(-1)  # [bz * seqlen * 1]

        if self.sequence_transformer_layer > 0:
            user_rule_ptm_out = torch.nn.utils.rnn.pad_sequence(user_rule_input)
            user_rule_mask = torch.nn.utils.rnn.pad_sequence(user_rule_mask, batch_first=True, padding_value=True)
            user_rule_transformer_out = self.transformer_encoder(user_rule_ptm_out, src_key_padding_mask=user_rule_mask)
            user_rule_transformer_out = torch.transpose(user_rule_transformer_out, 0, 1).contiguous()  # [bz * seqlen * dim]

            # select rule encoded
            rule_input = []
            for idx, user_rule_transformer_out_i in enumerate(user_rule_transformer_out):
                seq_rule_idx = torch.tensor([i for i in range(len(user_idx[idx]), len(user_idx[idx]) + len(rule_idx[idx]))],
                                            dtype=torch.long, device=self.device)
                rule_input.append(torch.index_select(user_rule_transformer_out_i, 0, seq_rule_idx))
            rule_out = torch.nn.utils.rnn.pad_sequence(rule_input, batch_first=True)
        else:
            user_rule_transformer_out = torch.nn.utils.rnn.pad_sequence(user_rule_input, batch_first=True)
            rule_input = []
            for idx, output in enumerate(outputs[0]):
                rule_input.append(torch.index_select(output, 0, rule_idx[idx]))
            rule_out = torch.nn.utils.rnn.pad_sequence(rule_input, batch_first=True)

        logits = self.logic_op(user_rule_transformer_out, user_rule_transformer_mask)  # [bz, 3]
        entail_logits = self.w_entail(self.dropout(rule_out))  # [bz * seqlen * 3]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # v3/5: rule cls + entailment
        entail_loss_fct = CrossEntropyLoss(ignore_index=-100)
        entail_loss = entail_loss_fct(entail_logits.view(-1, 3), label_entail.view(-1))
        loss += entail_loss * self.lambda_entailment

        # v4: rule cls + entailment (gold)
        # entail_loss_fct = CrossEntropyLoss(ignore_index=-100)
        # label_entail.masked_fill_(~gold_rule_idx_mask.bool(), -100)
        # entail_loss = entail_loss_fct(entail_logits.view(-1, 3), label_entail.view(-1))
        # loss += entail_loss * self.lambda_entailment

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
