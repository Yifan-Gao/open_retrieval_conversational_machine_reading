import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, SequenceClassifierOutput


class RobertaForMultipleSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.top_k_snippets = config.top_k_snippets

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.roberta = RobertaModel(config)

        # TODO is the dense layer necessary?
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_selfattn = nn.Linear(config.hidden_size, 1)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        # # TODO is the dense layer & tanh necessary?
        # x = self.dense(pooled_output)
        # x = torch.tanh(x)
        # x = self.dropout(x)

        batch_size = len(input_ids)
        selfattn_weight = self.w_selfattn(self.dropout(pooled_output))
        selfattn_weight = F.softmax(selfattn_weight.view(batch_size, self.top_k_snippets), dim=-1)
        selfattn = torch.sum(selfattn_weight.unsqueeze(-1) * pooled_output.view(batch_size, self.top_k_snippets, -1), dim=1)
        # score = self.w_output(self.dropout(selfattn))
        #
        # x = pooled_output

        logits = self.out_proj(selfattn)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )