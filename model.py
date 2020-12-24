from collections import namedtuple
from typing import Optional, Dict

import torch
from torch import nn
from transformers import BertModel

from data import SPOExtractionDataset

Output = namedtuple('Output', ['loss', 'predocate_logits', 'position_logits'])


class BertForJointSPOExtraction(nn.Module):

    def __init__(self, bert: BertModel, num_predicates: int = len(SPOExtractionDataset.predicate_schema)):
        """
        Model extracting subject, predicate, object triplet based on
        pre-trained BERT model.

        Args:
            bert: Pre-trained BERT.
            num_predicates: Size of predicate vocabulary.
        """
        super(BertForJointSPOExtraction, self).__init__()
        self.bert = bert
        hidden_size = self.bert.config.hidden_size
        self.predicate_fc = nn.Linear(hidden_size, num_predicates)
        self.position_fc = nn.Linear(hidden_size, num_predicates * 4)
        self.predicate_loss_fn = nn.BCEWithLogitsLoss()
        self.position_loss_fn_base = nn.BCEWithLogitsLoss(reduction='none')
        self.num_predicates = num_predicates

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor,
                predicate_hot: Optional[torch.Tensor] = None, position_hot: Optional[torch.Tensor] = None) -> Output:
        """

        Args:
            input_ids: batch_size, pad_len
            attention_mask: batch_size, pad_len
            predicate_hot: batch_size, num_predicates
            position_hot: batch_size, pad_len, num_predicates, 4

        Returns:
            Output with loss (optional) and logits.
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # batch_size, pad_len, hidden_size

        predicate_logits = self.predicate_fc(bert_out[:, 0, :])  # batch_size, num_predicates
        position_logits = self.position_fc(bert_out).view(input_ids.shape[0], -1, self.num_predicates, 4)
        # batch_size, pad_len, num_predicates, 4
        position_logits.masked_fill_(~attention_mask[:, :, None, None].bool(), float('-inf'))

        loss = None
        if predicate_hot is not None and position_hot is not None:
            predicate_loss = self.predicate_loss_fn(predicate_logits, predicate_hot)
            position_loss = self.position_loss_fn(position_logits, position_hot, predicate_hot, attention_mask)
            loss = predicate_loss + position_loss

        return Output(loss, predicate_logits, position_logits)

    def position_loss_fn(self, position_logits: torch.Tensor, position_hot: torch.Tensor, predicate_hot: torch.Tensor,
                         attention_mask: torch.LongTensor) -> torch.Tensor:
        """
        Position loss function.

        Args:
            position_logits: batch_size, pad_len, num_predicates, 4
            position_hot: batch_size, pad_len, num_predicates, 4
            predicate_hot: batch_size, num_predicates
            attention_mask: batch_size, pad_len

        Returns:
            0-dim loss.
        """
        seq_mask = attention_mask[:, :, None, None]
        loss = self.position_loss_fn_base(position_logits, position_hot)  # batch_size, pad_len, num_predicates, 4
        masked_loss = loss * predicate_hot[:, None, :, None] * seq_mask
        sentence_loss = (masked_loss / seq_mask.sum(dim=1, keepdim=True)).sum(1)  # batch_size, num_predicates, 4
        return sentence_loss.mean()
