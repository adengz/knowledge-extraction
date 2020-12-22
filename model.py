from collections import namedtuple
from typing import Optional, Dict

import torch
from torch import nn
from transformers import BertModel

from data import SPOExtractionDataset

Output = namedtuple('Output', ['loss', 'logits'])


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
        self.position_fc = nn.Linear(hidden_size, 4)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor,
                targets: Optional[Dict[str, torch.LongTensor]] = None) -> Output:
        """

        Args:
            input_ids: batch_size, pad_len
            attention_mask: batch_size, pad_len
            targets: batch_size

        Returns:
            Output with loss and logits.
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # batch_size, pad_len, hidden_size

        predicate_logits = self.predicate_fc(bert_out[:, 0, :])  # batch_size, num_predicates
        position_logits = self.position_fc(bert_out)  # batch_size, pad_len, 4
        position_logits.masked_fill_(~attention_mask[:, :, None].bool(), float('-inf'))
        logits = {'predicate': predicate_logits,
                  'subject_start': position_logits[:, :, 0], 'subject_end': position_logits[:, :, 1],
                  'object_start': position_logits[:, :, 2], 'object_end': position_logits[:, :, 3]}

        loss = None
        if targets is not None:
            loss = 0
            for k in logits:
                weight = 1 if k == 'predicate' else 0.5
                loss += weight * self.loss_fn(logits[k], targets[k])

        return Output(loss, logits)
