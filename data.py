from pathlib import Path
import json
import re
from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast

DATA_ROOT = Path('data')


class PredicateSchema:

    def __init__(self, predicate2idx: Dict[str, int]):
        """
        Predicate schema. Calling from_file constructor is recommended.

        Args:
            predicate2idx: Dict mapping predicate string to index.
        """
        self.predicate2idx = predicate2idx
        self.idx2predicate = [0] * len(predicate2idx)
        for p, i in predicate2idx.items():
            self.idx2predicate[i] = p

    @classmethod
    def from_file(cls, filename: str = 'schemas.json'):
        """
        Constructor reads predicate schema from file.

        Args:
            filename: Filename in DATA_ROOT.
        """
        with open(DATA_ROOT / filename) as f:
            predicate2idx = json.load(f)[1]
        return cls(predicate2idx)

    def __len__(self) -> int:
        return len(self.idx2predicate)


class SPOExtractionDataset(Dataset):

    predicate_schema = PredicateSchema.from_file()
    sub_patterns = [(re.compile(r'\s?(\d+)\s?'), r'\g<1>')]

    def __init__(self, data: List[Dict[str, Union[str, List[List[str]]]]], tokenizer: BertTokenizerFast):
        """
        Dataset for subject, predicate, object triplet extraction.
        Calling from_file constructor is recommended.

        Args:
            data: List of data entries.
            tokenizer: BERT tokenizer.
        """
        self.tokenizer = tokenizer
        tokenized = tokenizer([self.clean_str(d['text']) for d in data])['input_ids']
        self.input_ids, self.predicates, self.subject_spans, self.object_spans = [], [], [], []
        for input_ids, entry in zip(tokenized, data):
            try:
                predicates, subject_spans, object_spans = self.process_entry(input_ids, entry['spo_list'])
            except ValueError:
                continue
            self.input_ids.append(input_ids)
            self.predicates.append(predicates)
            self.subject_spans.append(subject_spans)
            self.object_spans.append(object_spans)

    @classmethod
    def from_file(cls, filename: str, tokenizer: BertTokenizerFast):
        """
        Constructor reads dataset from file

        Args:
            filename: Filename in DATA_ROOT.
            tokenizer: BERT tokenizer.
        """
        with open(DATA_ROOT / filename) as f:
            data = json.load(f)
        return cls(data, tokenizer)

    @staticmethod
    def clean_str(s: str) -> str:
        """
        Implements strategies to clean strings.

        Current rules:
            1. Lower letters.
            2. Trim white spaces around numbers.

        Args:
            s: Input string.

        Returns:
            Cleaned string.
        """
        clean_s = s.lower()
        for pattern, sub in SPOExtractionDataset.sub_patterns:
            clean_s = pattern.sub(sub, clean_s)
        return clean_s

    def process_entry(self, input_ids: List[int], spo_list: List[List[str]]) \
            -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Processes each data entry.

        Args:
            input_ids: Pre-tokenized text.
            spo_list: List of subject, predicate and object triplets.

        Returns:
            Three lists, predicates, subject spans and object spans.
        """
        predicates = [self.predicate_schema.predicate2idx[p] for p in list(zip(*spo_list))[1]]
        entities = [self.clean_str(s) for spo in spo_list for s in (spo[0], spo[2])]
        tokenized_entities = self.tokenizer(entities)['input_ids']

        subject_spans, object_spans = [], []
        unicoded_text = ''.join(map(chr, input_ids))
        for i in range(0, len(tokenized_entities), 2):
            subject_encoded = tokenized_entities[i][1:-1]
            object_encoded = tokenized_entities[i + 1][1:-1]
            subject_start = unicoded_text.find(''.join(map(chr, subject_encoded)))
            object_start = unicoded_text.find(''.join(map(chr, object_encoded)))

            if -1 in (subject_start, object_start):
                raise ValueError('Substring not found.')

            subject_spans.append((subject_start, subject_start + len(subject_encoded) - 1))
            object_spans.append((object_start, object_start + len(object_encoded) - 1))

        return predicates, subject_spans, object_spans

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            idx: Index.

        Returns:
            input_ids: seq_len
            predicate_hot: num_predicates
            position_hot: seq_len, num_predicates, 4
        """
        input_ids = torch.LongTensor(self.input_ids[idx])
        predicate_hot = torch.zeros(len(self.predicate_schema))
        predicate_hot[self.predicates[idx]] = 1

        position_hot = torch.zeros((len(input_ids), len(self.predicate_schema), 4))
        for i in range(len(self.predicates[idx])):
            for j, pos in enumerate(self.subject_spans[idx][i] + self.object_spans[idx][i]):
                position_hot[pos, self.predicates[idx][i], j] = 1

        return input_ids, predicate_hot, position_hot

    @staticmethod
    def padding_collate(batch: List[Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]]) \
            -> Dict[str, Union[torch.LongTensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Collate function generates tensors with a batch of data.

        Args:
            batch: Batch of data.

        Returns:
            input_ids: batch_size, pad_len
            attention_mask: batch_size, pad_len
            targets:
                predicate_hot: batch_size, num_predicates
                position_hot: batch_size, pad_len, num_predicates, 4
        """
        input_ids, predicate_hot, position_hot = zip(*batch)
        attention_mask = list(map(torch.ones_like, input_ids))
        return {
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=0),
            'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=0),
            'targets': (torch.vstack(predicate_hot), pad_sequence(position_hot, batch_first=True, padding_value=0.))
        }

    def get_dataloader(self, batch_size: int, shuffle: bool = True, pin_memory: bool = True, **kwargs) -> DataLoader:
        """
        Creates dataloader for dataset instance.

        Args:
            batch_size: Batch size.
            shuffle: Whether shuffle data at every epoch. Default: True
            pin_memory: Whether use pinned memory for faster data
                transfer to GPUs. Default: True
            **kwargs: Other kwargs (except for collate_fn) supported by
                DataLoader.

        Returns:
            DataLoader.
        """
        return DataLoader(self, batch_size=batch_size, collate_fn=self.padding_collate,
                          shuffle=shuffle, pin_memory=pin_memory, **kwargs)
