from pathlib import Path
import json
import random
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

    def __init__(self, data: List[Dict[str, Union[str, List[List[str]]]]], tokenizer: BertTokenizerFast,
                 train: bool = True):
        """
        Dataset for subject, predicate, object triplet extraction.
        Calling from_file constructor is recommended.

        Args:
            data: List of data entries.
            tokenizer: BERT tokenizer.
            train: Training mode. If set to True, a 'left join' on
                sentence is performed with one sampled triplet joined
                on each sentence. Otherwise, a 'right join' on triplet
                is performed with sentences joined on all triplets.
                Default: True
        """
        self.tokenizer = tokenizer
        self.train = train
        input_ids = tokenizer([self.clean_str(d['text']) for d in data])['input_ids']
        self.predicates, self.subject_spans, self.object_spans = [], [], []
        self.input_ids = input_ids if self.train else []
        for text_encoded, entry in zip(input_ids, data):
            try:
                predicates, subject_spans, object_spans = self.process_entry(text_encoded, entry['spo_list'])
            except ValueError:
                continue
            if self.train:
                self.predicates.append(predicates)
                self.subject_spans.append(subject_spans)
                self.object_spans.append(object_spans)
            else:
                self.input_ids.extend([text_encoded] * len(predicates))
                self.predicates.extend(predicates)
                self.subject_spans.extend(subject_spans)
                self.object_spans.extend(object_spans)

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        """
        Constructor reads dataset from file

        Args:
            filename: Filename in DATA_ROOT.
            **kwargs: kwargs supported in __init__.
        """
        with open(DATA_ROOT / filename) as f:
            data = json.load(f)
        return cls(data, **kwargs)

    @staticmethod
    def clean_str(s: str) -> str:
        """
        Preprocesses string.

        Args:
            s: Input string.

        Returns:
            Cleaned string.
        """
        return s.lower().replace(' ', '')

    def process_entry(self, text_encoded: List[int], spo_list: List[List[str]]) \
            -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Processes each data entry.

        Args:
            text_encoded: Pre-tokenized text.
            spo_list: List of subject, predicate and object triplets.

        Returns:
            Three lists, predicates, subject spans and object spans.
        """
        predicates = [self.predicate_schema.predicate2idx[p] for p in list(zip(*spo_list))[1]]
        entities = [self.clean_str(s) for spo in spo_list for s in (spo[0], spo[2])]
        tokenized_entities = self.tokenizer(entities)['input_ids']

        subject_spans, object_spans = [], []
        unicoded_text = ''.join(map(chr, text_encoded))
        for i in range(0, len(tokenized_entities), 2):
            subject_encoded = tokenized_entities[i][1:-1]
            object_encoded = tokenized_entities[i + 1][1:-1]
            subject_start = unicoded_text.find(''.join(map(chr, subject_encoded)))
            object_start = unicoded_text.find(''.join(map(chr, object_encoded)))

            if -1 in (subject_start, object_start):
                raise ValueError('Substring not found.')

            subject_spans.append((subject_start, subject_start + len(subject_encoded)))
            object_spans.append((object_start, object_start + len(object_encoded)))

        return predicates, subject_spans, object_spans

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, int, int, int, int, int]:
        """

        Args:
            idx: Index.

        Returns:
            Encoded text tensor, predicate and subject object position
                tensor
        """
        input_ids = torch.LongTensor(self.input_ids[idx])
        if self.train:
            r = random.randrange(len(self.predicates[idx]))
            predicate = self.predicates[idx][r]
            subject_start, subject_end = self.subject_spans[idx][r]
            object_start, object_end = self.object_spans[idx][r]
        else:
            predicate = self.predicates[idx]
            subject_start, subject_end = self.subject_spans[idx]
            object_start, object_end = self.object_spans[idx]

        return input_ids, predicate, subject_start, subject_end, object_start, object_end

    @staticmethod
    def padding_collate(batch: List[Tuple[torch.LongTensor, int, int, int, int, int]]) \
            -> Dict[str, Union[torch.LongTensor, Dict[str, torch.LongTensor]]]:
        """
        Collate function generates tensors with a batch of data.

        Args:
            batch: Batch of data.

        Returns:
            input_ids: batch_size, pad_len
            attention_mask: batch_size, pad_len
            targets: batch_size
        """
        input_ids, predicate, subject_start, subject_end, object_start, object_end = zip(*batch)
        attention_mask = list(map(torch.ones_like, input_ids))
        return {
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=0),
            'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=0),
            'targets': {
                'predicate': torch.LongTensor(predicate),
                'subject_start': torch.LongTensor(subject_start),
                'subject_end': torch.LongTensor(subject_end),
                'object_start': torch.LongTensor(object_start),
                'object_end': torch.LongTensor(object_end)
            }
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
