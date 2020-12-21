from pathlib import Path
import json
import random
from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

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


class SPOExtractionDataset(Dataset):

    predicate_schema = PredicateSchema.from_file()

    def __init__(self, data: List[Dict[str, Union[str, List[List[str]]]]], tokenizer: PreTrainedTokenizerFast,
                 train: bool = True):
        """
        Dataset for subject, predicate, object triplet extraction.
        Calling from_file constructor is recommended.

        Args:
            data: List of data entries.
            tokenizer: Tokenizer.
            train: Training mode. If set to True, a 'left join' on
                sentence is performed with one sampled triplet joined
                on each sentence. Otherwise, a 'right join' on triplet
                is performed with sentences joined on all triplets.
                Default: True
        """
        self.tokenizer = tokenizer
        self.train = train
        input_ids = tokenizer([self.clean_str(d['text']) for d in data])['input_ids']
        self.predicates, self.s_spans, self.o_spans = [], [], []
        self.encodings = input_ids if self.train else []
        for text_encoded, entry in zip(input_ids, data):
            try:
                predicates, s_spans, o_spans = self.process_entry(text_encoded, entry['spo_list'])
            except ValueError:
                continue
            if self.train:
                self.predicates.append(predicates)
                self.s_spans.append(s_spans)
                self.o_spans.append(o_spans)
            else:
                self.encodings.extend([text_encoded] * len(predicates))
                self.predicates.extend(predicates)
                self.s_spans.extend(s_spans)
                self.o_spans.extend(o_spans)

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
        s_and_o = [self.clean_str(s) for spo in spo_list for s in (spo[0], spo[2])]
        tokenized_so = self.tokenizer(s_and_o)['input_ids']

        s_spans, o_spans = [], []
        unicoded_text = ''.join(map(chr, text_encoded))
        for i in range(0, len(tokenized_so), 2):
            s_encoded = tokenized_so[i][1:-1]
            o_encoded = tokenized_so[i + 1][1:-1]
            s_start = unicoded_text.find(''.join(map(chr, s_encoded)))
            o_start = unicoded_text.find(''.join(map(chr, o_encoded)))

            if -1 in (s_start, o_start):
                raise ValueError('Substring not found.')

            s_spans.append((s_start, s_start + len(s_encoded) - 1))
            o_spans.append((o_start, o_start + len(o_encoded) - 1))

        return predicates, s_spans, o_spans

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, int, torch.Tensor]:
        """

        Args:
            idx: Index.

        Returns:
            Encoded text tensor, predicate and subject object position
                tensor
        """
        text_encoded = torch.LongTensor(self.encodings[idx])
        if self.train:
            r = random.randrange(len(self.predicates[idx]))
            predicate, s_span, o_span = self.predicates[idx][r], self.s_spans[idx][r], self.o_spans[idx][r]
        else:
            predicate, s_span, o_span = self.predicates[idx], self.s_spans[idx], self.o_spans[idx]

        so_pos = torch.zeros((4, len(text_encoded)))
        for i, pos in enumerate(s_span + o_span):
            so_pos[i, pos] = 1
        return text_encoded, predicate, so_pos
