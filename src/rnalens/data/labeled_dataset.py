import math
import random
from typing import Optional, TYPE_CHECKING

import torch

from .alphabet import Alphabet

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

class FinetuneDataset(object):
    def __init__(self, 
                 sequence_labels, 
                 sequence_strs, 
                 mask_prob = 0.15,
                 alphabet: Alphabet = None,
                 tokenizer: Optional["PreTrainedTokenizer"] = None,
                 clean_seq = False,
                 ):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        if clean_seq:
            self.sequence_strs = [
                seq.replace(" ", "").replace("<pad>", "")
                for seq in self.sequence_strs
            ]
        
        self.mask_prob = mask_prob
        self._alphabet = alphabet
        self._tokenizer = tokenizer
        
        if alphabet is None:
            assert tokenizer is not None, "Either alphabet or tokenizer should be provided"
            actual_max_len = max(
                [len(tokenizer.encode(seq, add_special_tokens=False)) for seq in self.sequence_strs]
            )
            self._padding_idx = tokenizer.pad_token_id
            self._mask_idx = tokenizer.mask_token_id
            self._vocab_size = tokenizer.vocab_size
        else:
            assert alphabet is not None, "Either alphabet or tokenizer should be provided"
            actual_max_len = max(
                [len(self._alphabet.encode(seq)) for seq in self.sequence_strs]
            )
            self._padding_idx = alphabet.padding_idx
            self._mask_idx = alphabet.mask_idx
            self._vocab_size = len(alphabet)

        print("[info] the maximum length of sequence=", actual_max_len)
        self._max_len = actual_max_len

    def __len__(self):
        return len(self.sequence_labels)
    
    def __getitem__(self, idx):
        sequence_str = self.sequence_strs[idx]
        sequence_label = self.sequence_labels[idx]

        if self._alphabet is None:
            tokenized_str: list[int] = self._tokenizer.encode(sequence_str, add_special_tokens=False)
        else:
            tokenized_str: list[int] = self._alphabet.encode(sequence_str)
        if len(tokenized_str) > self._max_len:
            tokenized_str = tokenized_str[:self._max_len]

        input_ids = torch.full(size=(self._max_len,1), fill_value = self._padding_idx, dtype=torch.long).squeeze(1)
        attention_mask = torch.full(size=(self._max_len,1), fill_value=0).squeeze(1)

        for idx, num in enumerate(tokenized_str):
            input_ids[idx] = num
            attention_mask[idx] = 1

        # masked_indices, masked_sequence_str = self.mask_sequence(sequence_str)
        # return sequence_label, sequence_str
        return {
            "labels":  torch.FloatTensor([sequence_label]),
            "sequence_str": sequence_str,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        """
        The function `get_batch_indices` takes a list of sequence lengths and groups them into batches based
        on a target token count per batch.
        
        :param toks_per_batch: The `toks_per_batch` parameter represents the maximum number of tokens
        allowed in each batch. This function `get_batch_indices` is designed to create batches of sequences
        based on their lengths while considering the `extra_toks_per_seq` parameter. The function sorts the
        sequences by length, then groups them
        :param extra_toks_per_seq: The `extra_toks_per_seq` parameter in the `get_batch_indices` function
        represents the additional number of tokens per sequence that should be considered when calculating
        the size of each sequence. This parameter allows you to account for any extra tokens that may be
        present in each sequence beyond its actual length. By, defaults to 0 (optional)
        :return: Batches of indices representing sequences grouped together based on the total number of
        tokens per batch and additional tokens per sequence.
        """
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches