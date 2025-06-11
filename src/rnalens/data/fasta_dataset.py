from typing import TypedDict, TYPE_CHECKING, Optional, Union

from Bio import SeqIO
import torch
from torch.utils.data import Dataset
import math
import random

from .alphabet import Alphabet

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# --- type hints ---
class PretrainData(TypedDict):
    input_ids: torch.Tensor # the masked tokenized sequence
    attention_mask: torch.Tensor # the attention mask
    labels: torch.Tensor # the labels of the masked tokens
    minimum_free_energies: torch.Tensor # the label of the sequence, shape=1x1
    secondary_structure_labels: torch.Tensor # the label of the sequence

class RawPretrainData(TypedDict):
    sequence: str
    minimum_free_energy: float
    secondary_structure: str

# --- util ---
def mask_sequence(
    input_ids: torch.Tensor, 
    mask_prob: float, 
    pad_token_id: int,
    mask_token_id: int,
    tokenizer_vocab_size: int = -1, # not used.
    ) -> torch.Tensor:
    """
    Mask the sequence with the given probability.
    
    Used for the mlm task.
    """
    assert input_ids.dim() == 1, f"input_ids should be of dimension 1, bot got {input_ids.dim()}"
    
    # input_ids_mask = torch.zeros_like(input_ids)
    labels = torch.empty_like(input_ids).fill_(-100)

    # Get all non-pad token positions
    eligible_indices = torch.where(input_ids != pad_token_id)[0]
    num_eligible = len(eligible_indices)

    # Calculate number of tokens to mask (at least 1)
    num_to_mask = max(1, math.floor(num_eligible * mask_prob))
    # Randomly select tokens to mask
    mask_indices = eligible_indices[torch.randperm(num_eligible)[:num_to_mask]]
    
    labels[mask_indices] = input_ids[mask_indices]
    
    if tokenizer_vocab_size == -1:
        input_ids[mask_indices] = mask_token_id
    else:
        # 80% of the time, replace with [MASK]
        # 10% of the time, replace with random token
        # 10% of the time, keep the same token
        # utlize the mask_indices to create a mask
        mask_indices_80 = mask_indices[:int(num_to_mask * 0.8)]
        mask_indices_10 = mask_indices[int(num_to_mask * 0.8):int(num_to_mask * 0.9)]
        mask_indices_10_10 = mask_indices[int(num_to_mask * 0.9):]
        
        input_ids[mask_indices_80] = mask_token_id
        input_ids[mask_indices_10] = torch.randint(0, tokenizer_vocab_size, input_ids[mask_indices_10].shape)
        input_ids[mask_indices_10_10] = input_ids[mask_indices_10_10]
    
    return input_ids, labels

# --- dataset ---
class PretrainDataset(Dataset):
    """
    A dataset for pretraining rnalens containing raw sequences and labels
    
    """
    STRUCT_TOK_TO_IDX = {
        "(": 0,
        ".": 1,
        ")": 2,
    }
    
    def __init__(self, 
                 fasta_file: str, 
                #  tokenizer: "PreTrainedTokenizer", 
                 alphabet: Optional[Alphabet] = None,
                 tokenizer: Optional["PreTrainedTokenizer"] = None,
                 mask_prob: float = 0.15, 
                 output_secondary_structure: bool = False,
                 max_len=1022, # be faithful to the original truncation param
                 ):
        assert alphabet is not None or tokenizer is not None, "Either alphabet or tokenizer should be provided"
        assert alphabet is None or tokenizer is None, "Either alphabet or tokenizer should be provided, not both"
        assert mask_prob >= 0 and mask_prob <= 1, "mask_prob should be between 0 and 1"
        assert max_len > 0, "max_len should be greater than 0"

        self._alphabet = alphabet
        self._tokenizer = tokenizer
        self._mask_prob = mask_prob
        self._max_len = max_len
        self._output_secondary_structure = output_secondary_structure
        
        self._data: list[RawPretrainData] = [
            data for data in self._load_fasta_dataset(fasta_file)
        ]
        
        self._padding_idx = self._alphabet.padding_idx if self._alphabet else self._tokenizer.pad_token_id
        self._mask_idx = self._alphabet.mask_idx if self._alphabet else self._tokenizer.mask_token_id
        self._vocab_size = len(self._alphabet) if self._alphabet else self._tokenizer.vocab_size

    def _load_fasta_dataset(self, fasta_file: str):
        """
        Load the fasta dataset from the given file.
        """
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    comments: list[str] = line.strip('>').split('|')
                    supervised_label = float(comments[-1]) # the data format is like this: >...|...|...|label
                    secondary_structure = comments[1]
                    if len(secondary_structure) > self._max_len:
                        secondary_structure = secondary_structure[:self._max_len]
                    
                    sequence = next(f).strip()
                    
                    yield {
                        "sequence": sequence,
                        "minimum_free_energy": supervised_label,
                        "secondary_structure": secondary_structure,
                    }

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx)->PretrainData:
        raw_data: RawPretrainData = self._data[idx]
        
        if self._alphabet is None:
            out: list[int] = self._tokenizer.encode(raw_data["sequence"], add_special_tokens=False)
        else:
            out: list[int] = self._alphabet.encode(raw_data["sequence"])
        if len(out) > self._max_len:
            out = out[:self._max_len]
        
        # TODO: squeeze is not necessary
        input_ids = torch.full(size=(self._max_len,1), fill_value = self._padding_idx, dtype=torch.long).squeeze(1)
        attention_mask = torch.full(size=(self._max_len,1), fill_value=0).squeeze(1)
        secondary_structure_labels = None
        if self._output_secondary_structure:
            secondary_structure_labels = torch.full(size=(self._max_len,1), fill_value=-100, dtype=torch.long).squeeze(1)
            for idx, char in enumerate(raw_data["secondary_structure"]):
                secondary_structure_labels[idx] = self.STRUCT_TOK_TO_IDX[char]
        
        for idx, num in enumerate(out):
            input_ids[idx] = num
            attention_mask[idx] = 1
        
        input_ids, labels = mask_sequence(input_ids, 
                                  mask_prob=self._mask_prob, 
                                  pad_token_id=self._padding_idx,
                                  mask_token_id=self._mask_idx,
                                  tokenizer_vocab_size=self._vocab_size if self._tokenizer else -1, # when using alphabet, this is not used
                                  )
        
        ret = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "minimum_free_energies": torch.tensor(raw_data["minimum_free_energy"], dtype=torch.float),
        }
        
        if self._output_secondary_structure:
            ret["secondary_structure_labels"] = secondary_structure_labels
        return ret
