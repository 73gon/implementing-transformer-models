# dataset_translation.py
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TranslationDataset(Dataset):
    """
    Expects a HuggingFace split like cleaned_ds['train'] with columns 'de', 'en'
    and a GPT2TokenizerFast (shared vocab).
    """

    def __init__(
        self,
        hf_split,  # e.g., cleaned_ds["train"]
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 128,
    ):
        self.data = hf_split
        self.tok = tokenizer
        self.max_len = max_len

        # cache IDs for speed
        self.pad_id = self.tok.convert_tokens_to_ids("[PAD]")  # type:ignore
        self.bos_id = self.tok.convert_tokens_to_ids("[BOS]")  # type:ignore
        self.eos_id = self.tok.convert_tokens_to_ids("[EOS]")  # type:ignore

        # minimal sanity
        assert self.pad_id is not None and self.pad_id >= 0, "Missing [PAD] in tokenizer."
        assert self.bos_id is not None and self.eos_id is not None, "Missing BOS/EOS in tokenizer."

    def _encode_line(self, text: str) -> List[int]:
        ids = self.tok.encode(text.lower().strip(), add_special_tokens=False)
        ids = ids[: self.max_len - 2]
        return [self.bos_id] + ids + [self.eos_id]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        src = self.data[idx]["de"]
        tgt = self.data[idx]["en"]

        src_ids = torch.tensor(self._encode_line(src), dtype=torch.long)
        tgt_ids = torch.tensor(self._encode_line(tgt), dtype=torch.long)

        # Decoder input is shifted right (starts with BOS), labels are shifted left
        tgt_in = tgt_ids[:-1]
        tgt_out = tgt_ids[1:]

        return {
            "src_ids": src_ids,  # (Ls,)
            "tgt_in": tgt_in,  # (Lt-1,)
            "tgt_out": tgt_out,  # (Lt-1,)
        }


def _pad(list_of_tensors: List[torch.Tensor], pad_id: int):
    maxlen = max(t.size(0) for t in list_of_tensors)
    B = len(list_of_tensors)
    out = torch.full((B, maxlen), pad_id, dtype=torch.long)
    for i, t in enumerate(list_of_tensors):
        out[i, : t.size(0)] = t
    mask = (out != pad_id).long()  # 1=token, 0=pad
    return out, mask


def collate_translation(batch: List[Dict[str, torch.Tensor]], pad_id: int):
    src_ids = [b["src_ids"] for b in batch]
    tgt_in = [b["tgt_in"] for b in batch]
    tgt_out = [b["tgt_out"] for b in batch]

    src_ids, src_mask = _pad(src_ids, pad_id)
    tgt_in, tgt_mask = _pad(tgt_in, pad_id)
    tgt_out, _ = _pad(tgt_out, pad_id)

    # Labels should ignore padding in the loss
    labels = tgt_out.masked_fill(tgt_mask == 0, -100)

    return {
        "src_ids": src_ids,  # (B, Ls)
        "src_mask": src_mask,  # (B, Ls)
        "tgt_in": tgt_in,  # (B, Lt)
        "tgt_mask": tgt_mask,  # (B, Lt)
        "labels": labels,  # (B, Lt) with -100 on pad
    }
