from transformers import GPT2TokenizerFast
from modelling.dataset_translation import TranslationDataset, collate_translation
from modelling.data_cleaning import cleaned_ds

from torch.utils.data import DataLoader

SPECIALS = {"pad_token":"[PAD]","unk_token":"[UNK]","bos_token":"[BOS]","eos_token":"[EOS]"}

# 1. Tokenizer laden
tok = GPT2TokenizerFast.from_pretrained("bpe_tok_gpt2", padding_side="right")
tok.add_special_tokens(SPECIALS)
pad_id = tok.convert_tokens_to_ids("[PAD]")

# 2. Dataset und Dataloader erstellen
train_ds = TranslationDataset(cleaned_ds["train"], tok, max_len=128)
train_dl = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda b: collate_translation(b, pad_id),
    num_workers=0,
)

# 3. Einen Batch prüfen
batch = next(iter(train_dl))
for k, v in batch.items():
    print(k, v.shape)

# --- Test: Positional Encoding auf einen Batch anwenden ---
from modelling.positional_encoding import PositionalEncoding
import torch.nn as nn

d_model = 128
emb = nn.Embedding(tok.vocab_size, d_model, padding_idx=pad_id)
pe  = PositionalEncoding(d_model)

enc_in = pe(emb(batch["src_ids"]))   # (B, Ls, d_model)

print("PE-Test OK, Shape:", enc_in.shape)

from modelling.embeddings import TokenEmbedding

d_model = 128
emb = TokenEmbedding(tok.vocab_size, d_model, pad_id)

emb_out = emb(batch["src_ids"])  # (B, L, d_model)
print("Embedding-Test OK, Shape:", emb_out.shape)


