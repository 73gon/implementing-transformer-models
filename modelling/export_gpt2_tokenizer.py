# export_gpt2_tokenizer.py
from bpe_tokenizer import BPETokenizer
from datasets import load_dataset
from pathlib import Path
import json
from transformers import GPT2TokenizerFast

SPECIALS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
}

def save_gpt2_tokenizer(bpe, out_dir="bpe_tok_gpt2"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab = {tok: i for i, tok in enumerate(bpe.vocab)}

    # add special tokens
    for tok in SPECIALS.values():
        if tok not in vocab:
            vocab[tok] = len(vocab)

    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in bpe.merges:
            f.write(f"{a} {b}\n")

    with open(out / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(SPECIALS, f)

    with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_max_length": 128,
            "bos_token": SPECIALS["bos_token"],
            "eos_token": SPECIALS["eos_token"],
            "unk_token": SPECIALS["unk_token"],
            "pad_token": SPECIALS["pad_token"],
        }, f)

    return str(out)


if __name__ == "__main__":
    # load cleaned dataset (assuming you already saved it or have it in memory)
    from data_cleaning import cleaned_ds

    corpus = []
    for split in ["train", "validation"]:
        corpus.extend(cleaned_ds[split]["de"][:500]) # type: ignore
        corpus.extend(cleaned_ds[split]["en"][:500]) # type: ignore

    print("Training BPE tokenizer...")
    bpe = BPETokenizer(vocab_size=8000)
    bpe.train(corpus)

    path = save_gpt2_tokenizer(bpe)
    print(f"Tokenizer saved to: {path}")

    tok = GPT2TokenizerFast.from_pretrained(path, padding_side="right")
    tok.add_special_tokens(SPECIALS)

    print("Quick test:", tok.encode("Hallo Welt!"))
