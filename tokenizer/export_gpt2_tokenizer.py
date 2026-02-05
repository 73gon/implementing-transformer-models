# export_gpt2_tokenizer.py
import json
import sys
from pathlib import Path

from bpe_tokenizer import BPETokenizer
from transformers import GPT2TokenizerFast

SPECIALS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
}

ROOT_DIR = Path(__file__).parent.parent

# Add root directory to Python path so we can import from root-level modules
sys.path.insert(0, str(ROOT_DIR))


def save_gpt2_tokenizer(bpe, out_dir="bpe_tok_gpt2"):
    # Use root directory for tokenizer
    out = ROOT_DIR / out_dir
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
        json.dump(
            {
                "tokenizer_class": "GPT2Tokenizer",
                "model_max_length": 256,
                "do_lower_case": False,
                "bos_token": SPECIALS["bos_token"],
                "eos_token": SPECIALS["eos_token"],
                "unk_token": SPECIALS["unk_token"],
                "pad_token": SPECIALS["pad_token"],
            },
            f,
            indent=2,
        )

    return str(out)


if __name__ == "__main__":
    # load cleaned dataset (assuming you already saved it or have it in memory)
    from data.data_cleaning import cleaned_ds

    # Use more data for better BPE merges
    corpus = []

    # Use more training data for better tokenizer
    max_samples = 5000  # Increased from 500
    for split in ["train"]:  # Focus on training split
        corpus.extend(cleaned_ds[split]["de"][:max_samples])  # type: ignore
        corpus.extend(cleaned_ds[split]["en"][:max_samples])  # type: ignore

    # Add validation data too
    if "validation" in cleaned_ds:
        corpus.extend(cleaned_ds["validation"]["de"][:max_samples])
        corpus.extend(cleaned_ds["validation"]["en"][:max_samples])

    print(f"Training BPE tokenizer on {len(corpus)} texts...")
    bpe = BPETokenizer(vocab_size=8000)
    bpe.train_with_space_markers(corpus)

    path = save_gpt2_tokenizer(bpe)
    print(f"Tokenizer saved to: {path}")

    tok = GPT2TokenizerFast.from_pretrained(path, padding_side="right")
    tok.add_special_tokens(SPECIALS)

    print("\nQuick tests:")
    test_cases = [
        "Hallo Welt!",
        "Louis Galicia said",
        "the quick brown fox",
    ]
    for text in test_cases:
        tokens = tok.encode(text)
        decoded = tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"  Input:   '{text}'")
        print(f"  Encoded: {tokens}")
        print(f"  Decoded: '{decoded}'")
        print()
