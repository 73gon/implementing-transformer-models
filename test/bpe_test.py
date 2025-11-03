import os, sys, re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelling.bpe_tokenizer import BPETokenizer

corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]
text = "Machine learning is a subset of artificial intelligence."

corpus_norm = [re.sub(r"[^A-Za-z\s]", "", s).lower() for s in corpus]

tok = BPETokenizer(vocab_size=64)
tok.train(corpus_norm)

print(f"Merges: {len(tok.merges)}")
print(f"Vocab size: {len(tok.vocab)}")

print(tok.encode(text))

print()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()  # type: ignore

trainer = BpeTrainer(
    vocab_size=295
)

tokenizer.train_from_iterator(corpus, trainer=trainer)

enc = tokenizer.encode("Machine learning is a subset of artificial intelligence.")
print(len(tokenizer.get_vocab()))    # ~295
print(enc.tokens)



