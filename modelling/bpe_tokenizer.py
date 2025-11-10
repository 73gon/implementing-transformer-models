class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = set()

    def train(self, corpus):
        print(f"[DEBUG] Starting BPE training with vocab_size={self.vocab_size}")

        for line in corpus:
            for ch in line:
                if ch != ' ':
                    self.vocab.add(ch)
        print(f"[DEBUG] Initial vocab size (characters): {len(self.vocab)}")
        print(f"[DEBUG] Initial vocab (characters): {self.vocab}")

        words_sym = []
        for line in corpus:
            for word in line.split():
                symbols = list(word)
                words_sym.append(symbols)
        print(f"[DEBUG] Total words to process: {len(words_sym)}")

        iteration = 0
        while len(self.vocab) < self.vocab_size:
            iteration += 1
            pair_counts = {}
            for symbols in words_sym:
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                print(f"[DEBUG] No more pairs to merge, stopping at iteration {iteration}")
                break
            else:
                best_pair = max(pair_counts, key=lambda k: pair_counts[k])
                print(f"[DEBUG] Iteration {iteration}: Best pair='{best_pair}' with count={pair_counts[best_pair]}, vocab_size={len(self.vocab)}")

            new_symbols = []
            for symbols in words_sym:
                i = 0
                merged = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        merged.append(symbols[i] + symbols[i+1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                new_symbols.append(merged)

            words_sym = new_symbols
            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])

    def encode(self, text):
        # Split into words (same way as training)
        words = text.lower().split()

        tokens = []
        for word in words:
            # start with characters
            symbols = list(word)

            # apply all merges in the order learned
            for pair in self.merges:
                i = 0
                merged = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                        merged.append(symbols[i] + symbols[i + 1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                symbols = merged  # update symbols after each merge

            tokens.extend(symbols)
        return tokens

