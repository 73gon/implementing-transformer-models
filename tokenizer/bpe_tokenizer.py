class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = set()

    def train(self, corpus):
        """
        Train BPE tokenizer with proper space handling (like GPT2).
        Uses Ġ prefix to mark spaces in subword tokens.

        This method expects pre-tokenized corpus with space markers already added.
        For raw corpus, use train_with_space_markers() instead.
        """
        print(f"[DEBUG] Starting BPE training with vocab_size={self.vocab_size}")

        # Initialize vocab with characters AND special space marker (Ġ)
        for line in corpus:
            for ch in line:
                if ch != " ":
                    self.vocab.add(ch)

        # Add space marker (Ġ represents a space in GPT2 style)
        self.vocab.add("Ġ")

        print(f"[DEBUG] Initial vocab size (characters + space marker): {len(self.vocab)}")

        # Split into words with space prefix (GPT2 style)
        words_sym = []
        for line in corpus:
            for word in line.split():
                # Prefix word with space marker (Ġ) to indicate word boundary
                symbols = ["Ġ"] + list(word)
                words_sym.append(symbols)
        print(f"[DEBUG] Total words to process: {len(words_sym)}")

        iteration = 0
        while len(self.vocab) < self.vocab_size:
            iteration += 1
            pair_counts = {}
            for symbols in words_sym:
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                # print(f"[DEBUG] No more pairs to merge, stopping at iteration {iteration}")
                break
            else:
                best_pair = max(pair_counts, key=lambda k: pair_counts[k])
                if iteration % 100 == 0:
                    print(f"[DEBUG] Iteration {iteration}: vocab_size={len(self.vocab)}")

            new_symbols = []
            for symbols in words_sym:
                i = 0
                merged = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        merged.append(symbols[i] + symbols[i + 1])
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                new_symbols.append(merged)

            words_sym = new_symbols
            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])

    def train_with_space_markers(self, corpus):
        """
        Train BPE tokenizer on RAW corpus, automatically adding space markers (Ġ).

        This is the recommended method for training on unprocessed text.
        Internally converts corpus to GPT2 style with Ġ space markers before training.
        """
        print("[DEBUG] Converting raw corpus to space-marked format...")

        # Convert raw corpus to space-marked format (GPT2 style)
        # Each word gets prefixed with Ġ, and words are joined with spaces
        tokenized_corpus = []
        for text in corpus:
            words = text.split()
            if words:  # Only process non-empty lines
                tokenized_text = " ".join([f"Ġ{word}" for word in words])
                tokenized_corpus.append(tokenized_text)

        print(f"[DEBUG] Converted {len(tokenized_corpus)} texts to space-marked format")

        # Now train using the regular train method on the converted corpus
        self.train(tokenized_corpus)

    def encode(self, text):
        """
        Encode text to BPE tokens with proper space handling.
        Each word is prefixed with Ġ (space marker).
        """
        # Split into words (preserving spaces as markers)
        words = text.lower().split()

        tokens = []
        for word in words:
            # Start with space marker + characters (GPT2 style)
            symbols = ["Ġ"] + list(word)

            # Apply all merges in the order learned
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
