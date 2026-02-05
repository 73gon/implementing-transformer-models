"""
Evaluation script for the Transformer translation model.

Implements:
1. Autoregressive generation with greedy decoding
2. Translation generation for WMT17 test set
3. BLEU score evaluation
4. Translation analysis
"""

import logging
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from data.data_cleaning import cleaned_ds
from model.model import Transformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TranslationGenerator:
    """
    Autoregressive translation generator using greedy decoding.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: GPT2TokenizerFast,
        device: torch.device,
        max_len: int = 128,
    ):
        """
        Args:
            model: Trained Transformer model
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run inference on
            max_len: Maximum generation length
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

        # Get special token IDs
        self.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
        self.bos_id = tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_id = tokenizer.convert_tokens_to_ids("[EOS]")

    def encode_source(self, text: str) -> torch.Tensor:
        """
        Encode the source sentence.

        Args:
            text: Source text (German)

        Returns:
            Tensor of token IDs (1, src_len)
        """
        # Tokenize and add special tokens
        ids = self.tokenizer.encode(text.lower().strip(), add_special_tokens=False)
        ids = ids[: self.max_len - 2]  # Leave room for BOS and EOS
        ids = [self.bos_id] + ids + [self.eos_id]

        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def generate_greedy(self, source_text: str) -> Tuple[str, List[int]]:
        """
        Generate translation using greedy decoding.

        Procedure:
        1. Encode the source sentence using the encoder
        2. Initialize decoder with BOS token
        3. Generate tokens one by one until EOS or max_len

        Args:
            source_text: Source text to translate (German)

        Returns:
            Tuple of (translated text, list of generated token IDs)
        """
        # Step 1: Encode source sentence
        src_ids = self.encode_source(source_text)  # (1, src_len)

        # Get encoder output (we'll cache this for efficiency)
        encoder_mask = (src_ids != self.pad_id).float()
        encoder_embeddings = self.model.encoder_embedding(src_ids)
        encoder_embeddings = self.model.encoder_positional_encoding(encoder_embeddings)
        encoder_output = self.model.encoder(encoder_embeddings, attention_mask=encoder_mask)

        # Step 2: Initialize decoder with BOS token
        generated_ids = [self.bos_id]

        # Step 3 & 4: Generate tokens autoregressively
        for _ in range(self.max_len - 1):
            # Prepare decoder input
            decoder_input = torch.tensor([generated_ids], dtype=torch.long, device=self.device)  # (1, current_len)

            # Decoder forward pass
            decoder_mask = torch.ones(1, len(generated_ids), dtype=torch.float, device=self.device)
            decoder_embeddings = self.model.decoder_embedding(decoder_input)
            decoder_embeddings = self.model.decoder_positional_encoding(decoder_embeddings)

            decoder_output = self.model.decoder(
                hidden_states=decoder_embeddings,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_mask,
                attention_mask=decoder_mask,
            )

            # Get logits for the last position
            logits = self.model.output_projection(decoder_output[:, -1, :])  # (1, vocab_size)

            # Greedy decoding: select token with highest probability
            next_token_id = logits.argmax(dim=-1).item()
            generated_ids.append(next_token_id)

            # Stop if EOS token is generated
            if next_token_id == self.eos_id:
                break

        # Decode generated IDs to text (exclude BOS and EOS)
        output_ids = generated_ids[1:]  # Remove BOS
        if output_ids and output_ids[-1] == self.eos_id:
            output_ids = output_ids[:-1]  # Remove EOS

        # Decode using space-aware method for this tokenizer
        translated_text = self._decode_with_spaces(output_ids)

        return translated_text, generated_ids

    def _decode_with_spaces(self, token_ids: List[int]) -> str:
        """
        Decode token IDs with proper BPE merging.

        This uses the tokenizer's built-in BPE merging to combine
        subword tokens into complete words.
        """
        if not token_ids:
            return ""

        # Use tokenizer's decode which properly merges BPE tokens
        # clean_up_tokenization_spaces=True handles GPT2's Ġ (space) markers
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return text.strip()

    def _add_word_spaces(self, text: str) -> str:
        """
        Add spaces between words for BPE tokenizer output.
        The tokenizer concatenates subwords without spaces, so we need to
        intelligently add them back.
        """
        import re

        # Add space before uppercase letters (except at start)
        text = re.sub(r"(?<!^)(?<![.\s])([A-Z])", r" \1", text)

        # Add space after punctuation if followed by letter
        text = re.sub(r"([.,!?;:])([a-zA-Z])", r"\1 \2", text)

        # Add space before numbers if preceded by letters
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)

        # Add space after numbers if followed by letters
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _clean_text(self, text: str) -> str:
        """Clean up decoded text by fixing spacing around punctuation."""
        import re

        # Remove space before punctuation
        text = re.sub(r"\s+([.,!?;:)])", r"\1", text)
        # Remove space after opening brackets
        text = re.sub(r"([(])\s+", r"\1", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def translate_batch(
        self,
        source_texts: List[str],
        show_progress: bool = True,
    ) -> List[str]:
        """
        Translate a batch of source texts.

        Args:
            source_texts: List of source texts to translate
            show_progress: Whether to show progress bar

        Returns:
            List of translated texts
        """
        translations = []

        iterator = tqdm(source_texts, desc="Translating") if show_progress else source_texts

        for text in iterator:
            translation, _ = self.generate_greedy(text)
            translations.append(translation)

        return translations


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
) -> dict:
    """
    Compute BLEU score using HuggingFace evaluate.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with BLEU score and related metrics
    """
    from evaluate import load

    bleu = load("bleu")

    # BLEU expects references as list of lists (multiple references per prediction)
    references_formatted = [[ref] for ref in references]

    results = bleu.compute(predictions=predictions, references=references_formatted)

    return results


def analyze_translations(
    source_texts: List[str],
    predictions: List[str],
    references: List[str],
    num_samples: int = 10,
) -> None:
    """
    Analyze and display sample translations.

    Args:
        source_texts: Source texts (German)
        predictions: Model predictions
        references: Reference translations
        num_samples: Number of samples to display
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRANSLATION ANALYSIS")
    logger.info("=" * 80)

    for i in range(min(num_samples, len(source_texts))):
        logger.info(f"\n--- Sample {i + 1} ---")
        logger.info(f"Source (DE):     {source_texts[i]}")
        logger.info(f"Reference (EN):  {references[i]}")
        logger.info(f"Prediction (EN): {predictions[i]}")

        # Simple error analysis
        ref_words = set(references[i].lower().split())
        pred_words = set(predictions[i].lower().split())

        missing = ref_words - pred_words
        extra = pred_words - ref_words

        if missing:
            logger.info(f"Missing words:   {missing}")
        if extra:
            logger.info(f"Extra words:     {extra}")

    logger.info("\n" + "=" * 80)


def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    d_model: int = 256,
    n_heads: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,  # No dropout during inference
    max_len: int = 128,
    pad_id: int = 0,
    device: torch.device = None,
) -> Transformer:
    """
    Load a trained model from checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_id=pad_id,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model


def main():
    """
    Main evaluation script.
    """
    # Configuration
    CHECKPOINT_PATH = "./checkpoints/best_model.pt"
    TEST_SUBSET_SIZE = 500  # Evaluate on subset for speed
    NUM_ANALYSIS_SAMPLES = 10

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")

    # Initialize tokenizer
    SPECIALS = {"pad_token": "[PAD]", "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
    tokenizer = GPT2TokenizerFast.from_pretrained("bpe_tok_gpt2", padding_side="right")
    tokenizer.add_special_tokens(SPECIALS)

    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        vocab_size=len(tokenizer),
        d_model=256,
        n_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.0,  # No dropout during evaluation
        max_len=128,
        pad_id=pad_id,
        device=device,
    )

    # Create generator
    generator = TranslationGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=128,
    )

    # Load test data
    logger.info("Loading test data...")
    test_data = cleaned_ds["test"]
    test_subset = test_data.select(range(min(TEST_SUBSET_SIZE, len(test_data))))

    source_texts = [example["de"] for example in test_subset]
    reference_texts = [example["en"] for example in test_subset]

    logger.info(f"Evaluating on {len(source_texts)} test examples\n")

    # Generate translations
    logger.info("Generating translations...")
    predictions = generator.translate_batch(source_texts, show_progress=True)

    # Compute BLEU score
    logger.info("\nComputing BLEU score...")
    bleu_results = compute_bleu_score(predictions, reference_texts)

    logger.info("\n" + "=" * 80)
    logger.info("BLEU SCORE RESULTS")
    logger.info("=" * 80)
    logger.info(f"BLEU Score: {bleu_results['bleu'] * 100:.2f}")
    logger.info(f"Precisions: {[f'{p:.4f}' for p in bleu_results['precisions']]}")
    logger.info(f"Brevity Penalty: {bleu_results['brevity_penalty']:.4f}")
    logger.info(f"Length Ratio: {bleu_results['length_ratio']:.4f}")
    logger.info("=" * 80)

    # Analyze sample translations
    analyze_translations(
        source_texts=source_texts,
        predictions=predictions,
        references=reference_texts,
        num_samples=NUM_ANALYSIS_SAMPLES,
    )

    # Save results to file
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test samples: {len(source_texts)}\n")
        f.write(f"BLEU Score: {bleu_results['bleu'] * 100:.2f}\n")
        f.write(f"Precisions: {bleu_results['precisions']}\n")
        f.write(f"Brevity Penalty: {bleu_results['brevity_penalty']:.4f}\n\n")

        f.write("SAMPLE TRANSLATIONS\n")
        f.write("-" * 80 + "\n\n")

        for i in range(min(50, len(source_texts))):
            f.write(f"Sample {i + 1}:\n")
            f.write(f"  Source (DE):     {source_texts[i]}\n")
            f.write(f"  Reference (EN):  {reference_texts[i]}\n")
            f.write(f"  Prediction (EN): {predictions[i]}\n\n")

    logger.info("\nResults saved to evaluation_results.txt")


if __name__ == "__main__":
    main()
