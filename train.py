import logging
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.data_cleaning import cleaned_ds
from data.dataset_translation import TranslationDataset, collate_translation
from model.lr_scheduler import TransformerLRScheduler
from model.model import Transformer
from model.optimiser import create_optimizer_with_weight_decay

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TransformerTrainer:
    """
    Trainer class for the Transformer model.

    Handles training loop, validation, checkpointing, and loss tracking.
    """

    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        scheduler: TransformerLRScheduler,
        device: torch.device,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: Transformer model instance
            optimizer: AdamW optimizer
            scheduler: Learning rate scheduler
            device: Device to train on (cuda/cpu)
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        # Loss function with label smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=0.1,
            reduction="mean",
        )

        # Training tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        epoch: int,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_dataloader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        logger.info(f"Epoch {epoch + 1} - Training")
        start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            src_ids = batch["src_ids"].to(self.device)
            tgt_in = batch["tgt_in"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            logits = self.model(src_ids, tgt_in)  # (B, tgt_len, vocab_size)

            # Calculate loss
            logits_reshaped = logits.reshape(-1, logits.size(-1))
            labels_reshaped = labels.reshape(-1)

            # Calculate loss only on valid tokens
            valid_mask = labels_reshaped != -100
            if valid_mask.sum() > 0:
                loss = self.loss_fn(logits_reshaped[valid_mask], labels_reshaped[valid_mask])
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            # Update weights
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                logger.info(f"  Batch {batch_idx + 1:4d} | Loss: {avg_loss:7.4f} | LR: {self.scheduler.get_lr():.2e} | Time: {elapsed:6.1f}s")

        if num_batches == 0:
            avg_loss = 0.0
        else:
            avg_loss = total_loss / num_batches

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s\n")

        return avg_loss

    def validate(
        self,
        val_dataloader: DataLoader,
    ) -> float:
        """
        Validate the model.

        Args:
            val_dataloader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                src_ids = batch["src_ids"].to(self.device)
                tgt_in = batch["tgt_in"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                logits = self.model(src_ids, tgt_in)

                logits_reshaped = logits.reshape(-1, logits.size(-1))
                labels_reshaped = labels.reshape(-1)

                valid_mask = labels_reshaped != -100
                if valid_mask.sum() > 0:
                    loss = self.loss_fn(logits_reshaped[valid_mask], labels_reshaped[valid_mask])
                    total_loss += loss.item()
                    num_batches += 1

        if num_batches == 0:
            avg_loss = float("inf")
        else:
            avg_loss = total_loss / num_batches

        logger.info(f"Validation Loss: {avg_loss:.4f}\n")

        return avg_loss

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = "./checkpoints/",
    ) -> Dict[str, List[float]]:
        """
        Full training loop for multiple epochs.

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary with training and validation losses
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info("=" * 80 + "\n")

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate(val_dataloader)
            self.val_losses.append(val_loss)

            # Save checkpoint if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_step": self.scheduler.current_step,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved best model to {checkpoint_path}\n")

            # Log epoch summary
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss:   {val_loss:.4f}")
            logger.info(f"  Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch + 1})")
            logger.info("-" * 80 + "\n")

        logger.info("=" * 80)
        logger.info("Training Complete")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
        logger.info("=" * 80)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }


def create_transformer_model(
    vocab_size: int,
    d_model: int = 256,
    n_heads: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    max_len: int = 128,
    pad_id: int = 0,
) -> Transformer:
    """
    Create a transformer model with specified parameters.
    """
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

    logger.info("Created Transformer model:")
    logger.info(f"  vocab_size: {vocab_size}")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  n_heads: {n_heads}")
    logger.info(f"  num_encoder_layers: {num_encoder_layers}")
    logger.info(f"  num_decoder_layers: {num_decoder_layers}")
    logger.info(f"  dim_feedforward: {dim_feedforward}")
    logger.info(f"  dropout: {dropout}")
    logger.info(f"  max_len: {max_len}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    return model


def load_wmt17_data(
    val_size: int = 5000,
    train_subset_size: int | None = None,
):
    """
    Load and prepare the WMT17 De-En translation dataset.
    """
    from transformers import GPT2TokenizerFast

    logger.info("Loading WMT17 dataset...")

    # Initialize tokenizer with special tokens
    SPECIALS = {"pad_token": "[PAD]", "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"}

    tokenizer = GPT2TokenizerFast.from_pretrained("bpe_tok_gpt2", padding_side="right")
    tokenizer.add_special_tokens(SPECIALS)

    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Split dataset
    train_data = cleaned_ds["train"]
    test_data = cleaned_ds["test"]

    # Use subset of training data if specified
    if train_subset_size is not None:
        train_data = train_data.select(range(min(train_subset_size, len(train_data))))
        logger.info(f"Using {len(train_data)} training examples (subset)")

    # Use first val_size examples from test set for validation
    val_data = test_data.select(range(min(val_size, len(test_data))))
    logger.info(f"Using {len(val_data)} validation examples")

    # Create datasets
    train_dataset = TranslationDataset(train_data, tokenizer, max_len=128)
    val_dataset = TranslationDataset(val_data, tokenizer, max_len=128)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}\n")

    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-7
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_STEPS = 1000
    TRAIN_SUBSET_SIZE = 1000000
    VAL_SIZE = 10000
    GRAD_CLIP = 1.0

    D_MODEL = 256
    N_HEADS = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    MAX_LEN = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}\n")

    # Load data
    train_dataset, val_dataset, tokenizer = load_wmt17_data(
        val_size=VAL_SIZE,
        train_subset_size=TRAIN_SUBSET_SIZE,
    )

    # Create dataloaders
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_translation(b, pad_id),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_translation(b, pad_id),
    )

    # Create model with improved hyperparameters
    model = create_transformer_model(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LEN,
        pad_id=pad_id,
    )

    # Create optimizer
    optimizer = create_optimizer_with_weight_decay(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Create scheduler
    scheduler = TransformerLRScheduler(
        optimizer=optimizer,
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
    )

    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=GRAD_CLIP,
    )

    # Train
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=NUM_EPOCHS,
    )
