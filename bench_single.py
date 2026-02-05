"""
Single benchmark configuration runner.
Called by bench_cpu_gpu_v2.py as a subprocess for isolation.

Usage:
    python bench_single.py <device> <seq_len> <batch_size> <repeat> <output_file>
"""

import csv
import os
import sys
import time
from datetime import datetime
from functools import partial
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from data.data_cleaning import cleaned_ds
from data.dataset_translation import TranslationDataset, collate_translation
from model.lr_scheduler import TransformerLRScheduler
from model.optimiser import create_optimizer_with_weight_decay
from train import create_transformer_model

# Configuration
WARMUP_STEPS = 10
TIMED_STEPS = 50
DATASET_SIZE = 5000

# Model hyperparameters
D_MODEL = 256
N_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1


def load_tokenizer():
    """Load tokenizer."""
    tokenizer = GPT2TokenizerFast.from_pretrained("bpe_tok_gpt2", padding_side="right")
    SPECIALS = {"pad_token": "[PAD]", "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
    tokenizer.add_special_tokens(SPECIALS)
    return tokenizer


def collate_with_seq_len(batch: List[Dict], pad_id: int, target_seq_len: int) -> Dict[str, torch.Tensor]:
    """Collate function that crops/pads sequences to a target length."""
    collated = collate_translation(batch, pad_id)

    def adjust_seq_len(tensor: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
        B, L = tensor.shape
        if L >= target_len:
            return tensor[:, :target_len]
        else:
            pad_tensor = torch.full((B, target_len - L), pad_value, dtype=tensor.dtype)
            return torch.cat([tensor, pad_tensor], dim=1)

    return {
        "src_ids": adjust_seq_len(collated["src_ids"], target_seq_len, pad_id),
        "src_mask": adjust_seq_len(collated["src_mask"], target_seq_len, 0),
        "tgt_in": adjust_seq_len(collated["tgt_in"], target_seq_len, pad_id),
        "tgt_mask": adjust_seq_len(collated["tgt_mask"], target_seq_len, 0),
        "labels": adjust_seq_len(collated["labels"], target_seq_len, -100),
    }


def create_dataloader(tokenizer, pad_id, batch_size, seq_len):
    """Create dataloader with fixed sequence length."""
    train_data = list(cleaned_ds["train"])[:DATASET_SIZE]
    dataset = TranslationDataset(train_data, tokenizer)
    collate_fn = partial(collate_with_seq_len, pad_id=pad_id, target_seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )


def training_step(model, optimizer, scheduler, loss_fn, batch, device):
    """Single training step."""
    src_ids = batch["src_ids"].to(device)
    tgt_in = batch["tgt_in"].to(device)
    labels = batch["labels"].to(device)

    logits = model(src_ids, tgt_in)
    logits_reshaped = logits.reshape(-1, logits.size(-1))
    labels_reshaped = labels.reshape(-1)

    valid_mask = labels_reshaped != -100
    if valid_mask.sum() > 0:
        loss = loss_fn(logits_reshaped[valid_mask], labels_reshaped[valid_mask])
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss.item()


def measure_cpu(model, optimizer, scheduler, loss_fn, dataloader, device):
    """Measure step times on CPU."""
    model.train()
    data_iter = iter(dataloader)
    step_times = []
    final_loss = 0.0

    for step in range(WARMUP_STEPS + TIMED_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if step >= WARMUP_STEPS:
            start = time.perf_counter()
            loss = training_step(model, optimizer, scheduler, loss_fn, batch, device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            step_times.append(elapsed_ms)
            final_loss = loss

    return step_times, final_loss, 0.0


def measure_cuda(model, optimizer, scheduler, loss_fn, dataloader, device):
    """Measure step times on CUDA."""
    model.train()
    data_iter = iter(dataloader)
    step_times = []
    final_loss = 0.0

    torch.cuda.reset_peak_memory_stats(device)

    for step in range(WARMUP_STEPS + TIMED_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if step >= WARMUP_STEPS:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            loss = training_step(model, optimizer, scheduler, loss_fn, batch, device)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            step_times.append(elapsed_ms)
            final_loss = loss

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return step_times, final_loss, peak_mem_mb


def run_benchmark(device_str, seq_len, batch_size, repeat):
    """Run a single benchmark configuration."""
    device = torch.device(device_str)

    # Load tokenizer
    tokenizer = load_tokenizer()
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    vocab_size = len(tokenizer)

    # Clear GPU if needed
    if device_str == "cuda":
        torch.cuda.empty_cache()

    status = "success"
    notes = ""

    try:
        # Create dataloader
        dataloader = create_dataloader(tokenizer, pad_id, batch_size, seq_len)

        if len(dataloader) < WARMUP_STEPS + TIMED_STEPS:
            notes = f"Limited batches: {len(dataloader)}"

        # Create model
        model = create_transformer_model(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            max_len=seq_len + 10,
            pad_id=pad_id,
        ).to(device)

        optimizer = create_optimizer_with_weight_decay(model, learning_rate=1e-4, weight_decay=0.01)
        scheduler = TransformerLRScheduler(optimizer, d_model=D_MODEL, warmup_steps=1000)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

        # Run benchmark
        if device_str == "cuda":
            step_times, final_loss, peak_mem_mb = measure_cuda(model, optimizer, scheduler, loss_fn, dataloader, device)
        else:
            step_times, final_loss, peak_mem_mb = measure_cpu(model, optimizer, scheduler, loss_fn, dataloader, device)

        # Compute metrics
        mean_step_ms = np.mean(step_times)
        p50_step_ms = np.percentile(step_times, 50)
        p90_step_ms = np.percentile(step_times, 90)
        tokens_per_step = batch_size * seq_len * 2
        tokens_per_sec = tokens_per_step / (mean_step_ms / 1000)

    except Exception as e:
        status = "error"
        notes = str(e)[:100]
        mean_step_ms = p50_step_ms = p90_step_ms = tokens_per_sec = peak_mem_mb = final_loss = 0

    return {
        "timestamp": datetime.now().isoformat(),
        "device": device_str,
        "precision": "fp32",
        "seq_len": seq_len,
        "batch_size": batch_size,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "repeat": repeat,
        "mean_step_ms": round(mean_step_ms, 3),
        "p50_step_ms": round(p50_step_ms, 3),
        "p90_step_ms": round(p90_step_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "final_loss": round(final_loss, 4) if isinstance(final_loss, float) else 0,
        "status": status,
        "notes": notes,
    }


def main():
    if len(sys.argv) != 6:
        print("Usage: python bench_single.py <device> <seq_len> <batch_size> <repeat> <output_file>")
        sys.exit(1)

    device = sys.argv[1]
    seq_len = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    repeat = int(sys.argv[4])
    output_file = sys.argv[5]

    print(f"Running: device={device}, seq_len={seq_len}, batch={batch_size}, repeat={repeat}")

    result = run_benchmark(device, seq_len, batch_size, repeat)

    # Append to CSV
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    # Print result
    if result["status"] == "success":
        print(f"  -> {result['mean_step_ms']:.1f} ms/step, {result['tokens_per_sec']:.0f} tok/s")
    else:
        print(f"  -> Error: {result['notes']}")

    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
