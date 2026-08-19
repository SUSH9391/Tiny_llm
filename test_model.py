#!/usr/bin/env python3
"""
Test script to verify the Tiny LLM model works correctly.
"""

import torch
import sys
import os

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Config import configure_model
from architecture import build_model

def test_model():
    # Create a mock config object
    class Config:
        pass

    cfg = Config()
    cfg = configure_model(cfg)

    # Set vocab size (needed for embeddings)
    cfg.vocab_size = 1000  # Small vocab for testing
    cfg.max_steps = 1000   # Needed for LR scheduler

    print("Building model...")
    model = build_model(cfg)
    print(f"Model built successfully: {model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass with dummy input
    batch_size = 2
    seq_length = 10
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_length))

    print(f"Running forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Output logits shape: {logits.shape}")
    expected_shape = (batch_size, seq_length, cfg.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

    print("Pass: Forward pass successful!")

    # Test that we can compute loss
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_length))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    print(f"Pass: Loss computation successful: {loss.item():.4f}")

    print("\nSuccess: All tests passed! The model is working correctly.")

if __name__ == "__main__":
    test_model()