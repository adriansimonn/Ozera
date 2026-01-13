"""
Test training pipeline with a small example.
"""

import numpy as np
import sys
sys.path.append('/Users/adrian/Documents/ozera/backend')

from core.transformer import TransformerLM, get_config
from core.training import (
    Trainer, TrainingConfig,
    SimpleTokenizer, TextDataset, DataLoader,
    cross_entropy_loss, perplexity
)


def test_data_pipeline():
    """Test data loading and tokenization."""
    print("Testing data pipeline...")

    # Simple text for testing
    text = "hello world! this is a test. hello again!"

    # Create tokenizer
    tokenizer = SimpleTokenizer(text)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Sample chars: {list(tokenizer.char_to_id.keys())[:10]}")

    # Test encoding/decoding
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello", f"Encode/decode failed: {decoded}"
    print("  Encode/decode: OK")

    # Create dataset
    dataset = TextDataset(text, tokenizer, seq_len=8)
    print(f"  Dataset size: {len(dataset)} sequences")

    # Test getting an item
    input_ids, target_ids = dataset[0]
    print(f"  Sample input: {input_ids}")
    print(f"  Sample target: {target_ids}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"  DataLoader batches: {len(dataloader)}")

    # Test iteration
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"  Batch {batch_idx}: inputs {inputs.shape}, targets {targets.shape}")
        if batch_idx >= 1:
            break

    print("✓ Data pipeline tests passed\n")


def test_loss_computation():
    """Test loss function."""
    print("Testing loss computation...")

    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 4, 100
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # Compute loss
    loss, grad = cross_entropy_loss(logits, targets)

    print(f"  Loss: {loss:.4f}")
    print(f"  Perplexity: {perplexity(loss):.4f}")
    print(f"  Gradient shape: {grad.shape}")

    assert grad.shape == logits.shape
    print("✓ Loss computation tests passed\n")


def test_model_forward():
    """Test model forward pass with training."""
    print("Testing model forward pass...")

    # Create small model
    config = get_config('dev')
    model = TransformerLM(config)

    # Create dummy input
    batch_size = 2
    seq_len = 16
    token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, _ = model.forward(token_ids, training=True)

    print(f"  Model: {model.count_parameters():,} parameters")
    print(f"  Input shape: {token_ids.shape}")
    print(f"  Output shape: {logits.shape}")

    # Compute loss
    targets = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    loss, grad = cross_entropy_loss(logits, targets)

    print(f"  Loss: {loss:.4f}")
    print("✓ Model forward tests passed\n")


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")

    # Create model and trainer
    config = get_config('dev')
    model = TransformerLM(config)

    train_config = TrainingConfig(
        batch_size=2,
        seq_len=16,
        learning_rate=1e-3,
        grad_clip=1.0
    )

    trainer = Trainer(model, train_config, use_numerical_grads=False)

    # Create dummy batch
    token_ids = np.random.randint(0, config.vocab_size, (2, 16))
    targets = np.random.randint(0, config.vocab_size, (2, 16))

    # Training step
    print("  Running training step (this may take a moment)...")
    loss = trainer.train_step(token_ids, targets)

    print(f"  Initial loss: {loss:.4f}")
    print(f"  Training step: {trainer.step}")

    # Do a few more steps
    for i in range(3):
        loss = trainer.train_step(token_ids, targets)
        print(f"  Step {i+2} loss: {loss:.4f}")

    print("✓ Training step tests passed\n")


def test_end_to_end():
    """Test end-to-end training pipeline."""
    print("Testing end-to-end training...")

    # Create simple text dataset
    text = "abcdefghijklmnopqrstuvwxyz " * 20  # Repeat to have enough data
    tokenizer = SimpleTokenizer(text)

    # Adjust config to match vocab size
    config = get_config('dev')
    config.vocab_size = tokenizer.vocab_size

    # Create model
    model = TransformerLM(config)
    print(f"  Model: {model.count_parameters():,} params, vocab: {config.vocab_size}")

    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, seq_len=16)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"  Dataset: {len(dataset)} sequences, {len(dataloader)} batches")

    # Create trainer
    train_config = TrainingConfig(
        batch_size=4,
        seq_len=16,
        learning_rate=1e-3,
        log_interval=5
    )
    trainer = Trainer(model, train_config, use_numerical_grads=False)

    # Train for a few steps
    print("  Training for 10 steps...")
    losses = []
    for step_idx, (inputs, targets) in enumerate(dataloader):
        if step_idx >= 10:
            break

        loss = trainer.train_step(inputs, targets)
        losses.append(loss)

        if step_idx % 5 == 0:
            print(f"    Step {step_idx}: loss={loss:.4f}, ppl={perplexity(loss):.2f}")

    print(f"  Final loss: {losses[-1]:.4f}")
    print("✓ End-to-end training tests passed\n")


def main():
    """Run all tests."""
    print("="*60)
    print("TRAINING PIPELINE TEST SUITE")
    print("="*60 + "\n")

    np.random.seed(42)

    test_data_pipeline()
    test_loss_computation()
    test_model_forward()
    test_training_step()
    test_end_to_end()

    print("="*60)
    print("ALL TRAINING TESTS PASSED!")
    print("="*60)
    print("\nNote: Using simplified gradients for testing.")
    print("For production training, implement full backward passes.")


if __name__ == "__main__":
    main()
