# My Custom String Reversal Environment

This example demonstrates using a locally-developed `verifiers` environment for RL training.

## What it does

The environment generates random strings (e.g., `"hello123"`) and asks the model to reverse them (e.g., `"321olleh"`).

Rewards:
- **70%** exact match (binary: 0 or 1)
- **30%** LCS (Longest Common Subsequence) ratio for partial credit

## Quick Start

### 1. Verify the environment is installed

```bash
uv run python -c "import my_custom_env; print('OK')"
```

### 2. Test the environment

```bash
# Start inference server
uv run inference --model.name google/gemma-4-26B-A4B-it

# In another terminal, test with vf-eval
uv run vf-eval my-custom-env -m google/gemma-4-26B-A4B-it -b http://localhost:8000/v1 -n 10 --max-tokens 128
```

### 3. Run RL training

```bash
uv run rl @ examples/my_custom_env/rl.toml
```

This will:
- Start inference server with Gemma 4 26B A4B
- Train for 20 steps with batch size 32
- Generate 500 training examples and 50 eval examples
- Save checkpoints to `outputs/weights/`

### 4. Run SFT (optional)

For better initialization, you can run SFT first using Wikipedia reversal data:

```bash
uv run sft @ examples/my_custom_env/sft.toml
```

## Customizing

Modify the environment parameters in `rl.toml`:

```toml
[[orchestrator.train.env]]
id = "my-custom-env"
args = {
    num_train_examples = 1000,  # More training data
    num_eval_examples = 100,     # More eval examples
    min_length = 10,             # Longer strings
    max_length = 50,             # Max string length
    seed = 42
}
```

## Files

- `my_custom_env.py` - The environment implementation
- `rl.toml` - RL training config
- `sft.toml` - SFT training config
