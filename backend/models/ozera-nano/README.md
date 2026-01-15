# Ozera-Nano Model

## Model Information

- **Name**: ozera-nano
- **Architecture**: Transformer Language Model
- **Parameters**: 12,412,608 (~12.4M)
- **Configuration**:
  - Layers: 6
  - Attention Heads: 6
  - Model Dimension: 192
  - Vocabulary Size: 50,257 (BPE tokenization)
- **Training Dataset**: OpenWebText
- **Validation Loss**: 4.8672
- **Trained**: January 14, 2026
- **File Size**: 142 MB

## Files

- `model.pt` - PyTorch model checkpoint (142 MB)

## Usage

### Local Generation (requires PyTorch)

```bash
cd backend
source ../venv/bin/activate

# Generate text
python scripts/generate_bpe.py \
  --checkpoint models/ozera-nano/model.pt \
  --prompt "Once upon a time" \
  --max-tokens 150 \
  --temperature 0.6 \
  --device cpu
```

### Parameters

- `--checkpoint`: Path to model checkpoint
- `--prompt`: Text prompt to start generation
- `--max-tokens`: Maximum number of tokens to generate (default: 200)
- `--temperature`: Sampling temperature, higher = more creative (default: 0.8)
  - **Recommended**: 0.5-0.7 for nano model (higher temps can cause repetition)
- `--top-k`: Top-k sampling parameter (default: 40)
- `--device`: Device to use (cuda/cpu, default: cuda)

### Example Outputs

**Prompt: "Once upon a time"** (temperature=0.8)
```
Once upon a time in order of the space, the first time, it was to be able
to use a small small space, and the best of the space space to get the chance
to the way to develop an space space space-based space to the space...
```

**Prompt: "In conclusion,"** (temperature=0.6)
```
In conclusion, the number of the findings were not subject to the findings.
This number of studies have been shown in the number of studies on the study.
It is clear that the findings were not evidence of evidence that the findings
were made by the study...
```

## Model Characteristics

### Strengths
- Faster inference due to smaller size (12M vs 51M parameters)
- Lower memory requirements (142 MB vs 586 MB)
- Good for resource-constrained environments
- Generates grammatically correct text

### Limitations
- More prone to repetition at high temperatures
- Less diverse vocabulary usage
- Better suited for simpler generation tasks
- **Best used with temperature 0.5-0.7** (vs 0.7-0.9 for mini)

## Model Checkpoint Structure

The checkpoint contains:
- `model_state_dict`: Model weights
- `config`: Model configuration
- `val_loss`: Validation loss (4.8672)
- `epoch`: Training epoch number (40 epochs total)

## Comparison with Ozera-Mini

| Metric | Nano | Mini |
|--------|------|------|
| Parameters | 12.4M | 51.2M |
| Layers | 6 | 8 |
| Hidden Dim | 192 | 512 |
| Val Loss | 4.867 | 4.009 |
| File Size | 142 MB | 586 MB |
| Best Temp | 0.5-0.7 | 0.7-0.9 |

## Notes

- The model uses GPT-2's BPE tokenizer (via tiktoken)
- No additional tokenizer files needed - tiktoken handles vocabulary
- Model was trained on Lambda GPU instance (NVIDIA A100)
- Trained for 40 epochs with batch size 64
- Use lower temperature values compared to mini model for best results
