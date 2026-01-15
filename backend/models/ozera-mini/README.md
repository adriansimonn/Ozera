# Ozera-Mini Model

## Model Information

- **Name**: ozera-mini
- **Architecture**: Transformer Language Model
- **Parameters**: 51,197,440 (~51M)
- **Configuration**:
  - Layers: 8
  - Attention Heads: 8
  - Model Dimension: 512
  - Vocabulary Size: 50,257 (BPE tokenization)
- **Training Dataset**: OpenWebText
- **Validation Loss**: 4.0085
- **Trained**: January 14-15, 2026
- **File Size**: 586 MB

## Files

- `model.pt` - PyTorch model checkpoint (586 MB)

## Usage

### Local Generation (requires PyTorch)

```bash
cd backend
source ../venv/bin/activate

# Generate text
python scripts/generate_bpe.py \
  --checkpoint models/ozera-mini/model.pt \
  --prompt "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8 \
  --device cpu
```

### Parameters

- `--checkpoint`: Path to model checkpoint
- `--prompt`: Text prompt to start generation
- `--max-tokens`: Maximum number of tokens to generate (default: 200)
- `--temperature`: Sampling temperature, higher = more creative (default: 0.8)
- `--top-k`: Top-k sampling parameter (default: 40)
- `--device`: Device to use (cuda/cpu, default: cuda)

### Example Outputs

**Prompt: "Once upon a time"** (temperature=0.8)
```
Once upon a time, we are in a state of war with the Syrian government,
which is a terrorist organization and we are in a military society...
```

**Prompt: "The future of artificial intelligence"** (temperature=0.7)
```
The future of artificial intelligence is already very difficult to understand.
We can't imagine the future of artificial intelligence, which is already in the wild...
```

## Model Checkpoint Structure

The checkpoint contains:
- `model_state_dict`: Model weights
- `config`: Model configuration
- `val_loss`: Validation loss
- `epoch`: Training epoch number

## Notes

- The model uses GPT-2's BPE tokenizer (via tiktoken)
- No additional tokenizer files needed - tiktoken handles vocabulary
- Model was trained on Lambda GPU instance
- Generates coherent, grammatically correct text
- Best suited for general web content generation
