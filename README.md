# Ozera

An advanced mechanistic interpretability platform for NLP/LLM researchers.

## Overview

Ozera enables deep understanding of how language models work internally through comprehensive visualization, analysis, and experimentation tools. Built from scratch with pure mathematical implementations (no framework abstractions for core models).

## Key Features

- **Custom Language Models**: Train 1M and 10M parameter models from scratch
- **Advanced Interpretability**: Layer-by-layer activation visualization, attention pattern analysis
- **Activation Patching Playground**: Interactive experimentation with model internals
- **SAE Analysis**: Sparse autoencoder training and feature discovery
- **Research-Grade Tools**: Publication-ready exports and systematic testing frameworks

## Project Structure

```
ozera/
├── backend/           # Python backend for training, inference, and API
│   ├── core/          # Core model implementations (pure math)
│   ├── api/           # REST/WebSocket API endpoints
│   ├── services/      # Business logic and compute management
│   ├── models/        # Saved model checkpoints
│   └── training/      # Training pipelines
├── frontend/          # React/TypeScript visualization interface
│   └── src/
│       ├── components/  # UI components
│       ├── pages/       # Application pages
│       └── utils/       # Frontend utilities
├── shared/            # Shared types and schemas
├── docs/              # Documentation
└── scripts/           # Development and deployment scripts
```

## Getting Started

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup instructions.

## Philosophy

- **No Frameworks, Pure Mathematics**: All models built from fundamental ML mathematics
- **Research-Grade Tools**: Rigorous, useful research capabilities with beautiful interfaces
- **Transparency**: Every transformation visible and understandable
- **Practicality**: Focus on tools researchers use daily

## Target Audience

NLP/LLM researchers, mechanistic interpretability practitioners, and ML enthusiasts working to understand neural network internals.

## License

TBD

## Contact

TBD
