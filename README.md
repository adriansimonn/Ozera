![Ozera Logo](frontend/src/assets/logos/LogoWallpaperTGradientText.png)

An advanced tool for NLP/LLM Researchers & Enthusiasts focused on Mechanistic Interpretability

## Overview

Ozera enables deep understanding of how language models work internally through comprehensive visualization, analysis, and experimentation tools. Built from scratch with pure mathematical implementations (no framework abstractions for core models).

## Key Features

- **Custom Language Models**: Use your own datasets to train and save models of the Ozera transformer architecture
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

## Philosophy

- **No Frameworks, Pure Mathematics**: All models built from fundamental ML mathematics
- **Research-Grade Tools**: Rigorous, useful research capabilities with beautiful interfaces
- **Transparency**: Every transformation visible and understandable
- **Practicality**: Focus on tools researchers use daily
- **Polished Interfaces**: Intuitive layout, clean visuals + graphics, and polished styling

## Target Audience

NLP/LLM researchers, mechanistic interpretability practitioners, and ML enthusiasts working to understand neural network internals.

## License

TBD

## Contact

TBD
