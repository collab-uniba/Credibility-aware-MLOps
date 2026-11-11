# BoneStrengthML

ML-based surrogate model for BoneStrength, a finite element (FE) simulation that predicts bone strain values from CT-derived femur characteristics.

## Overview

BoneStrengthML performs regression to predict bone strain values induced during simulated fall events based on patient-specific femur characteristics:

**Inputs** (95 features):

- 93 principal components representing a morpho-densitometric femur statistical atlas
- PosAnt and MedLat angles describing impact force orientation during side-fall events

**Outputs** (2 targets):

- `maxStrain_11`: maximum first principal strain
- `maxStrain_33`: absolute value of minimum third principal strain

## Installation

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager

**macOS only**: XGBoost requires the OpenMP runtime:

```bash
brew install libomp
```

### Setup

```bash
# Clone the repository
git clone https://github.com/collab-uniba/BoneStrengthML-SF3_application.git
cd BoneStrengthML-SF3_application

# Install dependencies
uv sync
```

## Usage

### Training Models

Train all configured models for all outputs:

```bash
bsml train
```

Train models for a specific output:

```bash
bsml train --output maxStrain_11
```

Train a specific model for a specific output:

```bash
bsml train --output maxStrain_11 --model RandomForestRegressor
```

By default, `bsml train` runs verification (convergence + smoothness) on the
best model per output and **automatically registers** any model that passes
all configured verification tests in the MLflow Model Registry. To skip
verification (and therefore registration) for fast iteration:

```bash
bsml train --skip-verification
```

The specific-model debug path (`bsml train --output X --model Y`) trains
only — no verification, no registration.

### Viewing Results

Start the MLflow UI to explore experiment results:

```bash
bsml mlflow-ui
```

Then open http://localhost:5000 in your browser.

### Workflow Monitoring (Optional)

The training pipeline uses Prefect for orchestration, running locally by default. To enable the Prefect UI for workflow monitoring:

1. Start the Prefect server (in a separate terminal):
   ```bash
   prefect server start
   ```

2. Open http://localhost:4200 in your browser

3. Run training as usual - flows will appear in the UI

### Configuration

List configured models:

```bash
bsml list-models
```

All pipeline behavior is controlled via `config/BoneStrengthML.yml`, including:

- Dataset path and format
- Model types and hyperparameter grids (RandomForest, XGBoost, SVM, Linear, MLP, PLS, CatBoost, GaussianProcess)
- Train/test split configuration
- Validation thresholds

## Project Structure

```text
bonestrength_ml/
    config.py           # Pydantic configuration models
    data_loading.py     # Data loading and validation with Pandera
    cli.py              # Command-line interface
    training/           # Model training pipeline
        model_factory.py    # Model instantiation from config (incl. GPR Pipeline & kernel registry)
        metrics.py          # Evaluation metrics
        splitter.py         # Train/test splitting
        trainer.py          # Hyperparameter optimization
        mlflow_utils.py     # MLflow integration
    verification/       # Verification tests (implemented by the vv4ml library)
        base.py             # Check interface, context, result and report
        registry.py         # Maps YAML test types to checks
        verify.py           # Runs the configured checks on a winning model
        convergence.py      # Adapter for the vv4ml convergence test
        smoothness.py       # Adapter for the vv4ml smoothness test
    workflows/          # Prefect orchestration
        tasks.py            # Prefect tasks
        flows.py            # Training flows (parallel execution)
```
