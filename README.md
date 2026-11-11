# Credibility-aware MLOps

Replication package of the paper *"An MLOps Framework for Traceable and Reproducible Verification Evidence in the Credibility Assessment of ML Predictors"*.

This repository contains `cmlops`, an MLOps pipeline that embeds verification activities into ML experimentation. Every time models are trained, the best model for each output is put through a set of verification tests, and the results are stored with the training run that produced the model. A model is registered only if it passes every test.

The verification tests are implemented by [vv4ml](https://github.com/collab-uniba/vv4ml), a Python library of ML verification activities adapted from the verification, validation, uncertainty quantification, and applicability (VVUQA) approach to the credibility assessment of computational models (ASME V&V 40).

The pipeline is demonstrated on a case study: BoneStrengthML, an ML surrogate of a finite-element bone-strain model (see [Case study](#case-study-bonestrengthml)).

## How the pipeline works

A single YAML configuration file describes the dataset, the models to train, and the verification tests to run. Given that file, `cmlops train` performs the following steps:

1. **Validate the inputs.** The configuration is validated with Pydantic, and the dataset is validated with a Pandera schema generated from the field specifications in the configuration (types, ranges, missing values, duplicates).
2. **Split the data** into train and test sets, either row by row or by groups of rows that must stay together.
3. **Train every configured model for every output**, each with a cross-validated hyperparameter search. Each trained model is logged as an MLflow run.
4. **Select a winner per output**: the model with the lowest test RMSE.
5. **Verify each winner** with the tests listed in the configuration.
6. **Record the evidence and register.** The verification results are attached to the winner's MLflow run, and the model is added to the MLflow Model Registry only if all tests passed.

The steps are orchestrated as [Prefect](https://www.prefect.io/) flows; [MLflow](https://mlflow.org/) provides experiment tracking and the model registry. See [docs/workflow-architecture.md](docs/workflow-architecture.md) for the details.

### Verification tests

| Test | What it checks |
|------|----------------|
| `convergence` | Retrains the winner's configuration on growing subsets of the training data and checks that its predictions converge to those of the winner. |
| `smoothness` | Perturbs each input slightly and checks that the model does not amplify the perturbation. |
| `numerical_error` | Retrains the winner's configuration several times on independent subsets and checks that the spread of the predictions stays below a threshold. |

All three evaluate the model on points drawn by Latin Hypercube Sampling from the input ranges declared in the configuration. Thresholds and parameters are set per test and per output in the `verification` section of the configuration.

### Verification evidence

The evidence is stored in the MLflow run of the model it refers to, next to the training record of that model:

- **Training record:** model type, selected hyperparameters, split and search settings, train and test metrics, the cross-validation results, and the fitted model.
- **Verification metrics and parameters**, named `verification.<test>.<name>`.
- **Verification outcome tags:** `verification.<test>` (`pass` or `fail`) for each test, and `verification.status` for the overall outcome.
- **Verification report:** the full structured results, as the artifact `verification/report.json`.

A registered model version points to the run it was registered from, and therefore to its evidence.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

**macOS only**: XGBoost requires the OpenMP runtime:

```bash
brew install libomp
```

### Setup

```bash
# Clone the repository
git clone https://github.com/collab-uniba/Credibility-aware-MLOps.git
cd Credibility-aware-MLOps

# Install dependencies
uv sync
```

The commands below assume the project environment is active (`source .venv/bin/activate`); otherwise, prefix them with `uv run`.

## Usage

Without a `--config` option, every command uses the configuration of the case study, `config/BoneStrengthML.yml`.

### Training and verifying models

Train all configured models for all outputs, then verify and register the winners:

```bash
cmlops train
```

This is the run that reproduces the case study. It trains eight model types for two outputs with large hyperparameter grids, so it takes a long time.

Train all models for one output, then verify and register the winner:

```bash
cmlops train --output maxStrain_11
```

Train one model for one output. This is a shortcut for debugging: it runs no verification and registers nothing.

```bash
cmlops train --output maxStrain_11 --model RandomForestRegressor
```

To stop after training, skipping verification and therefore registration:

```bash
cmlops train --skip-verification
```

Training runs several models at once, and each model's hyperparameter search is itself parallel. Two options size both levels so that together they stay within a share of the machine's CPUs:

```bash
cmlops train --max-workers 4 --cpu-fraction 0.5
```

`--max-workers` (default 2) is the number of models trained concurrently; `--cpu-fraction` (default 0.9) is the share of the available CPUs the searches may use together. Verification is not covered by this budget.

### Viewing results

Start the MLflow UI to explore the runs, their verification evidence, and the registered models:

```bash
cmlops mlflow-ui
```

Then open http://localhost:5000 in your browser.

### Monitoring the workflow (optional)

The pipeline runs locally by default. To follow it in the Prefect UI:

1. Start the Prefect server in a separate terminal:
   ```bash
   prefect server start
   ```
2. Open http://localhost:4200 in your browser.
3. Run training as usual; the flows appear in the UI.

[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) describes how to set up the pipeline, the MLflow UI, and the Prefect server on an Ubuntu server.

## Case study: BoneStrengthML

BoneStrengthML is an ML-based surrogate of BoneStrength, a finite-element simulation that predicts bone strain values from CT-derived femur characteristics. It performs regression to predict the strain induced in the bone during simulated fall events.

**Inputs** (95 features):

- 93 principal components representing a morpho-densitometric femur statistical atlas
- PosAnt and MedLat angles describing impact force orientation during side-fall events

**Outputs** (2 targets):

- `maxStrain_11`: maximum first principal strain
- `maxStrain_33`: absolute value of minimum third principal strain

The case study consists of three files:

| File | Content |
|------|---------|
| [`config/BoneStrengthML.yml`](config/BoneStrengthML.yml) | Pipeline configuration: dataset schema, models and hyperparameter grids, verification tests and thresholds |
| `data/raw/sampleDataset.csv` | Dataset |
| [`docs/Context_of_use.md`](docs/Context_of_use.md) | Context of use of the surrogate, from which the acceptance and verification thresholds are derived |

## Applying the pipeline to another predictor

The pipeline code does not depend on the case study. To apply it to another tabular regression problem, write a configuration file and pass it with `--config`:

```bash
cmlops train --config path/to/your-config.yml
```

The configuration has five sections:

- **`project`**: the project name, used as the default MLflow experiment name and as the prefix of registered model names.
- **`dataset`**: path and format of the CSV file, and the specification of every input and output field (name, type, valid range, units). The ranges are used both for data validation and for sampling the verification points.
- **`model_building`**: train/test split, hyperparameter search settings, and the list of models with their hyperparameter grids. Supported model types: random forest, XGBoost, support vector regression, linear regression, multi-layer perceptron, partial least squares, CatBoost, and Gaussian process regression.
- **`verification`**: the verification tests to run, each with its thresholds per output and its parameters.
- **`gate_thresholds`**: acceptance thresholds on model accuracy per output. They are declared in the configuration, but the pipeline does not enforce them yet.

`cmlops list-models --config path/to/your-config.yml` shows the models a configuration declares.

## Citation

The paper is in preparation. Its reference and the DOI of the archived version of this package will be added here.

## License

The code is released under the [MIT License](LICENSE).

## Project structure

```text
cmlops/                 # The pipeline
    config.py               # Pydantic configuration models
    data_loading.py         # Data loading and validation with Pandera
    cli.py                  # Command-line interface
    training/               # Model training
        model_factory.py        # Model instantiation from config (incl. GPR pipeline and kernel registry)
        metrics.py              # Evaluation metrics
        splitter.py             # Train/test splitting
        trainer.py              # Hyperparameter optimization
        mlflow_utils.py         # MLflow integration
    verification/           # Verification tests (implemented by the vv4ml library)
        base.py                 # Check interface, context, result and report
        registry.py             # Maps YAML test types to checks
        verify.py               # Runs the configured checks on a winning model
        convergence.py          # Adapter for the vv4ml convergence test
        smoothness.py           # Adapter for the vv4ml smoothness test
        numerical_error.py      # Adapter for the vv4ml numerical error test
    workflows/              # Prefect orchestration
        tasks.py                # Prefect tasks
        flows.py                # Training and verification flows
config/                 # Case-study configuration
data/raw/               # Case-study dataset
docs/                   # Workflow architecture, deployment guide, case-study context of use
notebooks/              # Data loading and validation demo
```
