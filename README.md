# Credibility-aware MLOps

Replication package of the paper *"An MLOps Framework for Traceable and Reproducible Verification Evidence in the Credibility Assessment of ML Predictors"*.

This repository contains `cmlops`, an MLOps pipeline that embeds verification activities into ML experimentation. Every time models are trained, the best model for each output is put through a set of verification tests, and the results are stored with the training run that produced the model, together with the data version, configuration, and code version the run used. A model is registered, and can then be served, only if it meets the accuracy thresholds of its context of use and passes every test.

The verification tests are implemented by [vv4ml](https://github.com/collab-uniba/vv4ml), a Python library of ML verification activities adapted from the verification, validation, uncertainty quantification, and applicability (VVUQA) approach to the credibility assessment of computational models (ASME V&V 40).

The pipeline is demonstrated on a case study: BoneStrengthML, an ML surrogate of a finite-element bone-strain model (see [Case study](#case-study-bonestrengthml)).

## How the pipeline works

A single YAML configuration file describes the dataset, the models to train, and the verification tests to run. Given that file, `cmlops train` performs the following steps:

1. **Validate the inputs.** The configuration is validated with Pydantic, and the dataset is validated with a Pandera schema generated from the field specifications in the configuration (types, ranges, missing values, duplicates).
2. **Identify what the run is produced from**: the dataset version (the MD5 of the dataset file, which must match the version recorded by DVC), the configuration file, the git commit of the code, and the random seed.
3. **Split the data** into train and test sets, either row by row or by groups of rows that must stay together.
4. **Train every configured model for every output**, each with a cross-validated hyperparameter search. Each trained model is logged as an MLflow run, with its lineage and with the outcome of the comparison between its test error and the accuracy thresholds.
5. **Select a winner per output**: the model with the lowest test RMSE.
6. **Verify each winner** with the tests listed in the configuration.
7. **Record the evidence and register.** The verification results are attached to the winner's MLflow run, and the model is added to the MLflow Model Registry only if it met the accuracy thresholds and all tests passed.

A registered model can then be served as a REST API (see [Serving a model](#serving-a-model)).

The steps are orchestrated as [Prefect](https://www.prefect.io/) flows; [MLflow](https://mlflow.org/) provides experiment tracking, the model registry, and model serving; [DVC](https://dvc.org/) versions the dataset. See [docs/workflow-architecture.md](docs/workflow-architecture.md) for the details.

### Verification tests

| Test | What it checks |
|------|----------------|
| `convergence` | Retrains the winner's configuration on growing subsets of the training data and checks that its predictions converge to those of the winner. |
| `smoothness` | Perturbs each input slightly and checks that the model does not amplify the perturbation. |
| `numerical_error` | Retrains the winner's configuration several times on independent subsets and checks that the spread of the predictions stays below a threshold. |

All three evaluate the model on points drawn by Latin Hypercube Sampling from the input ranges declared in the configuration. Thresholds and parameters are set per test and per output in the `verification` section of the configuration.

### Verification evidence

The evidence is stored in the MLflow run of the model it refers to, next to the training record of that model:

- **Lineage:** the dataset (name, MD5 digest, and source), the declared dataset version (tag `dataset.version`), the random seed (parameter `random_state`), the configuration file (artifact `config/`), and the git commit of the code (tag `mlflow.source.git.commit`; `source.git.dirty` tells whether tracked files had uncommitted changes).
- **Training record:** model type, selected hyperparameters, split and search settings, train and test metrics, the cross-validation results, and the fitted model.
- **Accuracy outcome:** each threshold as the parameter `accuracy.<metric>.threshold`, its outcome as the tag `accuracy.<metric>` (`pass` or `fail`), and the overall outcome as the tag `accuracy.status`. The metric itself is the test metric `test_<metric>`.
- **Verification metrics and parameters**, named `verification.<test>.<name>`.
- **Verification outcome tags:** `verification.<test>` (`pass` or `fail`) for each test, and `verification.status` for the overall outcome.
- **Verification report:** the full structured results, as the artifact `verification/report.json`.

Lineage, training record, and accuracy outcome are logged for every trained model; the verification items are logged for the winner of each output, the only model that is verified. A registered model version points to the run it was registered from, and therefore to its evidence.

### Data versioning

The dataset is versioned with DVC. Git tracks a small pointer file, `data/raw/sampleDataset.csv.dvc`, that holds the MD5 of the dataset; the dataset itself is stored in the DVC remote, hosted on [DagsHub](https://dagshub.com/louieQ/Credibility-aware-MLOps) (see [Getting the dataset](#getting-the-dataset)). A git commit therefore pins the code, the configuration, and the data version together.

The pipeline ties each run to that version in two ways:

- It logs the MD5 of the dataset file it loaded as the dataset digest of the run. The digest of a run is the `md5` field of a pointer committed to git.
- It refuses to run on a dataset file that differs from its pointer, since such a file corresponds to no recorded version.

To record a new version of the dataset, replace the file and run:

```bash
dvc add data/raw/sampleDataset.csv
git add data/raw/sampleDataset.csv.dvc
git commit -m "Update the dataset"
dvc push
```

Update the `dataset.version` label in the configuration in the same commit; it is logged on every run next to the digest. An earlier version is restored by checking out its commit and running `dvc checkout`.

A dataset file without a pointer is not tracked by DVC and is not checked; its MD5 is logged all the same.

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

### Getting the dataset

The case-study dataset is not in the git repository; it is downloaded from the DVC remote on DagsHub. Without a DagsHub account, download the version that belongs to the checked-out commit:

```bash
curl -L -o data/raw/sampleDataset.csv \
  "https://dagshub.com/louieQ/Credibility-aware-MLOps/raw/$(git rev-parse HEAD)/data/raw/sampleDataset.csv"
```

The pipeline checks the downloaded file against the MD5 in `data/raw/sampleDataset.csv.dvc` before using it (see [Data versioning](#data-versioning)).

DagsHub does not serve `dvc pull` to anonymous users. To use DVC, which is needed to record new versions of the dataset, create a DagsHub account and an access token, and store them in the local, untracked DVC configuration:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <your-dagshub-username>
uv run dvc remote modify origin --local password <your-dagshub-token>
uv run dvc pull
```

Pushing data also requires write access to the DagsHub repository.

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

A winner that misses an accuracy threshold is still verified, so that its evidence is recorded, but it is not registered.

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

### Querying the evidence

The evidence can also be queried through the MLflow API. For example, to list the runs trained on a given data version whose model met the accuracy thresholds and passed verification:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
runs = mlflow.search_runs(
    experiment_names=["BoneStrengthML"],
    filter_string=(
        "dataset.digest = '92be51463af78164dc0ed8c5048bd8f2' "
        "and tags.`accuracy.status` = 'passed' "
        "and tags.`verification.status` = 'passed'"
    ),
)
print(runs[["tags.mlflow.runName", "metrics.test_max_ae", "tags.mlflow.source.git.commit"]])
```

The digest is the MD5 of the dataset file, as recorded in the `md5` field of `data/raw/sampleDataset.csv.dvc`.

### Serving a model

Serve the latest registered model of an output as a REST API:

```bash
cmlops serve --output maxStrain_11
```

The command serves models from the MLflow Model Registry only, so a model can be served only after it met the accuracy thresholds and passed every verification test. It prints the name, version, and run of the model it serves, and listens on port 5001 unless `--port` says otherwise.

Predictions are requested with a POST to `/invocations`, with the input fields as columns:

```bash
curl http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["PC_1", "PC_2", "..."], "data": [[0.12, -0.34, "..."]]}}'
```

Every input field declared in the configuration must be present. A complete example request is stored with each model, as the artifact `serving_input_example.json`.

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
| `data/raw/sampleDataset.csv` | Dataset, versioned with DVC: git tracks the pointer `data/raw/sampleDataset.csv.dvc`, and the file is downloaded from the DVC remote (see [Getting the dataset](#getting-the-dataset)) |
| [`docs/Context_of_use.md`](docs/Context_of_use.md) | Context of use of the surrogate, from which the acceptance and verification thresholds are derived |

## Applying the pipeline to another predictor

The pipeline code does not depend on the case study. To apply it to another tabular regression problem, write a configuration file and pass it with `--config`:

```bash
cmlops train --config path/to/your-config.yml
```

The configuration has five sections:

- **`project`**: the project name, used as the default MLflow experiment name and as the prefix of registered model names.
- **`dataset`**: path and format of the CSV file, and the specification of every input and output field (name, type, valid range, units). The ranges are used both for data validation and for sampling the verification points. To version the dataset, track the file with `dvc add` (see [Data versioning](#data-versioning)).
- **`model_building`**: train/test split, hyperparameter search settings, and the list of models with their hyperparameter grids. Supported model types: random forest, XGBoost, support vector regression, linear regression, multi-layer perceptron, partial least squares, CatBoost, and Gaussian process regression.
- **`verification`**: the verification tests to run, each with its thresholds per output and its parameters.
- **`gate_thresholds`**: acceptance thresholds on model accuracy per output, on the maximum absolute error or the RMSE measured on the test set. A model that exceeds a threshold of its output is not registered.

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
    cli.py                  # Command-line interface (train, serve, mlflow-ui, list-models)
    training/               # Model training
        model_factory.py        # Model instantiation from config (incl. GPR pipeline and kernel registry)
        metrics.py              # Evaluation metrics
        acceptance.py           # Accuracy assessment against the acceptance thresholds
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
data/raw/               # Case-study dataset (DVC pointer; the file is downloaded from the DVC remote)
.dvc/                   # DVC configuration
docs/                   # Workflow architecture, deployment guide, case-study context of use
```
