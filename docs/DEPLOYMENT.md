# Deploying the pipeline on an Ubuntu server

This guide covers deploying the training and verification pipeline on an Ubuntu server, including optional Prefect UI setup.

## Port Configuration

Choose ports that are available on your server:

```bash
# Set your custom ports (adjust as needed)
export PREFECT_PORT=4250
export MLFLOW_PORT=5050
```

---

## 1. System Prerequisites

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential git curl
```

## 2. Install Python 3.11+

```bash
# Add deadsnakes PPA for latest Python versions
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.11 and venv support
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

## 3. Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH (for current session)
source $HOME/.local/bin/env

# Verify installation
uv --version
```

## 4. Clone the Repository

```bash
git clone https://github.com/collab-uniba/Credibility-aware-MLOps.git

cd Credibility-aware-MLOps
```

## 5. Install Project Dependencies

```bash
# Install all dependencies (including Prefect)
uv sync
```

## 6. Verify Installation

```bash
# Check CLI is available
uv run cmlops --help

# List configured models
uv run cmlops list-models
```

## 7. Configure Prefect Server (Optional - for UI)

If you want the Prefect web UI for workflow monitoring:

```bash
# Set Prefect API URL with your custom port
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:${PREFECT_PORT}/api

# Start Prefect server on custom port (run in background or separate terminal)
uv run prefect server start --host 0.0.0.0 --port ${PREFECT_PORT} &
```

The Prefect UI will be available at `http://<server-ip>:${PREFECT_PORT}`

## 8. Run Training

```bash
# Train all models
uv run cmlops train

# Or train the models for one output (here, an output of the case study)
uv run cmlops train --output maxStrain_11
```

## 9. View MLflow Results

```bash
# Start MLflow UI on custom port
uv run cmlops mlflow-ui --port ${MLFLOW_PORT}
```

The MLflow UI will be available at `http://<server-ip>:${MLFLOW_PORT}`

---

## Running as Background Services (Production)

For persistent deployment, create systemd services.

**Important:** Replace `<your-user>`, `/path/to/Credibility-aware-MLOps`, and port numbers with your actual values.

### Prefect Server Service

```bash
sudo tee /etc/systemd/system/prefect-server.service << 'EOF'
[Unit]
Description=Prefect Server
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/path/to/Credibility-aware-MLOps
Environment="PREFECT_API_URL=http://127.0.0.1:4250/api"
ExecStart=/home/<your-user>/.local/bin/uv run prefect server start --host 0.0.0.0 --port 4250
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prefect-server
sudo systemctl start prefect-server
```

### MLflow UI Service

```bash
sudo tee /etc/systemd/system/mlflow-ui.service << 'EOF'
[Unit]
Description=MLflow UI
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/path/to/Credibility-aware-MLOps
ExecStart=/home/<your-user>/.local/bin/uv run cmlops mlflow-ui --port 5050
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow-ui
sudo systemctl start mlflow-ui
```

---

## Firewall Configuration

If using UFW (adjust ports to match your configuration):

```bash
# Allow MLflow UI
sudo ufw allow 5050/tcp

# Allow Prefect UI (if using)
sudo ufw allow 4250/tcp
```

---

## Quick Reference

| Service | Default Port | Custom Port Flag | Environment Variable |
|---------|--------------|------------------|---------------------|
| MLflow UI | 5000 | `--port <port>` | — |
| Prefect Server | 4200 | `--port <port>` | `PREFECT_API_URL` must match |

**Note:** When changing the Prefect server port, you must also update `PREFECT_API_URL` to match, otherwise flows won't connect to the server.
