#  Contract Analysis AI — Backend
> Fine-grained contract clause classification powered by **DistilBERT** + **FastAPI**

---

## Architecture
```
contract-ai/
├── app/
│   ├── api/
│   │   ├── routes.py          # GET /health · POST /predict · POST /train
│   │   └── schema.py          # Pydantic request/response models
│   ├── core/
│   │   ├── config.py          # Pydantic Settings (env vars / .env)
│   │   └── model_loader.py    # Singleton model registry (loaded once)
│   ├── services/
│   │   ├── inference.py       # predict_clause(text) → label + confidence
│   │   ├── preprocessing.py   # Dataset load, clean, tokenize
│   │   └── training.py        # HuggingFace Trainer pipeline
│   └── main.py                # FastAPI app + lifespan startup
├── models/
│   ├── contract_classifier/   # Fine-tuned model weights (save here)
│   └── label_map.json         # { "0": "Termination", "1": "Indemnification", … }
├── data/                      # Raw / cached dataset files
├── notebooks/
│   └── colab_training.ipynb   # 🔥 Google Colab training notebook (GPU)
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) — install with `pip install uv`

### 1 — Clone & Install
```bash
git clone <your-repo-url>
cd legal-contract/contract-ai
cp .env.example .env

uv sync          # installs all dependencies from pyproject.toml
```

### 2 — Train the Model

#### Option A — 🔥 Google Colab (Recommended, free GPU)
1. Upload `notebooks/colab_training.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **GPU → T4** via *Runtime → Change runtime type*
3. Run all cells — training takes ~20–30 minutes on T4
4. Download `contract_classifier.zip` when prompted
5. Extract into your `models/` directory:
   ```powershell
   Expand-Archive contract_classifier.zip -DestinationPath models\contract_classifier
   copy models\contract_classifier\label_map.json models\label_map.json
   ```

#### Option B — Local Training via API endpoint
Start the server first, then trigger training:
```bash
# Start server (model will be unavailable until training finishes)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal — launch training
curl -X POST http://localhost:8000/api/v1/train \
     -H "Content-Type: application/json" \
     -d '{"num_train_epochs": 3, "per_device_train_batch_size": 16, "learning_rate": 2e-5}'
```

#### Option C — Standalone script
```bash
uv run python -m app.services.training
```

### 3 — Run the API
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

---

## API Reference

### `GET /api/v1/health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /api/v1/predict`
**Request**
```json
{ "text": "This agreement may be terminated by either party upon 30 days written notice." }
```
**Response**
```json
{ "label": "Termination", "confidence": 0.9821 }
```

### `POST /api/v1/train`
**Request** (all fields optional)
```json
{
  "num_train_epochs": 3,
  "per_device_train_batch_size": 16,
  "learning_rate": 2e-5,
  "test_size": 0.15
}
```
**Response** `202 Accepted`
```json
{
  "status": "training_started",
  "message": "Fine-tuning started in the background. Check GET /api/v1/health — when model_loaded=true the new model is ready."
}
```

---

## Dataset

- **Source**: [`alisha4walunj/quad_ledgar_merged_dataset`](https://huggingface.co/datasets/alisha4walunj/quad_ledgar_merged_dataset) (HuggingFace Hub)
- **Text column**: `provision`
- **Label column**: `label`
- **Model**: `distilbert-base-uncased`

---

## Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Inference latency | < 200 ms | Model loaded once at startup |
| Device | CPU / CUDA / MPS | Auto-detected at startup |
| Max input length | 512 tokens | Truncated automatically |

---

## Configuration

All settings can be overridden via `.env` or environment variables. See `.env.example` for the full list.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| API Framework | FastAPI + Uvicorn |
| Model | DistilBERT (HuggingFace Transformers) |
| Training | HuggingFace Trainer API |
| Dataset | HuggingFace Datasets |
| Metrics | `evaluate` (accuracy, weighted F1) |
| Dependency mgmt | `uv` |
| Settings | Pydantic Settings |
| PDF parsing (future) | PyMuPDF |
| LoRA fine-tuning (future) | PEFT |
