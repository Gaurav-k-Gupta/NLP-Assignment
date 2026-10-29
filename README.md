# From-scratch Transformer abstractive summarizer (Q3)

This repository implements Question 3 of NLP Group Assignment 2: CNN/DailyMail 3.0.0 summarization with a hand-built PyTorch encoder-decoder Transformer. The Hugging Face dependency is used only to load `bert-base-uncased` tokenizer files. There are no pretrained model weights.

## Design

See [docs/DESIGN.md](docs/DESIGN.md) for tensor dimensions, mask semantics, formula mapping, and explicitly estimated T4 sizing. Defaults use a 400-token article, 128-token summary, d_model=256, 8 heads, three encoder and decoder layers, FFN width 1024, post-LN, shared tied embeddings, Adam (0.9, 0.98), and warmup/inverse-square-root learning rate. Summaries retain newline sentence boundaries through a reserved WordPiece id. Validation CE selects the best checkpoint; test scores come only from greedy autoregressive predictions.

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python scripts/check_forbidden_apis.py
python -m summarizer.train --config configs/smoke.yaml --run-name smoke --max-train-samples 8 --max-val-samples 4 --epochs 1
```

The smoke command downloads the tokenizer and data. If network access is unavailable, the full data pipeline cannot run; component tests remain network-free. For Colab, install requirements (keeping Colab's torch), set `CKPT_DIR` and `TOKEN_CACHE_DIR` to persistent Drive paths, then run:

```bash
!nvidia-smi
from google.colab import drive; drive.mount('/content/drive')
%env CKPT_DIR=/content/drive/MyDrive/GA2/checkpoints
%env TOKEN_CACHE_DIR=/content/drive/MyDrive/GA2/cache
!git clone <your-repository-url> /content/GA2  # or git -C /content/GA2 pull
%cd /content/GA2
!pip install -r requirements.txt
!export PYTHONPATH=src && pytest -q && python scripts/check_forbidden_apis.py
!export PYTHONPATH=src && python -m summarizer.train --config configs/smoke.yaml --run-name smoke --max-train-samples 32 --max-val-samples 16 --epochs 1
!export PYTHONPATH=src && python scripts/benchmark.py --config configs/small.yaml
!export PYTHONPATH=src && python -m summarizer.train --config configs/small.yaml --run-name cnn-small --time-budget-minutes 300
!python scripts/plot_curves.py cnn-small
!export PYTHONPATH=src && python -m summarizer.evaluate --config configs/small.yaml --run-name cnn-small --checkpoint checkpoints/cnn-small_best.pt
!python scripts/make_report_assets.py
!python scripts/make_submission_zip.py
```

Training resumes from `checkpoints/<run>_last.pt`; best validation weights are kept separately. Use `--max-train-samples`, `--max-val-samples`, `--max-test-samples` for controlled subsets. Never use test data for decisions. Results are written to `results/<run>/`; missing measurements are reported as NOT RUN YET.

## Repository map

`src/summarizer/model/` contains attention, positional encoding, layers and full model; `data.py` and `tokenization.py` handle data; `train.py`, `generate.py`, and `evaluate.py` implement the workflow. `scripts/` contains checks and report utilities. `tests/` holds CPU unit tests. `notebooks/colab_run.ipynb` mirrors the run order. Dataset caches and checkpoints are git-ignored.

## Results

No experiments are represented as completed until code writes their artifacts. Current result status: **NOT RUN YET**.
