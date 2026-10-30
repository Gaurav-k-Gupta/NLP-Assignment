"""Create offline post-training artifacts and run test evaluation when inputs exist."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "cnn-small"
CHECKPOINT = ROOT / "checkpoints" / "cnn-small_best.pt"
HISTORY = RUN / "history.csv"
STATUS = RUN / "POSTRUN_STATUS.md"


def read_points():
    train, val = [], []
    with (RUN / "history.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["train_loss"]:
                train.append((int(row["step"]), float(row["train_loss"])))
            if row["val_loss"]:
                val.append((int(row["step"]), float(row["val_loss"])))
    return train, val


def write_svg(train, val):
    w, h, left, right, top, bottom = 900, 500, 70, 25, 30, 55
    all_points = train + val
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymax = ymin + 1

    def coords(points):
        out = []
        for x, y in points:
            px = left + (x - xmin) / max(xmax - xmin, 1) * (w - left - right)
            py = h - bottom - (y - ymin) / (ymax - ymin) * (h - top - bottom)
            out.append(f"{px:.1f},{py:.1f}")
        return " ".join(out)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#222">',
           f'<text x="{w/2}" y="20" text-anchor="middle">CNN-small loss</text>',
           f'<path d="M{left} {top}V{h-bottom}H{w-right}" fill="none" stroke="#444"/>',
           f'<text x="{w/2}" y="{h-10}" text-anchor="middle">Step</text>',
           f'<text x="18" y="{h/2}" transform="rotate(-90 18,{h/2})" text-anchor="middle">Loss</text>',
           f'<polyline points="{coords(train)}" fill="none" stroke="#2369bd" stroke-width="1.5"/>',
           f'<polyline points="{coords(val)}" fill="none" stroke="#d04436" stroke-width="2"/>',
           f'<text x="{left+12}" y="{top+20}" fill="#2369bd">Train</text>',
           f'<text x="{left+12}" y="{top+40}" fill="#d04436">Validation</text>',
           '</g></svg>']
    (RUN / "loss_curves.svg").write_text("\n".join(svg), encoding="utf8")


def main():
    if not CHECKPOINT.exists() or not HISTORY.exists():
        STATUS.write_text("Post-run incomplete: best checkpoint or history file missing.\n")
        return 1

    train, val = read_points()
    if train and val:
        write_svg(train, val)
    best = min(val, key=lambda item: item[1]) if val else None
    lines = ["# CNN-small post-run status", "", f"Best checkpoint: `{CHECKPOINT}`"]
    if best:
        lines.append(f"Best validation loss: {best[1]:.6f} at step {best[0]}")

    test_file = Path(os.environ.get("DATA_DIR", ROOT / "offline_data")) / "test.jsonl.gz"
    missing = [name for name in ("rouge_score", "nltk") if importlib.util.find_spec(name) is None]
    if test_file.exists() and not missing:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["TOKENIZER_DIR"] = str(Path(os.environ.get("TOKENIZER_DIR", ROOT / "offline_data" / "tokenizer")))
        cmd = [sys.executable, "-m", "summarizer.evaluate", "--config", "configs/small.yaml",
               "--run-name", "cnn-small", "--checkpoint", str(CHECKPOINT)]
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode == 0:
            subprocess.run([sys.executable, "scripts/make_report_assets.py"], cwd=ROOT, env=env, check=False)
            subprocess.run([sys.executable, "scripts/qualitative.py", "--run-name", "cnn-small"], cwd=ROOT, env=env, check=False)
            lines.append("Test evaluation and report artifacts completed.")
        else:
            lines.append(f"Test evaluation failed with exit code {result.returncode}; see `logs/postrun.log`.")
    else:
        reason = "test.jsonl.gz missing" if not test_file.exists() else "missing packages: " + ", ".join(missing)
        lines.append(f"Test evaluation and qualitative examples skipped: {reason}.")

    STATUS.write_text("\n".join(lines) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
