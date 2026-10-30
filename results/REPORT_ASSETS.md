# Q3 report and results index

Detailed report: [`FINAL_REPORT.md`](../report/FINAL_REPORT.md)

Figures: `report/figures/` (architecture, loss, learning rate, runtime, ROUGE, summary lengths).

Training outputs, test metrics, predictions, qualitative review, and attention heatmap: `results/cnn-small/`.

Dataset and checkpoints are excluded from the submission archive.

## cnn-small

```json
{
  "metrics": {
    "rouge1": 19.030657308150932,
    "rouge2": 3.116853521023708,
    "rougeL": 14.741502668646335,
    "rougeLsum": 17.404996939552362
  },
  "n_examples": 11490,
  "test_set_label": "full test set",
  "decoding": "plain greedy",
  "tokenizer": "bert-base-uncased",
  "max_source_length": 400,
  "max_target_length": 128,
  "checkpoint_step": 358900,
  "val_loss": 4.407252830312246
}
```
