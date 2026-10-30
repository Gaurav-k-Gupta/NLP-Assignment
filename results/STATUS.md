# CNN-small run status

Training and full test evaluation completed on the GPU host.
Detailed report, curated qualitative review, supplementary figures, and submission archive are now complete.

- Training: 10 epochs, all 287,113 training examples; 13,368 validation examples.
- Model: 13,343,232 parameters; `d_model=256`, 3 encoder and decoder layers, 8 heads; CUDA RTX 3060.
- Best validation loss: 4.4073 at step 358,900.
- Test: all 11,490 held-out examples; plain greedy decoding; BERT uncased tokenizer; article limit 400, summary limit 128.
- ROUGE F1: ROUGE-1 19.03, ROUGE-2 3.12, ROUGE-L 14.74, ROUGE-Lsum 17.40.

Run artifacts are in `results/cnn-small/`: training history and loss curves, test metrics and predictions, five distinct manually reviewed examples, attention data and heatmap, config, run metadata, and post-run status. The detailed report is `report/FINAL_REPORT.md`; supplementary plots are in `report/figures/`.

Checkpoints are in `results/cnn-small_best.pt` and `results/cnn-small_last.pt`. Git ignores checkpoint files because of their size.

Evaluation took about 17.5 minutes with batch size 8. Test scores are below the assignment's Lead-3 reference; see the report’s comparison and limitations discussion.

Submission archive: `submission.zip`. It contains source, report, and result summaries; dataset, full test predictions, and checkpoints are excluded.
