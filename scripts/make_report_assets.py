"""Write an index of completed Q3 report and result artifacts."""
from pathlib import Path


def main():
    root = Path('results')
    lines = [
        '# Q3 report and results index',
        '',
        'Detailed report: [`FINAL_REPORT.md`](../report/FINAL_REPORT.md)',
        '',
        'Figures: `report/figures/` (architecture, loss, learning rate, runtime, ROUGE, summary lengths).',
        '',
        'Training outputs, test metrics, predictions, qualitative review, and attention heatmap: `results/cnn-small/`.',
        '',
        'Dataset and checkpoints are excluded from the submission archive.',
    ]
    for metrics_path in sorted(root.glob('*/test_metrics.json')):
        lines += ['', f'## {metrics_path.parent.name}', '', '```json', metrics_path.read_text(), '```']
    (root / 'REPORT_ASSETS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
