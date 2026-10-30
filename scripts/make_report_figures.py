"""Build report SVG figures using only Python's standard library."""
import csv
import gzip
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results' / 'cnn-small'
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def esc(value):
    return html.escape(str(value))


def line_plot(path, title, xlabel, ylabel, series, x_ticks, y_min=None, y_max=None):
    width, height = 900, 540
    left, right, top, bottom = 90, 35, 65, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    all_x = [x for _, pts, _ in series for x, _ in pts]
    all_y = [y for _, pts, _ in series for _, y in pts]
    xmin, xmax = min(all_x), max(all_x)
    ymin = min(all_y) if y_min is None else y_min
    ymax = max(all_y) if y_max is None else y_max
    pad = (ymax - ymin) * 0.05 or 1
    if y_min is None:
        ymin -= pad
    if y_max is None:
        ymax += pad

    def px(x): return left + (x - xmin) / max(xmax - xmin, 1) * plot_w
    def py(y): return top + (ymax - y) / max(ymax - ymin, 1e-9) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20">{esc(title)}</text>']
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = py(value)
        parts.append(f'<path d="M{left} {y:.1f}H{width-right}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11">{value:.4g}</text>')
    for x, label in x_ticks:
        xx = px(x)
        parts.append(f'<path d="M{xx:.1f} {top}V{height-bottom}" stroke="#eee"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="11">{esc(label)}</text>')
    parts += [f'<path d="M{left} {top}V{height-bottom}H{width-right}" fill="none" stroke="#333"/>',
              f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-size="13">{esc(xlabel)}</text>',
              f'<text x="20" y="{height/2}" transform="rotate(-90 20,{height/2})" text-anchor="middle" font-size="13">{esc(ylabel)}</text>']
    for idx, (name, pts, color) in enumerate(series):
        coordinates = ' '.join(f'{px(x):.1f},{py(y):.1f}' for x, y in pts)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        if len(pts) < 30:
            for x, y in pts:
                parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')
        lx = left + 15 + idx * 180
        parts.append(f'<path d="M{lx} {top+18}h25" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+32}" y="{top+22}" font-size="12">{esc(name)}</text>')
    parts += ['</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf8')


def rouge_chart():
    metrics = json.loads((RUN / 'test_metrics.json').read_text())['metrics']
    model = [metrics['rouge1'], metrics['rouge2'], metrics['rougeL'], metrics['rougeLsum']]
    lead = [40.06, 17.48, 25.02, 36.36]
    names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']
    width, height, left, top = 900, 420, 155, 55
    plot_w, max_value, row_h = 680, 45, 78
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-size="19">Test ROUGE F1: model vs Lead-3 reference</text>']
    for tick in range(0, 46, 5):
        x = left + tick / max_value * plot_w
        parts.append(f'<path d="M{x:.1f} {top-5}V{height-45}" stroke="#ddd"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-size="11">{tick}</text>')
    for i, name in enumerate(names):
        y = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y+27}" text-anchor="end" font-size="12">{name}</text>')
        for j, (value, color, label) in enumerate(((model[i], '#2864a5', 'CNN-small'), (lead[i], '#e28a36', 'Lead-3'))):
            yy = y + j * 25
            bar_w = value / max_value * plot_w
            parts.append(f'<rect x="{left}" y="{yy}" width="{bar_w:.1f}" height="18" fill="{color}"/>')
            parts.append(f'<text x="{left+bar_w+7:.1f}" y="{yy+14}" font-size="11">{value:.2f}</text>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-5}" text-anchor="middle" font-size="12">ROUGE F1 score (%)</text>',
              '<rect x="625" y="44" width="12" height="12" fill="#2864a5"/><text x="642" y="54" font-size="11">CNN-small</text>',
              '<rect x="725" y="44" width="12" height="12" fill="#e28a36"/><text x="742" y="54" font-size="11">Lead-3</text>',
              '</g></svg>']
    (OUT / 'rouge_comparison.svg').write_text('\n'.join(parts), encoding='utf8')


def length_chart():
    refs, preds = [], []
    word = re.compile(r"\b[\w']+\b")
    with gzip.open(RUN / 'test_predictions.jsonl.gz', 'rt', encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            refs.append(len(word.findall(row['reference'])))
            preds.append(len(word.findall(row['prediction'])))
    edges = list(range(0, 161, 20))
    def counts(values):
        result = [0] * (len(edges) - 1)
        for value in values:
            idx = min(value // 20, len(result) - 1)
            result[idx] += 1
        return [100 * n / len(values) for n in result]
    a, b = counts(refs), counts(preds)
    peak = max(a + b) or 1
    width, height, left, top = 900, 440, 80, 55
    plot_w, plot_h = 780, 300
    group_w = plot_w / len(a)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-size="19">Reference and generated summary lengths</text>']
    for i in range(5):
        value = peak * i / 4
        y = top + plot_h - value / peak * plot_h
        parts.append(f'<path d="M{left} {y:.1f}H{left+plot_w}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="10">{value:.1f}%</text>')
    for i in range(len(a)):
        x = left + i * group_w
        for j, (value, color) in enumerate(((a[i], '#2864a5'), (b[i], '#e28a36'))):
            bar_w = group_w * 0.36
            bx = x + group_w * (0.12 + j * 0.4)
            bh = value / peak * plot_h
            parts.append(f'<rect x="{bx:.1f}" y="{top+plot_h-bh:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+group_w/2:.1f}" y="{top+plot_h+19}" text-anchor="middle" font-size="10">{edges[i]}–{edges[i+1]}</text>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-size="12">Words per summary (final bin includes longer summaries)</text>',
              '<rect x="620" y="40" width="12" height="12" fill="#2864a5"/><text x="637" y="50" font-size="11">Reference</text>',
              '<rect x="715" y="40" width="12" height="12" fill="#e28a36"/><text x="732" y="50" font-size="11">Generated</text>',
              '</g></svg>']
    (OUT / 'summary_lengths.svg').write_text('\n'.join(parts), encoding='utf8')
    (OUT / 'summary_length_stats.json').write_text(json.dumps({
        'unit': 'whitespace-like word count via regex, not WordPiece tokens',
        'n': len(refs),
        'reference_mean_words': sum(refs) / len(refs),
        'prediction_mean_words': sum(preds) / len(preds),
        'reference_median_words': sorted(refs)[len(refs)//2],
        'prediction_median_words': sorted(preds)[len(preds)//2]
    }, indent=2))


def duration_chart():
    log = (ROOT / 'results' / 'host-cnn-small-training.log').read_text(errors='replace')
    rows = [(int(epoch), float(seconds) / 60) for epoch, seconds in re.findall(
        r'Epoch (\d+)/10 complete:.*?elapsed=([0-9.]+)s', log)]
    if not rows:
        return
    width, height, left, top = 850, 400, 75, 55
    plot_w, plot_h = 740, 270
    peak = max(v for _, v in rows) * 1.08
    group_w = plot_w / len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-size="19">Measured training time per epoch</text>']
    for i in range(5):
        value = peak * i / 4
        y = top + plot_h - value / peak * plot_h
        parts.append(f'<path d="M{left} {y:.1f}H{left+plot_w}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="10">{value:.1f}</text>')
    for epoch, minutes in rows:
        x = left + (epoch - 1) * group_w + group_w * 0.2
        bar_w = group_w * 0.6
        bh = minutes / peak * plot_h
        parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-bh:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="#55916d"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-size="10">{epoch}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h-bh-5:.1f}" text-anchor="middle" font-size="9">{minutes:.1f}</text>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-size="12">Epoch</text>',
              f'<text x="20" y="{height/2}" transform="rotate(-90 20,{height/2})" text-anchor="middle" font-size="12">Minutes</text>',
              '</g></svg>']
    (OUT / 'epoch_duration.svg').write_text('\n'.join(parts), encoding='utf8')


def architecture_diagram():
    width, height = 1050, 440
    blocks = [
        (25, 145, 145, 115, 'CNN/DailyMail\narticle', '#e9f1fb'),
        (205, 145, 155, 115, 'BERT WordPiece\n+ truncate to 400', '#f2f2f2'),
        (410, 65, 255, 125, 'Encoder × 3\nSelf-attention + FFN\nSinusoidal positions', '#dceeff'),
        (410, 225, 255, 125, 'Decoder × 3\nCausal self-attention\nCross-attention + FFN', '#fce8d5'),
        (750, 145, 135, 115, 'Greedy\nautoregressive\ndecoding', '#e8f4e3'),
        (920, 145, 115, 115, 'Summary\nmax 128 tokens', '#f3e6f4'),
    ]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20">From-scratch Transformer summarization pipeline</text>']
    for x1, x2, y in ((170, 205, 202), (360, 410, 145), (665, 750, 202), (885, 920, 202)):
        parts.append(f'<path d="M{x1} {y}H{x2}" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('<path d="M360 202H385V127H410M385 202V287H410" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('<path d="M665 127H700V202H750M665 287H700V202" fill="none" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#555"/></marker></defs>')
    for x, y, w, h, label, color in blocks:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" stroke="#4b5563"/>')
        lines = label.split('\n')
        start_y = y + h/2 - (len(lines)-1)*9
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x+w/2}" y="{start_y+i*19}" text-anchor="middle" font-size="12">{esc(line)}</text>')
    parts += ['</g></svg>']
    (OUT / 'architecture.svg').write_text('\n'.join(parts), encoding='utf8')


def main():
    epochs = {}
    learning = []
    with (RUN / 'history.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            e = int(row['epoch'])
            if row['train_loss']:
                stats = epochs.setdefault(e, {'train': [], 'val': None, 'ppl': None})
                stats['train'].append(float(row['train_loss']))
                if row['lr'] and int(row['step']) % 250 == 0:
                    learning.append((int(row['step']), float(row['lr'])))
            if row['val_loss']:
                stats = epochs.setdefault(e, {'train': [], 'val': None, 'ppl': None})
                stats['val'] = float(row['val_loss'])
                stats['ppl'] = float(row['val_perplexity'])
    train = [(e, sum(v['train']) / len(v['train'])) for e, v in sorted(epochs.items()) if v['train']]
    val = [(e, v['val']) for e, v in sorted(epochs.items()) if v['val'] is not None]
    ticks = [(e, str(e)) for e in sorted(epochs)]
    line_plot(OUT / 'epoch_losses.svg', 'Mean training and validation loss by epoch', 'Epoch', 'Cross-entropy loss',
              [('Train mean', train, '#2864a5'), ('Validation', val, '#c63d35')], ticks)
    line_plot(OUT / 'learning_rate.svg', 'Warmup and inverse-square-root learning-rate schedule', 'Optimizer step', 'Learning rate',
              [('Learning rate', learning, '#6a4c93')], [(250, '250'), (90000, '90k'), (180000, '180k'), (270000, '270k'), (358900, '359k')], y_min=0)
    rouge_chart()
    length_chart()
    architecture_diagram()
    duration_chart()
    print('Wrote report figures to', OUT)


if __name__ == '__main__':
    main()
