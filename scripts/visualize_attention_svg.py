"""Render saved autoregressive cross-attention as a dependency-light SVG heatmap."""
import argparse
import html
import json
from pathlib import Path

from summarizer.data import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='run')
    parser.add_argument('--source-window', type=int, default=75)
    args = parser.parse_args()

    root = Path('results') / args.run_name
    item = json.loads((root / 'attention_example.json').read_text())
    tokenizer = get_tokenizer()
    source_ids = [token for token in item['source_ids'] if token != tokenizer.pad_token_id]
    source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
    generated_ids = []
    for token in item['generated_ids']:
        if token in (tokenizer.sep_token_id, tokenizer.pad_token_id):
            break
        if token != tokenizer.cls_token_id:
            generated_ids.append(token)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    heads_by_step = item['attention'][:len(generated_tokens)]
    matrix = []
    for step in heads_by_step:
        rows = [head[:len(source_tokens)] for head in step]
        if not rows:
            continue
        matrix.append([sum(head[col] for head in rows) / len(rows)
                       for col in range(len(source_tokens))])
    generated_tokens = generated_tokens[:len(matrix)]
    if not matrix or not source_tokens:
        raise ValueError('Saved attention has no generated or source tokens')

    width = min(max(args.source_window, 1), len(source_tokens))
    start = max(range(len(source_tokens) - width + 1),
                key=lambda i: sum(sum(row[i:i + width]) for row in matrix))
    visible_tokens = source_tokens[start:start + width]
    visible = [row[start:start + width] for row in matrix]
    max_value = max(max(row) for row in visible) or 1.0
    window_mass = sum(sum(row) for row in visible) / max(sum(sum(row) for row in matrix), 1e-12)

    cell_w, cell_h, left, top = 16, 18, 190, 55
    label_h = 150
    svg_w = left + width * cell_w + 20
    svg_h = top + len(visible) * cell_h + label_h + 20
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#222">',
             f'<text x="{svg_w / 2:.1f}" y="22" text-anchor="middle" font-size="15">Greedy-generation cross-attention; mean over {len(heads_by_step[0])} heads</text>',
             f'<text x="{svg_w / 2:.1f}" y="42" text-anchor="middle" font-size="12">Last decoder layer; source window {start + 1}–{start + width}; attention mass {window_mass:.3f}</text>']

    for row_i, (label, values) in enumerate(zip(generated_tokens, visible)):
        y = top + row_i * cell_h
        parts.append(f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" font-size="10">{html.escape(label)}</text>')
        for col_i, value in enumerate(values):
            strength = max(0.0, min(1.0, value / max_value))
            red = round(248 * (1 - strength) + 8 * strength)
            green = round(251 * (1 - strength) + 48 * strength)
            blue = round(255 * (1 - strength) + 107 * strength)
            color = f'#{red:02x}{green:02x}{blue:02x}'
            x = left + col_i * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="white" stroke-width="0.3"/>')

    axis_y = top + len(visible) * cell_h
    for col_i, label in enumerate(visible_tokens):
        x = left + col_i * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{axis_y + 7}" transform="rotate(60 {x} {axis_y + 7})" text-anchor="start" font-size="8">{html.escape(label)}</text>')
    parts += ['</g></svg>']
    out = root / 'attention_mean_heads.svg'
    out.write_text('\n'.join(parts), encoding='utf8')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
