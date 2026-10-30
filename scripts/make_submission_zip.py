"""Build source/report submission archive without data, predictions, or checkpoints."""
import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='submission.zip')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    allowed = [
        'README.md', 'requirements.txt', '.gitignore', 'configs', 'docs', 'src',
        'scripts', 'tests', 'notebooks', 'report', 'results'
    ]
    excluded_dirs = {'.git', 'checkpoints', 'cache', 'data', 'offline_data', '__pycache__', 'host-cnn-small'}
    excluded_suffixes = {'.pt', '.pth', '.ckpt'}
    excluded_names = {'.DS_Store', 'test_predictions.jsonl.gz', 'loss_curves.png'}

    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in allowed:
            path = root / item
            if not path.exists():
                continue
            files = [path] if path.is_file() else path.rglob('*')
            for file in files:
                if not file.is_file() or file.resolve() == output:
                    continue
                relative = file.relative_to(root)
                if any(part in excluded_dirs for part in relative.parts):
                    continue
                if file.name in excluded_names or file.suffix in excluded_suffixes:
                    continue
                archive.write(file, relative)

    print(output)


if __name__ == '__main__':
    main()
