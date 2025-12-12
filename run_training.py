"""
run_training.py - helper wrapper to prepare directories, optionally download dataset, and run training.
Usage:
  python run_training.py [--download-dataset URL] [--generate-demo]
"""
import argparse
import subprocess
from pathlib import Path
import requests
import zipfile
import io

parser = argparse.ArgumentParser()
parser.add_argument('--download-dataset', type=str, default=None, help='URL to download dataset zip (optional)')
parser.add_argument('--generate-demo', action='store_true', help='Generate a small demo dataset for testing')
args = parser.parse_args()

root = Path(__file__).resolve().parent
# ensure dirs
(root / 'data').mkdir(exist_ok=True)
(root / 'meta').mkdir(exist_ok=True)
(root / 'checkpoint').mkdir(exist_ok=True)
(root / 'pretrained_models').mkdir(exist_ok=True)

if args.download_dataset:
    print('Downloading dataset from', args.download_dataset)
    r = requests.get(args.download_dataset)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(root)
    print('Dataset extracted')

# build train command
cmd = ['python', 'train.py']
if args.generate_demo:
    cmd.append('--generate-demo')

print('Running training:', ' '.join(cmd))
subprocess.check_call(cmd)
