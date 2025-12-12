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
parser.add_argument('--train-args', type=str, default='', help='Additional args passed to train.py')
args = parser.parse_args()

root = Path(__file__).resolve().parent
# ensure dirs
(root / 'data').mkdir(exist_ok=True)
(root / 'meta').mkdir(exist_ok=True)
(root / 'checkpoint').mkdir(exist_ok=True)
(root / 'pretrained_models').mkdir(exist_ok=True)

if args.download_dataset:
    print('Downloading dataset from', args.download_dataset)
    r = requests.get(args.download_dataset, stream=True)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(root)
    print('Dataset extracted')

import sys
# build train command using the current Python interpreter
cmd = [sys.executable, 'train.py']
if args.generate_demo:
    # export env var used by train.py to allow demo creation
    import os
    os.environ['GENERATE_DEMO'] = '1'
    cmd.append('--generate-demo')
if args.train_args:
    cmd.extend(args.train_args.split())

print('Running training:', ' '.join(cmd))
# inherit current environment so venv packages are visible
subprocess.check_call(cmd, env=None)
