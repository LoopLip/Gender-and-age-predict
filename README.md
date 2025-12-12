age-gender-estimation (modernized)

Quick start

1) Create and activate virtual environment (Windows PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

2) Install dependencies:
   pip install -r requirements.txt
   # On Windows installing dlib often requires MS Build Tools or use conda: conda install -c conda-forge dlib

3) Run batch inference:
   python -m src.inference --image_dir test_images --output results.csv --batch_size 16 --save_crops

4) Train (example):
   python train.py

What's new (summary):
- Improved inference: batched predictions, logging, optional saving of face crops (src/inference.py).
- Training modernized: deterministic seed, backbone preprocessing, better generator, callbacks (TensorBoard, ReduceLROnPlateau, EarlyStopping).
- Demo updated to use backbone preprocessing and structured logging.

See MODERNIZATION.md for full details and next steps.

Note: demo.py can auto-download pretrained weights if a URL is provided via the environment variable PRETRAINED_WEIGHTS_URL or via src/config.yaml under demo.weights_url. Example:
  set PRETRAINED_WEIGHTS_URL=https://example.com/path/to/EfficientNetB3_224_weights.hdf5

Current timestamp: 2025-12-12T20:31:44.624Z
