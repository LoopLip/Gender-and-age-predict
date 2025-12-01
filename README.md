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

Current timestamp: 2025-11-30T20:30:52.056Z
