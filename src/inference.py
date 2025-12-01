from pathlib import Path
import csv
import cv2
import dlib
import numpy as np
import sys
# ensure project root is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tensorflow.keras.utils import get_file
from omegaconf import OmegaConf
from src.factory import get_model
from src.logger import get_logger
from tqdm import tqdm

logger = get_logger("inference")

pretrained_model = None
modhash = None


def load_model(weight_file: str = None):
    if not weight_file:
        local = Path(__file__).resolve().parent.joinpath("pretrained_models", "EfficientNetB3_224_weights.11-3.44.hdf5")
        if local.exists():
            weight_file = str(local)
        else:
            raise FileNotFoundError(f"Weight file not provided and local pretrained model not found at {local}. Place weights there or pass --weight_file.")
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist(
        [f"model.model_name={model_name}", f"model.img_size={img_size}"]
    )
    model = get_model(cfg)
    model.load_weights(weight_file)
    return model, img_size


def detect_faces(detector, img_rgb, margin, img_size):
    img_h, img_w, _ = img_rgb.shape
    detected = detector(img_rgb, 1)
    faces = []
    boxes = []
    for d in detected:
        x1, y1, x2, y2 = d.left(), d.top(), d.right() + 1, d.bottom() + 1
        w, h = d.width(), d.height()
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), img_w - 1)
        yw2 = min(int(y2 + margin * h), img_h - 1)
        crop = cv2.resize(img_rgb[yw1 : yw2 + 1, xw1 : xw2 + 1], (img_size, img_size))
        faces.append(crop)
        boxes.append((x1, y1, x2, y2))
    return faces, boxes


def predict_folder(
    weight_file: str = None,
    image_dir: str = None,
    output_csv: str = None,
    margin: float = 0.4,
    batch_size: int = 16,
    save_crops: bool = False,
    crops_dir: str = None,
):
    base_dir = Path(__file__).resolve().parent
    image_dir = Path(image_dir) if image_dir else base_dir.joinpath("test_images")
    output_csv = Path(output_csv) if output_csv else base_dir.joinpath("results.csv")
    if save_crops:
        crops_dir = (
            Path(crops_dir) if crops_dir else base_dir.joinpath("predicted_crops")
        )
        crops_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model (may download weights if missing)...")
    model, img_size = load_model(weight_file)
    detector = dlib.get_frontal_face_detector()

    image_paths = sorted([p for p in image_dir.glob("*.*") if p.is_file()])
    if not image_paths:
        logger.warning(f"No images found in {image_dir}")
        return

    rows = []
    # process images sequentially but predict in batches of faces
    buffer_faces = []
    buffer_meta = []  # tuples (image_path, face_index)

    for image_path in tqdm(image_paths, desc="Images"):
        img_bgr = cv2.imread(str(image_path), 1)
        if img_bgr is None:
            logger.warning(f"Skipped unreadable: {image_path.name}")
            rows.append([image_path.name, "", "", "", "", "", "", ""])
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces, boxes = detect_faces(detector, img_rgb, margin, img_size)
        if not faces:
            logger.info(f"{image_path.name}: no faces detected")
            rows.append([image_path.name, "", "", "", "", "", "", ""])
            continue
        for idx, (face, box) in enumerate(zip(faces, boxes)):
            buffer_faces.append(face)
            buffer_meta.append((image_path.name, idx, box, image_path))
            # if buffer full, run prediction
            if len(buffer_faces) >= batch_size:
                _process_buffer(
                    model, buffer_faces, buffer_meta, rows, save_crops, crops_dir
                )
                buffer_faces = []
                buffer_meta = []
    # final buffer
    if buffer_faces:
        _process_buffer(model, buffer_faces, buffer_meta, rows, save_crops, crops_dir)

    # write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "face_index", "age", "gender", "x1", "y1", "x2", "y2"]
        )
        for r in rows:
            writer.writerow(r)
    logger.info(f"Results written to {output_csv}")


def _process_buffer(model, faces, metas, rows, save_crops, crops_dir):
    faces_np = np.asarray(faces)
    try:
        results = model.predict(faces_np)
    except Exception as e:
        logger.exception("Model prediction failed")
        # fallback: mark as failed
        for meta in metas:
            rows.append([meta[0], meta[1], "", "", "", "", "", ""])
        return
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()
    for i, (meta, age, gender) in enumerate(
        zip(metas, predicted_ages, predicted_genders)
    ):
        filename, face_index, box, image_path = meta
        g = "M" if gender[0] < 0.5 else "F"
        x1, y1, x2, y2 = box
        rows.append([filename, face_index, int(age), g, x1, y1, x2, y2])
        if save_crops:
            try:
                crop_path = crops_dir.joinpath(
                    f"{image_path.stem}_face{face_index}.jpg"
                )
                # faces were RGB resized to img_size; save as BGR
                crop_bgr = cv2.cvtColor(faces[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)
            except Exception:
                logger.exception(f"Failed saving crop for {filename} face {face_index}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch inference for age-gender-estimation (modernized)"
    )
    parser.add_argument("--weight_file", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_crops", action="store_true")
    parser.add_argument("--crops_dir", type=str, default=None)
    args = parser.parse_args()

    predict_folder(
        weight_file=args.weight_file,
        image_dir=args.image_dir,
        output_csv=args.output,
        margin=args.margin,
        batch_size=args.batch_size,
        save_crops=args.save_crops,
        crops_dir=args.crops_dir,
    )
