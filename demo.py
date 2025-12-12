from pathlib import Path
import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model, get_preprocess_fn
from src.logger import get_logger

logger = get_logger("demo")

pretrained_model = None
modhash = None


def get_args():
    parser = argparse.ArgumentParser(
        description="This script detects faces from web cam input, "
        "and estimates age and gender for the detected faces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weight_file",
        type=str,
        default=None,
        help="path to weight file (e.g. weights.28-3.73.hdf5)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.4,
        help="margin around detected face for age-gender estimation",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="target image directory; if set, images in image_dir are used instead of webcam",
    )
    args = parser.parse_args()
    return args


def draw_label(
    image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1
):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(
        image,
        label,
        point,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


def main():
    args = get_args()
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        local = Path(__file__).resolve().parent.joinpath("pretrained_models", "EfficientNetB3_224_weights.11-3.44.hdf5")
        if local.exists():
            weight_file = str(local)
        else:
            # try to download weights from environment or config
            download_url = os.environ.get('PRETRAINED_WEIGHTS_URL')
            try:
                cfg = OmegaConf.load(Path(__file__).resolve().parent.joinpath('src', 'config.yaml'))
                # support config key: demo.weights_url
                cfg_url = None
                try:
                    cfg_url = cfg.demo.weights_url
                except Exception:
                    cfg_url = None
                if not download_url and cfg_url:
                    download_url = cfg_url
            except Exception:
                download_url = download_url

            if download_url:
                logger.info(f"Attempting to download pretrained weights from {download_url}")
                try:
                    cache_dir = Path(__file__).resolve().parent.joinpath('pretrained_models')
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    fname = download_url.split('/')[-1].split('?')[0]
                    # use keras get_file to download into pretrained_models
                    downloaded = get_file(fname, download_url, cache_dir=str(cache_dir), cache_subdir='')
                    weight_file = str(downloaded)
                    logger.info(f"Downloaded weights to {weight_file}")
                except Exception:
                    logger.exception("Failed to download pretrained weights")
                    print("Failed to download weights; provide via --weight_file or place pretrained model in pretrained_models/")
                    return
            else:
                logger.warning(
                    f"Weight file not provided and local pretrained model not found at {local}. Demo will not run inference without weights."
                )
                print("Provide weights via --weight_file or place pretrained model in pretrained_models/ to run demo.")
                return

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist(
        [f"model.model_name={model_name}", f"model.img_size={img_size}"]
    )
    model = get_model(cfg)
    model.load_weights(weight_file)
    preprocess_fn = get_preprocess_fn(cfg)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = []

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = (
                    d.left(),
                    d.top(),
                    d.right() + 1,
                    d.bottom() + 1,
                    d.width(),
                    d.height(),
                )
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                face = cv2.resize(
                    input_img[yw1 : yw2 + 1, xw1 : xw2 + 1], (img_size, img_size)
                )
                faces.append(face)

            # prepare batch and predict ages and genders of the detected faces
            faces_np = np.asarray(faces).astype(np.float32)
            if preprocess_fn:
                try:
                    faces_np = np.asarray([preprocess_fn(f) for f in faces_np])
                except Exception:
                    logger.exception(
                        "preprocess_fn failed; falling back to 0-1 scaling"
                    )
                    faces_np = faces_np / 255.0

            results = model.predict(faces_np)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(
                    int(predicted_ages[i]),
                    "M" if predicted_genders[i][0] < 0.5 else "F",
                )
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == "__main__":
    main()
