import argparse
import subprocess
from pathlib import Path
from src.logger import get_logger

logger = get_logger("cli")


def run_train(extra_args=None):
    cmd = ["python", "train.py"]
    if extra_args:
        cmd += extra_args
    logger.info(f"Running training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_infer(
    weight_file=None,
    image_dir=None,
    output=None,
    margin=0.4,
    batch_size=16,
    save_crops=False,
    crops_dir=None,
):
    from src.inference import predict_folder

    predict_folder(
        weight_file=weight_file,
        image_dir=image_dir,
        output_csv=output,
        margin=margin,
        batch_size=batch_size,
        save_crops=save_crops,
        crops_dir=crops_dir,
    )


def run_eval(results="results.csv", db="imdb", output="metrics.json"):
    from evaluate_predictions import compute_metrics

    metrics = compute_metrics(Path(results), Path("meta").joinpath(f"{db}.csv"))
    print(metrics)
    Path(output).write_text(str(metrics))


def main():
    parser = argparse.ArgumentParser(prog="age-gender-cli")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="Run training (delegates to train.py)")
    t.add_argument(
        "extra", nargs=argparse.REMAINDER, help="Extra args forwarded to train.py"
    )

    i = sub.add_parser("infer", help="Run inference")
    i.add_argument("--weight", type=str, default=None)
    i.add_argument("--image_dir", type=str, default=None)
    i.add_argument("--output", type=str, default=None)
    i.add_argument("--margin", type=float, default=0.4)
    i.add_argument("--batch_size", type=int, default=16)
    i.add_argument("--save_crops", action="store_true")
    i.add_argument("--crops_dir", type=str, default=None)

    e = sub.add_parser("evaluate", help="Evaluate predictions against meta csv")
    e.add_argument("--results", type=str, default="results.csv")
    e.add_argument("--db", type=str, default="imdb")
    e.add_argument("--output", type=str, default="metrics.json")

    args = parser.parse_args()

    if args.cmd == "train":
        run_train(args.extra)
    elif args.cmd == "infer":
        run_infer(
            weight_file=args.weight,
            image_dir=args.image_dir,
            output=args.output,
            margin=args.margin,
            batch_size=args.batch_size,
            save_crops=args.save_crops,
            crops_dir=args.crops_dir,
        )
    elif args.cmd == "evaluate":
        run_eval(results=args.results, db=args.db, output=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
