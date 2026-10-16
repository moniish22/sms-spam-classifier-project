"""Classify new messages with the saved model.

    python -m spamclf.predict "WINNER!! Claim your free prize now"
    python -m spamclf.predict --file my_messages.txt     # one SMS per line
    python -m spamclf.predict                            # interactive mode
"""
import argparse

import joblib

from . import config
from .evaluate import get_scores


def load_model(path=None):
    path = path or config.MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError("No trained model found. Run: python -m spamclf.train")
    return joblib.load(path)


def classify(model, messages):
    """Return list of (label, spam_score) for each message."""
    scores = get_scores(model, messages)
    return [("SPAM" if s >= 0.5 else "LEGIT", float(s)) for s in scores]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("messages", nargs="*")
    p.add_argument("--file")
    args = p.parse_args(argv)
    model = load_model()

    msgs = list(args.messages)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            msgs += [line.strip() for line in f if line.strip()]

    if msgs:
        for m, (label, score) in zip(msgs, classify(model, msgs)):
            print(f"[{label:5}] score={score:.3f} | {m}")
        return
    print("Interactive mode. Type an SMS (empty line to quit).")
    while True:
        m = input("> ").strip()
        if not m:
            break
        label, score = classify(model, [m])[0]
        print(f"  -> {label} (score {score:.3f})")


if __name__ == "__main__":
    main()
