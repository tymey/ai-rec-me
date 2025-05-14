"""
src/models/evaluate.py
-----------------------
Evaluate a trained recommendation model.
"""

import argparse
import pickle
from surprise import accuracy


def evaluate(model_file: str, test_file: str):
    # Load
    with open(model_file, "rb") as f:
        algo = pickle.load(f)
    with open(test_file, "rb") as f:
        testset = pickle.load(f)
    # Evaluate
    preds = algo.test(testset)
    accuracy.rmse(preds, verbose=True)  # rmse
    accuracy.mae(preds, verbose=True)  # mae


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", required=True)
    p.add_argument("--test-file", required=True)
    args = p.parse_args()
    evaluate(args.model_file, args.test_file)
