"""
src/models/train.py
--------------------
Script to train a recommendation model.
"""

# src/models/train.py
import argparse
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_model(ratings_csv: str, test_size: float, n_factors: int):
    # 1. Load CSV into pandas
    df = pd.read_csv(ratings_csv)
    # 2. Create a Surprise dataset from the DataFrame
    reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
    data = Dataset.load_from_df(df[["user_id","item_id","rating"]], reader)
    # 3. Split
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    # 4. Train SVD
    algo = SVD(n_factors=n_factors, random_state=42)
    algo.fit(trainset)
    return algo, trainset, testset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ratings-csv", default="data/processed/ratings.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--factors", type=int, default=50)
    p.add_argument("--model-out", default="models/svd_model.pkl")
    p.add_argument("--test-out",  default="models/testset.pkl")
    args = p.parse_args()

    # Run training
    algo, trainset, testset = train_model(
        args.ratings_csv, args.test_size, args.factors
    )

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Save model
    with open(args.model_out, "wb") as f:
        pickle.dump(algo, f)
    # Save testset
    with open(args.test_out, "wb") as f:
        pickle.dump(testset, f)

    print(f"Trained SVD({args.factors}) on {trainset.n_ratings} ratings")
    print(f"Model saved to {args.model_out}")
    print(f"Testset saved to {args.test_out}")

