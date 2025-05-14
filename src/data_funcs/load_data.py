"""
src/data/load_data.py
---------------------
Functions to load and preprocess datasets.
"""

from pathlib import Path
import pandas as pd

def load_movielens(path: str | Path) -> pd.DataFrame:
    """
    Load the MovieLens-100K ratings from the folder at `path`
    and return a DataFrame with columns:
     - user_id (int)
     - item_id (int)
     - rating  (int)
     - timestamp (int)
    """
    path = Path(path)
    data_file = path / "u.data"
    if not data_file.exists():
      raise FileNotFoundError(f"No file at {data_file}")
    
    # u.data is tab-separated, no header; columns are in order user, item, rating, timestamp
    df = pd.read_csv(
        data_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": int,
            "item_id": int,
            "rating": int,
            "timestamp": int,
        },
    )
    return df

def save_processed(df: pd.DataFrame, out_path: str | Path) -> None:
   out_path = Path(out_path)
   out_path.parent.mkdir(parents=True, exist_ok=True)
   # keep only user_id, item_id, rating
   df[["user_id", "item_id", "rating"]].to_csv(out_path, index=False)

if __name__ == "__main__":
    # Example CLI usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Folder of MovieLens files")
    args = parser.parse_args()

    df = load_movielens(args.path)
    print(df.head())
