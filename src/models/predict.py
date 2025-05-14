# src/models/predict.py
import pickle
from surprise import Dataset, Reader

def get_top_n(algo, trainset, user_id, n=10):
    # All items in the dataset
    all_items = set(trainset.all_items())
    # Convert surprise inner ids to raw ids
    raw_item_ids = {trainset.to_raw_iid(iid) for iid in all_items}
    # Items user has already rated
    seen = {trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]}
    # Predict score for each unseen
    preds = []
    for item in raw_item_ids - seen:
        preds.append((item, algo.predict(user_id, item).est))
    # Sort highest rating first
    top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    return top_n

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", required=True)
    p.add_argument("--user",       type=int, required=True)
    p.add_argument("--n",          type=int, default=10)
    p.add_argument("--trainset-file", required=False)
    args = p.parse_args()

    # Load model and trainset
    algo = pickle.load(open(args.model_file, "rb"))
    # If you saved trainset too, load it; otherwise re-split or reload full dataset
    # Here we assume you re-import trainset from your training code
    from src.models.train import train_model
    _, trainset, _ = train_model("data/processed/ratings.csv", test_size=0.2, n_factors=50)

    top_n = get_top_n(algo, trainset, args.user, args.n)
    print(f"Top {args.n} for user {args.user}:")
    for item, score in top_n:
        print(f"  Item {item}: {score:.3f}")
