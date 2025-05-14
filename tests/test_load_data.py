# tests/test_load_data.py

import pandas as pd
import pytest

from src.data.load_data import load_movielens


def test_load_movielens_minimal(tmp_path):
    # Create a tiny u.data sample
    sample = tmp_path / "u.data"
    sample.write_text("1\t10\t4\t1234567890\n2\t20\t5\t9876543210\n")

    # Should load two rows, four columns
    df = load_movielens(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]

    # Bad path should raise
    with pytest.raises(FileNotFoundError):
        load_movielens(tmp_path / "no_such_dir")
