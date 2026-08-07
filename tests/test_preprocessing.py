import numpy as np
import pandas as pd

from tabpfn_ir.data import TabularPreprocessor


def test_preprocessor_keeps_model_and_retrieval_rows_aligned():
    train = pd.DataFrame(
        {
            "number": [0.0, 1.0, np.nan, 3.0],
            "category": pd.Series(["a", "b", "a", None], dtype="category"),
        }
    )
    test = pd.DataFrame(
        {
            "number": [100.0],
            "category": pd.Series(["unseen"], dtype="category"),
        }
    )
    preprocessor = TabularPreprocessor(["category"])
    train_views = preprocessor.fit_transform(train)
    test_views = preprocessor.transform(test)

    assert train_views.model.shape[0] == train_views.retrieval.shape[0] == len(train)
    assert test_views.model.shape[0] == test_views.retrieval.shape[0] == len(test)
    assert np.isfinite(train_views.model).all()
    assert np.isfinite(train_views.retrieval).all()
    assert test_views.retrieval[0, 0] > 10
