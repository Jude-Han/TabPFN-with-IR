import gzip
import json

import numpy as np

from scripts.run_benchmark import _subsample_query_indices
from tabpfn_ir.data import (
    discover_localpfn_dataset_directories,
    load_openml_manifest,
    load_tabzilla_dataset,
    tabpfn_v1_split_indices,
)


def _write_gzip_numpy(path, value):
    with gzip.GzipFile(path, "wb") as stream:
        np.save(stream, value)


def _write_tabzilla_dataset(
    root,
    name,
    *,
    target_type="classification",
    num_features=2,
    num_classes=3,
):
    path = root / name
    path.mkdir()
    X = np.asarray(
        [[0, 0.0], [1, 0.2], [2, 0.4], [0, 0.6], [1, 0.8], [2, 1.0]],
        dtype=object,
    )
    y = np.asarray([0, 1, 2, 0, 1, 2])
    splits = np.asarray(
        [
            {
                "train": np.asarray([0, 1, 2, 3]),
                "val": np.asarray([4]),
                "test": np.asarray([5]),
            }
        ],
        dtype=object,
    )
    _write_gzip_numpy(path / "X.npy.gz", X)
    _write_gzip_numpy(path / "y.npy.gz", y)
    _write_gzip_numpy(path / "split_indeces.npy.gz", splits)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "name": name,
                "cat_idx": [0],
                "cat_dims": [3],
                "target_type": target_type,
                "num_classes": num_classes,
                "num_features": num_features,
                "num_instances": len(y),
                "split_source": "test",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_tabpfn_v1_manifest_contains_the_fixed_30_datasets():
    manifest = load_openml_manifest("data/manifests/tabpfn_v1_30.json")

    assert manifest.benchmark == "tabpfn-v1-30"
    assert len(manifest.datasets) == 30
    assert len({entry.dataset_id for entry in manifest.datasets}) == 30
    assert {entry.dataset_id for entry in manifest.datasets} >= {11, 31, 40994}


def test_tabpfn_v1_protocol_builds_five_disjoint_half_splits():
    y = np.repeat(np.arange(3), 20)
    splits = tabpfn_v1_split_indices(y, random_state=7)

    assert len(splits) == 5
    for split in splits:
        assert len(split.train) == 30
        assert len(split.test) == 30
        assert len(split.validation) == 0
        assert not np.intersect1d(split.train, split.test).size
        assert np.array_equal(np.unique(y[split.train], return_counts=True)[1], [10, 10, 10])


def test_load_tabzilla_dataset_preserves_stored_folds_and_categories(tmp_path):
    path = _write_tabzilla_dataset(tmp_path, "openml__synthetic__123")

    dataset = load_tabzilla_dataset(path)

    assert dataset.task_id == 123
    assert dataset.X.shape == (6, 2)
    assert dataset.categorical_columns == ("feature_0",)
    assert len(dataset.splits) == 1
    assert dataset.splits[0].train.tolist() == [0, 1, 2, 3]
    assert dataset.splits[0].validation.tolist() == [4]
    assert dataset.splits[0].test.tolist() == [5]


def test_localpfn_discovery_applies_official_filters(tmp_path):
    included = _write_tabzilla_dataset(tmp_path, "openml__included__1")
    _write_tabzilla_dataset(tmp_path, "openml__regression__2", target_type="regression")
    _write_tabzilla_dataset(tmp_path, "openml__wide__3", num_features=101)
    _write_tabzilla_dataset(tmp_path, "openml__many-classes__4", num_classes=11)
    _write_tabzilla_dataset(tmp_path, "openml__cjs__14967")

    selected = discover_localpfn_dataset_directories(tmp_path)

    assert selected == [included]


def test_query_smoke_subsample_keeps_every_class():
    y = np.asarray([0, 0, 0, 1, 1, 2, 2, 2])

    selected = _subsample_query_indices(
        np.arange(len(y)),
        4,
        seed=3,
        y=y,
    )

    assert len(selected) == 4
    assert set(y[selected]) == {0, 1, 2}
