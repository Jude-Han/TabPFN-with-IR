from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tabpfn_ir.models.tabpfn_v1 import LegacyTabPFNClassifier


def _write_fake_v1_runtime(runtime: Path, *, with_checkpoint: bool = True) -> None:
    package = runtime / "tabpfn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
import numpy as np

class TabPFNClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, overwrite_warning=False):
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([
            np.asarray(X)[np.asarray(y) == label].mean(axis=0)
            for label in self.classes_
        ])
        return self

    def predict_proba(self, X):
        distances = np.linalg.norm(
            np.asarray(X)[:, None, :] - self.centroids_[None, :, :], axis=2
        )
        scores = np.exp(-distances)
        return scores / scores.sum(axis=1, keepdims=True)
""".lstrip(),
        encoding="utf-8",
    )
    if with_checkpoint:
        models = package / "models_diff"
        models.mkdir()
        (models / "prior_diff_real_checkpoint_n_0_epoch_100.cpkt").touch()


def test_v1_proxy_reuses_an_isolated_worker(tmp_path):
    runtime = tmp_path / "runtime"
    _write_fake_v1_runtime(runtime)
    classifier = LegacyTabPFNClassifier(
        runtime_path=runtime,
        device="cpu",
        n_estimators=1,
    )
    try:
        classifier.fit(
            np.asarray([[0.0], [0.2], [9.8], [10.0]]),
            np.asarray([0, 0, 1, 1]),
        )
        probabilities = classifier.predict_proba(np.asarray([[0.1], [9.9]]))
        assert classifier.classes_.tolist() == [0, 1]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert probabilities.argmax(axis=1).tolist() == [0, 1]
        process = classifier._process

        classifier.fit(
            np.asarray([[0.0], [1.0], [4.0], [5.0]]),
            np.asarray([0, 0, 1, 1]),
        )
        assert classifier._process is process
    finally:
        classifier.close()


def test_v1_proxy_reports_missing_runtime(tmp_path):
    classifier = LegacyTabPFNClassifier(runtime_path=tmp_path / "missing")
    with pytest.raises(RuntimeError, match="setup_tabpfn_v1_runtime"):
        classifier.fit(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]))


def test_v1_proxy_reports_missing_default_checkpoint(tmp_path):
    runtime = tmp_path / "runtime"
    _write_fake_v1_runtime(runtime, with_checkpoint=False)
    classifier = LegacyTabPFNClassifier(runtime_path=runtime)
    with pytest.raises(RuntimeError, match="default TabPFN v1 checkpoint is missing"):
        classifier.fit(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]))
