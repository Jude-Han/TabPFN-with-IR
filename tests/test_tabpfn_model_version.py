import sys
from types import ModuleType

import pytest

from tabpfn_ir.models import ContextualTabPFNClassifier


def test_default_estimator_factory_explicitly_selects_v2_6(monkeypatch):
    calls = []
    sentinel = object()

    class FakeModelVersion:
        V2_6 = "v2.6-enum"
        V3 = "v3-enum"

    class FakeTabPFNClassifier:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            calls.append((version, kwargs))
            return sentinel

    tabpfn_module = ModuleType("tabpfn")
    tabpfn_module.TabPFNClassifier = FakeTabPFNClassifier
    constants_module = ModuleType("tabpfn.constants")
    constants_module.ModelVersion = FakeModelVersion
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn_module)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants_module)

    predictor = ContextualTabPFNClassifier(tabpfn_kwargs={"device": "cpu", "n_estimators": 4})

    assert predictor._new_estimator() is sentinel
    assert calls == [("v2.6-enum", {"device": "cpu", "n_estimators": 4})]

    predictor = ContextualTabPFNClassifier(
        tabpfn_kwargs={"device": "cpu", "n_estimators": 2},
        model_version="v3",
    )

    assert predictor._new_estimator() is sentinel
    assert calls[-1] == ("v3-enum", {"device": "cpu", "n_estimators": 2})


def test_adapter_rejects_a_checkpoint_override():
    try:
        ContextualTabPFNClassifier(tabpfn_kwargs={"model_path": "some-other-model.ckpt"})
    except ValueError as exc:
        assert "selected official TabPFN checkpoint" in str(exc)
    else:
        raise AssertionError("model_path override must not bypass the v2.6 checkpoint pin")


def test_adapter_accepts_all_supported_inference_versions_and_rejects_unknown():
    for model_version in ("v1", "v2.6", "v3"):
        predictor = ContextualTabPFNClassifier(
            estimator_factory=lambda: object(),
            model_version=model_version,
        )
        assert predictor.model_version == model_version

    with pytest.raises(ValueError, match="Unsupported TabPFN model version"):
        ContextualTabPFNClassifier(model_version="v4")
