import sys
from types import ModuleType

from tabpfn_ir.models import ContextualTabPFNClassifier


def test_default_estimator_factory_explicitly_selects_v2_6(monkeypatch):
    calls = []
    sentinel = object()

    class FakeModelVersion:
        V2_6 = "v2.6-enum"

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

    predictor = ContextualTabPFNClassifier(
        tabpfn_kwargs={"device": "cpu", "n_estimators": 4}
    )

    assert predictor._new_estimator() is sentinel
    assert calls == [("v2.6-enum", {"device": "cpu", "n_estimators": 4})]


def test_adapter_rejects_a_checkpoint_override():
    try:
        ContextualTabPFNClassifier(tabpfn_kwargs={"model_path": "some-other-model.ckpt"})
    except ValueError as exc:
        assert "v2.6" in str(exc)
    else:
        raise AssertionError("model_path override must not bypass the v2.6 checkpoint pin")
