import pytest

from tabpfn_ir.models import build_tabpfn_classifier_kwargs


def test_build_tabpfn_kwargs_supports_four_explicit_gpus():
    kwargs = build_tabpfn_classifier_kwargs(
        device="auto",
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        n_estimators=8,
    )

    assert kwargs == {
        "device": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "ignore_pretraining_limits": True,
        "fit_mode": "fit_preprocessors",
        "n_estimators": 8,
    }


def test_multi_gpu_rejects_fit_with_cache():
    with pytest.raises(ValueError, match="Multiple GPUs require"):
        build_tabpfn_classifier_kwargs(
            device="auto",
            devices=["cuda:0", "cuda:1"],
            fit_mode="fit_with_cache",
        )


def test_tabpfn_configuration_rejects_duplicate_devices_and_invalid_estimators():
    with pytest.raises(ValueError, match="duplicate"):
        build_tabpfn_classifier_kwargs(
            device="auto",
            devices=["cuda:0", "cuda:0"],
        )
    with pytest.raises(ValueError, match="positive"):
        build_tabpfn_classifier_kwargs(device="auto", n_estimators=0)
