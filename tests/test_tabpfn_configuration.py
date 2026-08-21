import pytest

from tabpfn_ir.models import (
    build_tabpfn_classifier_kwargs,
    resolve_n_estimators,
    validate_model_version,
)


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


def test_v1_rejects_multiple_devices():
    with pytest.raises(ValueError, match="v1 supports exactly one device"):
        build_tabpfn_classifier_kwargs(
            device="auto",
            model_version="v1",
            devices=["cuda:0", "cuda:1"],
        )


def test_finetuning_accepts_modern_versions_but_not_v1():
    assert validate_model_version("v2.6", finetuning=True) == "v2.6"
    assert validate_model_version("v3", finetuning=True) == "v3"
    with pytest.raises(ValueError, match="fine-tuning"):
        validate_model_version("v1", finetuning=True)


def test_tabpfn_configuration_rejects_duplicate_devices_and_invalid_estimators():
    with pytest.raises(ValueError, match="duplicate"):
        build_tabpfn_classifier_kwargs(
            device="auto",
            devices=["cuda:0", "cuda:0"],
        )
    with pytest.raises(ValueError, match="positive"):
        build_tabpfn_classifier_kwargs(device="auto", n_estimators=0)


def test_default_inference_profile_preserves_requested_estimators():
    assert resolve_n_estimators(inference_profile="default", n_estimators=8) == 8
    assert resolve_n_estimators(inference_profile="default", n_estimators=None) is None


def test_single_estimator_profile_overrides_requested_ensemble_size():
    assert (
        resolve_n_estimators(
            inference_profile="single-estimator",
            n_estimators=8,
        )
        == 1
    )


def test_unknown_inference_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown inference profile"):
        resolve_n_estimators(inference_profile="unknown", n_estimators=None)
