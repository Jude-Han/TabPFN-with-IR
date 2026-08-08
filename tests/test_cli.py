import pytest

from scripts.run_baseline import resolve_context_size


def test_resolve_context_size_accepts_fixed_and_localpfn_values():
    assert resolve_context_size("128", n_train=1_000) == 128
    assert resolve_context_size("localpfn", n_train=10_000) == 1_000
    assert resolve_context_size("localpfn", n_train=25) == 25
    assert resolve_context_size("localpfn", n_train=10_000, maximum=512) == 512


def test_resolve_context_size_caps_fixed_value_at_training_size():
    assert resolve_context_size("128", n_train=50) == 50


def test_resolve_context_size_rejects_invalid_value():
    with pytest.raises(ValueError):
        resolve_context_size("zero", n_train=1_000)
