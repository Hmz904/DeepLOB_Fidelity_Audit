import torch

from deeplob_replication.utils import environment_manifest, set_seed


def test_strict_deterministic_mode_is_enabled():
    set_seed(123, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()


def test_environment_manifest_contains_reproducibility_versions():
    env = environment_manifest()
    for key in ("python", "numpy", "pandas", "scikit_learn", "torch"):
        assert env[key]
