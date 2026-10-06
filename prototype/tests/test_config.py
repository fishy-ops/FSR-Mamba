"""Architecture inference, sidecars, and strict checkpoint reconstruction."""

import argparse
import inspect
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsrmamba.config import (ARCH_DEFAULTS, add_arch_args, build_model, infer_config,
                             load_checkpoint, load_sidecar, model_kwargs, save_sidecar)
from fsrmamba.mamba import MambaAccumulator


def shapes(sd):
    return {key: tuple(value.shape) for key, value in sd.items()}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    common = dict(rectify=True, robust_disocc=True, kernel_predict=True,
                  lr_upsampler=True, feature_channels=48, state_channels=24, lr_dim=64)
    cases = [common, dict(common, separable=True), dict(unet=2, kernel_predict=True),
             dict(lock_feature=True), dict(lr_upsampler=True, lr_film=True),
             dict(num_experts=3), dict(encoder_depth=3),
             dict(learned_resolve=True, sharpen=True), dict(swin_resolve=True, swin_dim=24),
             dict(unet=2, separable=True, lock_feature=True)]
    accepted = inspect.signature(MambaAccumulator.__init__).parameters
    if "history_input" in accepted:
        cases.extend([dict(history_input=True), dict(history_input=True, lock_feature=True)])
    if "unet_unshuffle" in accepted:
        cases.extend([dict(unet=2, unet_unshuffle=True),
                      dict(unet=2, unet_unshuffle=True, lock_feature=True)])
        if "history_input" in accepted:
            cases.extend([dict(unet=2, unet_unshuffle=True, history_input=True),
                          dict(unet=2, unet_unshuffle=True, history_input=True, lock_feature=True)])
    with tempfile.TemporaryDirectory() as directory:
        for i, cfg in enumerate(cases):
            model = build_model(cfg, (8, 12), (16, 24))
            path = Path(directory) / f"model_{i}.pt"
            torch.save(model.state_dict(), path)
            sd = torch.load(path, weights_only=True)
            inferred = infer_config(sd)
            rebuilt = build_model(inferred, (8, 12), (16, 24))
            assert not rebuilt.training
            assert shapes(sd) == shapes(rebuilt.state_dict()), (cfg, inferred)
            rebuilt.load_state_dict(sd, strict=True)
            loaded, loaded_cfg = load_checkpoint(path, (8, 12), (16, 24))
            assert shapes(loaded.state_dict()) == shapes(sd)
            assert loaded_cfg == inferred
            save_sidecar(path, argparse.Namespace(**cfg))
            expected = dict(ARCH_DEFAULTS, arch="mamba", **cfg)
            assert load_sidecar(path) == expected
            loaded, loaded_cfg = load_checkpoint(path, (8, 12), (16, 24))
            assert loaded_cfg == expected
            for key, value in sd.items():
                assert torch.equal(loaded.state_dict()[key], value)
            _, override_cfg = load_checkpoint(path, (8, 12), (16, 24),
                                              overrides={"robust_disocc": False, "alpha_scale": 0.7})
            assert override_cfg["robust_disocc"] is False and override_cfg["alpha_scale"] == 0.7
        assert load_sidecar(Path(directory) / "missing.pt") is None
        legacy = build_model({}, (8, 12), (16, 24)).state_dict()
        assert infer_config(legacy)["robust_disocc"] is True
        custom = {"robust_disocc": False, "box_max": 1.8, "alpha_scale": 0.5, "ablate_ssm": True}
        assert all(infer_config(legacy, custom)[k] == v for k, v in custom.items())
        broken = dict(legacy)
        broken["encoder.0.weight"] = torch.zeros(24, 14, 3, 3)
        try:
            infer_config(broken)
        except ValueError as exc:
            assert "14" in str(exc) and "encoder.0" in str(exc)
        else:
            raise AssertionError("Unsupported input shape accepted")
        broken_path = Path(directory) / "broken.pt"
        broken = dict(legacy)
        broken.pop("align_shift")
        torch.save(broken, broken_path)
        try:
            load_checkpoint(broken_path, (8, 12), (16, 24))
        except RuntimeError as exc:
            assert "align_shift" in str(exc)
        else:
            raise AssertionError("Strict load accepted missing parameter")
    with patch("fsrmamba.config.inspect.signature", return_value=inspect.Signature()):
        assert model_kwargs({}) == {"arch": "mamba"}
        try:
            model_kwargs({"history_input": True})
        except ValueError as exc:
            assert "history_input" in str(exc)
        else:
            raise AssertionError("Unsupported feature accepted")
    ap = argparse.ArgumentParser()
    add_arch_args(ap)
    defaults = vars(ap.parse_args([]))
    assert all(defaults[key] == value for key, value in ARCH_DEFAULTS.items())
    assert defaults["arch"] == "mamba"
    add_arch_args(ap)
    assert vars(ap.parse_args([])) == defaults
    assert ap.parse_args(["--kernel-predict", "--state-channels", "24"]).kernel_predict
    print("test_config passed")


if __name__ == "__main__":
    main()
