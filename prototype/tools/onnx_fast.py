"""Tensor-only steady-state fast accumulator export and GPU runtime timing."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch import nn

from fsrmamba.fast import FastState


class FastStep(nn.Module):
    """Network and blend with prewarped history; first frames use the native model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hist, lr, mv_lr, depth, jitter, feat_prev, prev_depth=None):
        state = FastState(hist, feat_prev, depth if prev_depth is None else prev_depth,
                          frame_index=1)
        rgb, state = self.model(state, lr[0].permute(1, 2, 0), mv_lr[0], depth[0, 0],
                                jitter, hist=hist)
        return rgb.permute(2, 0, 1).unsqueeze(0), state.feat


def export_inputs(model, inputs, device):
    lr, mv, depth = inputs
    state = model.init_state(device)
    state.frame_index = 1
    hist = model.reproject(state, torch.nn.functional.interpolate(
        mv.permute(2, 0, 1)[None], size=model.output_size, mode="bilinear",
        align_corners=False).permute(0, 2, 3, 1))
    tensors = (hist, lr.permute(2, 0, 1)[None].contiguous(), mv[None].contiguous(),
               depth[None, None], torch.tensor([0.25, -0.25], device=device), state.feat)
    names = ["hist", "lr", "mv_lr", "depth", "jitter", "feat_prev"]
    if model.depth_test:
        tensors += (state.depth,)
        names.append("prev_depth")
    return tensors, names


def export_fast(model, inputs, path, device, fp16=False):
    try:
        import onnx
    except ImportError:
        print("ONNX unavailable; export skipped.")
        return None
    try:
        tensors, names = export_inputs(model, inputs, device)
        wrapper = FastStep(model).eval()
        context = (torch.autocast(device.type, dtype=torch.float16) if fp16
                   else nullcontext())
        with torch.no_grad(), context:
            torch.onnx.export(wrapper, tensors, path, input_names=names,
                              output_names=["out", "feat_new"], opset_version=18,
                              dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        print(f"ONNX exported to {path} (network + blend, steady state)")
        return tensors, names
    except Exception as exc:
        print(f"ONNX export failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return None


def feature_buffer(tensor):
    """Retain a valid buffer pointer for zero-channel state in I/O binding."""
    if tensor.numel():
        return tensor.contiguous()
    return torch.empty(1, dtype=tensor.dtype, device=tensor.device)[:0].reshape(tensor.shape)


def time_ort(path, exported, model, device, frames, warmup):
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime unavailable; timing skipped.")
        return
    if exported is None:
        print("ONNX Runtime timing skipped: no exported graph.")
        return
    available = ort.get_available_providers()
    if device.type != "cuda" or "CUDAExecutionProvider" not in available:
        print("ONNX Runtime GPU timing unavailable: CUDA device/provider required.")
        return
    import numpy as np
    from bench_latency import clock_read, report_times

    tensors, names = exported
    dev_id = device.index if device.index is not None else torch.cuda.current_device()
    stream = str(torch.cuda.current_stream(device).cuda_stream)
    cuda = ("CUDAExecutionProvider", {"device_id": dev_id, "user_compute_stream": stream})
    choices = [[cuda]]
    if "TensorrtExecutionProvider" in available:
        choices.append([("TensorrtExecutionProvider", {"device_id": dev_id,
                        "trt_fp16_enable": True, "user_compute_stream": stream}), cuda])

    dtypes = {"tensor(float)": (torch.float32, np.float32),
              "tensor(float16)": (torch.float16, np.float16)}

    def bindings(session):
        sources = dict(zip(names, tensors))
        output_types = {out.name: dtypes[out.type] for out in session.get_outputs()}
        feat_dtype = output_types["feat_new"][0]
        features = [sources["feat_prev"].to(feat_dtype).contiguous(),
                    torch.empty_like(sources["feat_prev"], dtype=feat_dtype)]
        features = [feature_buffer(tensor) for tensor in features]
        outputs = [torch.empty((1, 3, *model.output_size), device=device,
                               dtype=output_types["out"][0]) for _ in range(2)]
        # Keep allocated tensors alive while their raw pointers are bound.
        bindings = []
        for i in range(2):
            binding = session.io_binding()
            for inp in session.get_inputs():
                tensor = features[i] if inp.name == "feat_prev" else sources[inp.name]
                dtype, element_type = dtypes[inp.type]
                if tensor.dtype != dtype or not tensor.is_contiguous():
                    raise ValueError(f"Input {inp.name} requires contiguous {dtype} tensors")
                binding.bind_input(inp.name, "cuda", dev_id, element_type,
                                   tuple(tensor.shape), tensor.data_ptr()
                                   or tensor.untyped_storage().data_ptr())
            for name, tensor in (("out", outputs[i]), ("feat_new", features[1 - i])):
                binding.bind_output(name, "cuda", dev_id, output_types[name][1],
                                    tuple(tensor.shape), tensor.data_ptr()
                                   or tensor.untyped_storage().data_ptr())
            bindings.append(binding)
        return bindings, features, outputs

    for providers in choices:
        requested = providers[0][0]
        profile_path = None
        try:
            session = ort.InferenceSession(str(path), providers=providers)
            session.disable_fallback()
            bound, features, outputs = bindings(session)
            times = []
            for i in range(warmup + frames):
                start = clock_read(device)
                session.run_with_iobinding(bound[i % 2])
                elapsed = (clock_read(device) - start) * 1000
                if i >= warmup:
                    times.append(elapsed)
            # Verify node placement in a separate profiling pass.
            options = ort.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(path).with_suffix("")) + ".ort-profile"
            profile = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            profile.disable_fallback()
            prof_bound, prof_features, prof_outputs = bindings(profile)
            profile.run_with_iobinding(prof_bound[0])
            clock_read(device)
            profile_path = Path(profile.end_profiling())
            events = json.loads(profile_path.read_text())
            ran = sorted({e.get("args", {}).get("provider") for e in events
                          if e.get("args", {}).get("provider")})
            actual = ", ".join(ran) if ran else "unknown (no profiled kernels)"
            report_times(f"ORT {requested} (actual: {actual})", times)
        except Exception as exc:
            print(f"ORT {requested} failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        finally:
            if profile_path is not None:
                profile_path.unlink(missing_ok=True)
