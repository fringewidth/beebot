"""Device plumbing so train-vla also runs on a TPU through torch_xla (--device xla).

On XLA the model trains on the TPU core, while the data (scripted or play store), batch sampling
and the closed-loop evaluation stay on the CPU: they are full of data-dependent shapes and
control flow that XLA would recompile endlessly. Batches are moved to the TPU once per step, and
each evaluation runs on a CPU copy of the model. On any other device every helper is a no-op, so
train.py reads the same on CUDA, MPS and CPU.

The xla_* helpers below serve voxel/think_train.py (one process per chip, data-parallel); torch_xla
is imported only inside them.
"""

import torch


def is_xla(name):
    return str(name).startswith("xla")


def devices(name):
    """(train device, data device): the TPU core and the CPU on XLA, else the same device twice."""
    if is_xla(name):
        import torch_xla.core.xla_model as xm

        return xm.xla_device(), torch.device("cpu")
    device = torch.device(name)
    return device, device


def to_device(device, *tensors):
    return tuple(t.to(device) for t in tensors)


def step_done(device):
    """End of one optimizer step: cut and run the lazy XLA graph (no-op elsewhere)."""
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()


def cpu_view(policy, build, device):
    """The model to evaluate and save: on XLA a fresh CPU copy with the current weights (built
    by build()), else the training model itself."""
    if device.type != "xla":
        return policy
    copy = build().to("cpu")
    copy.load_state_dict({k: v.detach().to("cpu") for k, v in policy.state_dict().items()})
    return copy


# ---- one process per chip (voxel/think_train.py)


def xla_setup(index=0):
    """(rank, world, device) of this process: its TPU chip, its ordinal among the processes. With
    BEEBOT_XLA_CACHE set, compiled graphs persist under it (one directory per process index and
    process count), so a later run of the same shapes skips compiling."""
    import os

    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    cache = os.environ.get("BEEBOT_XLA_CACHE")
    if cache:
        try:
            xr.initialize_cache(os.path.join(cache, f"{os.environ.get('BEEBOT_XLA_PROCESSES', '1')}-{index}"), readonly=False)
        except Exception as error:  # an older torch_xla, or the runtime already started
            print(f"no persistent XLA cache: {error}", flush=True)
    return xr.global_ordinal(), xr.world_size(), xm.xla_device()


def xla_launch(fn, args):
    """fn(index, *args) in one process per TPU chip."""
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(fn, args=args)


def xla_average_gradients(params, world):
    """Mean gradient over the processes, inside the step's graph (no host sync)."""
    import torch_xla.core.xla_model as xm

    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / world)


def xla_agree(flags, device):
    """Boolean flags true on any process, on every process (one host sync)."""
    import torch_xla.core.xla_model as xm

    t = torch.tensor([float(f) for f in flags], device=device)
    t = xm.all_reduce(xm.REDUCE_MAX, t)
    return [bool(f) for f in t.cpu().tolist()]


def xla_barrier(tag):
    import torch_xla.core.xla_model as xm

    xm.rendezvous(tag)


def xla_mark_step():
    import torch_xla.core.xla_model as xm

    xm.mark_step()


def xla_memory_gb(device):
    """HBM in use (peak when torch_xla reports it) on this chip, in GB."""
    import torch_xla.core.xla_model as xm

    info = xm.get_memory_info(device)
    used = info.get("peak_bytes_used", info.get("bytes_used"))
    if used is None and "kb_total" in info:
        used = (info["kb_total"] - info["kb_free"]) * 1024
    return round((used or 0) / 1e9, 2)


def xla_report():
    """Graph compilations so far, their total seconds, and aten ops that fell back to the CPU."""
    import torch_xla.debug.metrics as met

    data = met.metric_data("CompileTime")
    fallbacks = met.executed_fallback_ops() if hasattr(met, "executed_fallback_ops") else [n for n in met.counter_names() if n.startswith("aten::")]
    return {"compiles": data[0] if data else 0, "compile_s": round(data[1] / 1e9, 1) if data else 0.0, "cpu_fallbacks": sorted(fallbacks)}


def xla_checkpoint_function():
    """torch_xla's activation checkpointing (it keeps XLA from merging the recomputation back
    into the forward pass), reentrant; torch's when torch_xla lacks it."""
    try:
        from torch_xla.utils.checkpoint import checkpoint
    except ImportError:
        from torch.utils.checkpoint import checkpoint
    return checkpoint


def to_cpu(value):
    """Tensors (in dicts, lists and tuples, as in an optimizer state) moved to the CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value
