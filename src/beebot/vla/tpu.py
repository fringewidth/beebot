"""Device plumbing so train-vla also runs on a TPU through torch_xla (--device xla).

On XLA the model trains on the TPU core, while the data (scripted or play store), batch sampling
and the closed-loop evaluation stay on the CPU: they are full of data-dependent shapes and
control flow that XLA would recompile endlessly. Batches are moved to the TPU once per step, and
each evaluation runs on a CPU copy of the model. On any other device every helper is a no-op, so
train.py reads the same on CUDA, MPS and CPU.
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
