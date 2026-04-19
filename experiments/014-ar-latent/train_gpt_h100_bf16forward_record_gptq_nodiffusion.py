from __future__ import annotations

import os
import re
from collections import defaultdict

import torch
from torch import Tensor, nn

import train_gpt_h100_bf16forward as base


# Record-style GPTQ path, but with diffusion removed from the train-time model.


_DISABLED_DIFFUSION_KEYS = {"mask_embed", "time_proj.weight"}


class _NoDiffusionTimeProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0, 1, dtype=torch.float32))

    def forward(self, _x: Tensor) -> Tensor:
        raise RuntimeError("time_proj is disabled in the no-diffusion trainer")


class GPTNoDiffusion(base.GPT):
    def __init__(self, *args, **kwargs):
        kwargs["diffusion_loss_weight"] = 0.0
        kwargs["diffusion_aux_prob"] = 0.0
        super().__init__(*args, **kwargs)
        self.diffusion_loss_weight = 0.0
        self.diffusion_aux_prob = 0.0
        self.mask_embed = nn.Parameter(torch.empty(0, dtype=torch.float32))
        self.time_proj = _NoDiffusionTimeProj()
        self.latent_proj = None

    def _diffusion_loss(self, input_ids: Tensor, _target_ids: Tensor, subsample_frac: float = 1.00) -> Tensor:
        raise RuntimeError(
            f"diffusion is disabled for this trainer (input_shape={tuple(input_ids.shape)} subsample_frac={subsample_frac})"
        )

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for key in _DISABLED_DIFFUSION_KEYS:
            sd.pop(key, None)
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered = dict(state_dict)
        for key in _DISABLED_DIFFUSION_KEYS:
            filtered.pop(key, None)
        incompatible = super().load_state_dict(filtered, strict=False)
        if strict:
            missing = [k for k in incompatible.missing_keys if k not in _DISABLED_DIFFUSION_KEYS]
            unexpected = [k for k in incompatible.unexpected_keys if k not in _DISABLED_DIFFUSION_KEYS]
            if missing or unexpected:
                missing_str = ", ".join(missing) if missing else "[]"
                unexpected_str = ", ".join(unexpected) if unexpected else "[]"
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                    f"missing_keys={missing_str} unexpected_keys={unexpected_str}"
                )
        return incompatible


def _patch_record_quant_hparams() -> None:
    gptq_calibration_batches = int(
        os.environ.get("GPTQ_CALIBRATION_BATCHES", os.environ.get("GPTQ_CALIB_BATCHES", "64"))
    )
    base.Hyperparameters.gptq_calibration_batches = gptq_calibration_batches
    base.Hyperparameters.gptq_calib_batches = gptq_calibration_batches
    base.Hyperparameters.gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", "12.0"))


def _patch_no_diffusion_hparams() -> None:
    base.Hyperparameters.diffusion_loss_weight = 0.0
    base.Hyperparameters.diffusion_aux_prob = 0.0
    base.Hyperparameters.diffusion_start_frac = 1.0
    base.Hyperparameters.diffusion_stop_frac = 0.0
    base.Hyperparameters.diffusion_subsample_frac = 1.0
    base.Hyperparameters.diffusion_scalar_wd = 0.0
    base.Hyperparameters.diffusion_head_wd = 0.0


def generate_train_calibration_batches(
    _model: "base.GPT",
    train_loader: "base.ShuffledSequenceLoader",
    args: "base.Hyperparameters",
    _device: torch.device,
    grad_accum_steps: int,
    num_batches: int,
) -> list[Tensor]:
    batches: list[Tensor] = []
    for _ in range(num_batches):
        x, y = train_loader.next_batch(args.train_batch_tokens, grad_accum_steps)
        batches.append(torch.cat((x, y[:, -1:]), dim=1).detach().cpu())
    return batches


def collect_hessians_from_token_batches(
    hessian_model: "base._HessianGPT",
    token_batches: list[Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_input_hook(param_name: str):
        def hook_fn(_module, input, _output):
            x = input[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            hessians[param_name] += (x.T @ x).cpu()

        return hook_fn

    for name, module in hessian_model.named_modules():
        if not isinstance(module, base.CastedLinear) or module.weight.numel() <= 65536:
            continue
        param_name = f"{name}.weight"
        category = base._classify_param(param_name)
        if category not in ("mlp", "attn") and param_name != "lm_head.weight":
            continue
        cols = module.weight.shape[1]
        hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device="cpu")
        hooks.append(module.register_forward_hook(make_input_hook(param_name)))

    if hessian_model.tie_embeddings:
        hessians["tok_emb.weight"] = torch.zeros(
            hessian_model.tok_emb.weight.shape[1],
            hessian_model.tok_emb.weight.shape[1],
            dtype=torch.float32,
            device="cpu",
        )

        def output_hook(_module, _input, output):
            x = output.detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            hessians["tok_emb.weight"] += (x.T @ x).cpu()

        hooks.append(hessian_model.final_norm.register_forward_hook(output_hook))

    was_training = hessian_model.training
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for tokens in token_batches:
            x = tokens[:, :-1].to(device=device, dtype=torch.int64, non_blocking=True)
            y = tokens[:, 1:].to(device=device, dtype=torch.int64, non_blocking=True)
            hessian_model(x, y)

    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] /= max(len(token_batches), 1)
    if was_training:
        hessian_model.train()
    return hessians


def _log_quantized_weights(meta: dict[str, object]) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    categories: defaultdict[str, set[str]] = defaultdict(set)
    for name, cat in meta.items():
        short = re.sub(r"\.\d+$", "", re.sub(r"blocks\.\d+", "blocks", name))
        categories[str(cat)].add(short)
    print("Quantized weights:")
    for cat in sorted(categories):
        print(f"  {cat}: {', '.join(sorted(categories[cat]))}")


def mixed_quantize_int6(
    state_dict: dict[str, Tensor],
    args: "base.Hyperparameters",
    hessians: dict[str, Tensor],
):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        if t.ndim != 2 or name not in hessians:
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough (float16)"
            continue
        clip_sigmas = args.embed_clip_sigmas if "tok_emb" in name else args.matrix_clip_sigmas
        bits = args.embed_bits if "tok_emb" in name else args.matrix_bits
        q, s = base.gptq_quantize_weight(
            t,
            hessians[name],
            clip_sigmas=clip_sigmas,
            clip_range=2 ** (bits - 1) - 1,
            block_size=128,
        )
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    _log_quantized_weights(meta)
    return result, meta


def main() -> None:
    _patch_record_quant_hparams()
    _patch_no_diffusion_hparams()
    # The source record path does not apply the later selective +/-1 pruning step.
    os.environ["TARGET_MB"] = "999.0"
    base.GPT = GPTNoDiffusion
    base.generate_ar_selfgen_calibration_batches = generate_train_calibration_batches
    base.collect_hessians_from_token_batches = collect_hessians_from_token_batches
    base.mixed_quantize_int6 = mixed_quantize_int6
    base.main()


if __name__ == "__main__":
    main()
