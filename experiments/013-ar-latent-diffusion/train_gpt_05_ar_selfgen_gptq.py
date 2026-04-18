from __future__ import annotations

import copy
import io
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor

from train_gpt_02_loader_prefetch import (
    CONTROL_TENSOR_NAME_PATTERNS,
    CastedLinear,
    GPT,
    Hyperparameters as BaseHyperparameters,
    Muon,
    ShuffledSequenceLoader,
    StepCommProfiler,
    _HessianGPT,
    _compress_quant_payload,
    _decompress_quant_payload,
    _rebank_state_dict,
    _unbank_state_dict,
    build_sentencepiece_luts,
    dequantize_mixed_int6,
    eval_val,
    eval_val_sliding,
    load_validation_tokens,
    mixed_quantize_int6,
    restore_low_dim_params_to_fp32,
)


class Hyperparameters(BaseHyperparameters):
    diffusion_stop_frac = float(os.environ.get("DIFFUSION_STOP_FRAC", 0.40))
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 64))
    selfgen_prompt_len = int(os.environ.get("SELFGEN_PROMPT_LEN", os.environ.get("GPTQ_CALIB_PROMPT_LEN", "32")))
    selfgen_seq_len = int(os.environ.get("SELFGEN_SEQ_LEN", "256"))
    selfgen_batch_size = int(os.environ.get("SELFGEN_BATCH_SIZE", "8"))
    selfgen_temperature = float(os.environ.get("SELFGEN_TEMPERATURE", "0.8"))
    diffusion_scalar_wd = float(os.environ.get("DIFFUSION_SCALAR_WD", "0.08"))
    diffusion_head_wd = float(os.environ.get("DIFFUSION_HEAD_WD", "0.08"))


def _effective_selfgen_lengths(args: Hyperparameters) -> tuple[int, int]:
    seq_len = max(2, min(args.selfgen_seq_len, args.train_seq_len))
    prompt_len = max(1, min(args.selfgen_prompt_len, seq_len - 1, args.train_seq_len - 1))
    return prompt_len, seq_len


def generate_ar_selfgen_calibration_batches(
    model: GPT,
    train_loader: ShuffledSequenceLoader,
    args: Hyperparameters,
    device: torch.device,
    grad_accum_steps: int,
    num_batches: int,
) -> list[Tensor]:
    prompt_len, seq_len = _effective_selfgen_lengths(args)
    batch_size = max(1, args.selfgen_batch_size)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 17)
    calib_batches: list[Tensor] = []
    was_training = model.training
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for _ in range(num_batches):
            x, _ = train_loader.next_batch(args.train_batch_tokens, grad_accum_steps)
            if x.size(0) < batch_size:
                raise ValueError(
                    f"SELFGEN_BATCH_SIZE={batch_size} exceeds available local batch size {x.size(0)}"
                )
            tokens = x[:batch_size, :prompt_len].clone()
            while tokens.size(1) < seq_len:
                logits = model.forward_logits(tokens)
                next_logits = logits[:, -1, :]
                if args.selfgen_temperature > 0.0:
                    probs = torch.softmax(next_logits.float() / args.selfgen_temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1, generator=rng)
                else:
                    next_tok = next_logits.argmax(dim=-1, keepdim=True)
                tokens = torch.cat((tokens, next_tok), dim=1)
            calib_batches.append(tokens.detach().cpu())
    if was_training:
        model.train()
    return calib_batches


def collect_hessians_from_token_batches(
    hessian_model: _HessianGPT,
    token_batches: list[Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device="cpu")

            def make_hook(pname: str):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()

                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(param_name)))
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for tokens in token_batches:
            x = tokens[:, :-1].to(device=device, dtype=torch.int64, non_blocking=True)
            y = tokens[:, 1:].to(device=device, dtype=torch.int64, non_blocking=True)
            hessian_model(x, y)
    for hook in hooks:
        hook.remove()
    num_batches = max(len(token_batches), 1)
    for name, hessian in hessians.items():
        hessian /= num_batches
        damp = 0.01 * torch.diag(hessian).mean().clamp_min(1e-6)
        hessian += damp * torch.eye(hessian.shape[0])
        hessians[name] = hessian
    hessian_model.train()
    return hessians


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    if args.weight_share:
        grad_accum_steps *= 2
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        train_seq_len=args.train_seq_len,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        skip_gates_enabled=args.skip_gates_enabled,
        diffusion_loss_weight=args.diffusion_loss_weight,
        diffusion_aux_prob=args.diffusion_aux_prob,
        diffusion_subsample_frac=args.diffusion_subsample_frac,
        parallel_start_layer=args.parallel_start_layer,
        recurrence_depth=args.recurrence_depth,
        recurrence_start=args.recurrence_start,
        recurrence_end=args.recurrence_end,
        weight_share=args.weight_share,
        swigelu=args.swigelu,
        swigelu_hidden_dim=args.swigelu_hidden_dim,
    ).to(device).bfloat16()
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    matrix_params = [
        base_model.qo_bank,
        base_model.kv_bank,
        base_model.mlp_up_bank,
        base_model.mlp_down_bank,
    ]
    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
        scalar_params.append(base_model.skip_gates)
    diffusion_scalar_params = [base_model.mask_embed, base_model.time_proj.weight]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    diffusion_head_params: list[Tensor] = []
    if base_model.latent_proj is not None:
        diffusion_head_params.append(base_model.latent_proj.weight)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.embed_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
        row_normalize=args.muon_row_normalize,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": args.adam_wd}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
        fused=True,
    )
    optimizer_diffusion = torch.optim.AdamW(
        [
            {
                "params": diffusion_scalar_params,
                "lr": args.scalar_lr,
                "base_lr": args.scalar_lr,
                "weight_decay": args.diffusion_scalar_wd,
            },
            {
                "params": diffusion_head_params,
                "lr": args.head_lr,
                "base_lr": args.head_lr,
                "weight_decay": args.diffusion_head_wd,
            },
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
        fused=True,
    )
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)
    replicated_params.extend(diffusion_scalar_params)
    replicated_params.extend(diffusion_head_params)

    optimizer_head = None
    head_params: list[Tensor] = []
    if base_model.lm_head is not None:
        head_params.append(base_model.lm_head.weight)
    if head_params:
        optimizer_head = torch.optim.Adam(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.extend(head_params)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_diffusion]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)
    comm_profiler = StepCommProfiler(
        enabled=args.comm_profile,
        log_every=args.comm_profile_log_every,
        reducer_name="per_tensor",
        replicated_params=replicated_params,
        bank_params=matrix_params,
    )
    n_params = sum(p.numel() for p in base_model.parameters())
    diffusion_params = (
        base_model.mask_embed.numel()
        + sum(p.numel() for p in base_model.time_proj.parameters())
        + (base_model.latent_proj.weight.numel() if base_model.latent_proj is not None else 0)
    )
    prompt_len, selfgen_seq_len = _effective_selfgen_lengths(args)
    log0(f"model_params:{n_params}")
    log0(
        f"diffusion_aux:prob:{args.diffusion_aux_prob:.2f} "
        f"weight:{args.diffusion_loss_weight:.4f} start_frac:{args.diffusion_start_frac:.2f} stop_frac:{args.diffusion_stop_frac:.2f} "
        f"subsample:{args.diffusion_subsample_frac:.2f} objective:latent_mse params:{diffusion_params}"
    )
    log0(
        f"diffusion_decay:scalar_wd:{args.diffusion_scalar_wd:.4f} "
        f"head_wd:{args.diffusion_head_wd:.4f}"
    )
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if (head_params or diffusion_head_params) else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"quant_stack:compressor={args.compressor} matrix_bits={args.matrix_bits} embed_bits={args.embed_bits} "
        f"gptq_scale=clip_sigmas matrix_clip_sigmas={args.matrix_clip_sigmas} embed_clip_sigmas={args.embed_clip_sigmas} "
        f"calib_mode=ar_selfgen calib_batches={args.gptq_calib_batches} "
        f"selfgen_prompt_len={prompt_len} selfgen_seq_len={selfgen_seq_len} "
        f"selfgen_batch_size={args.selfgen_batch_size} selfgen_temp={args.selfgen_temperature:.2f} "
        f"muon_row_normalize={args.muon_row_normalize}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    comm_profiler.log_static(log0)
    if args.weight_share:
        log0(f"weight_share:enabled (2x effective depth = {args.num_layers * 2} layers from {args.num_layers} physical)")
    if args.recurrence_depth > 1:
        log0(
            f"recurrence:depth={args.recurrence_depth} "
            f"layers=[{args.recurrence_start},{args.recurrence_end}] start_frac={args.recurrence_start_frac}"
        )
    train_loader = ShuffledSequenceLoader(args, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    effective_gptq_reserve_seconds = 0.0 if args.skip_quant else args.gptq_reserve_seconds
    if max_wallclock_ms is not None and effective_gptq_reserve_seconds > 0:
        max_wallclock_ms = max(max_wallclock_ms - 1000.0 * effective_gptq_reserve_seconds, 0.0)
        log0(f"gptq:reserving {effective_gptq_reserve_seconds:.0f}s for quant/export effective_wallclock_ms:{max_wallclock_ms:.0f}")
    elif max_wallclock_ms is not None and args.skip_quant and args.gptq_reserve_seconds > 0:
        log0(f"gptq:reserve_skipped skip_quant:enabled effective_wallclock_ms:{max_wallclock_ms:.0f}")

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0 and args.warmdown_frac <= 0:
            return 1.0
        if max_wallclock_ms is None:
            if args.warmdown_iters > 0:
                warmdown_start = max(args.iterations - args.warmdown_iters, 0)
                if warmdown_start <= step < args.iterations:
                    return max((args.iterations - step) / max(args.warmdown_iters, 1), args.min_lr)
                return 1.0
            frac = step / max(args.iterations, 1)
            if frac >= 1.0 - args.warmdown_frac:
                return max((1.0 - frac) / max(args.warmdown_frac, 1e-9), args.min_lr)
            return 1.0
        frac = elapsed_ms / max(max_wallclock_ms, 1e-9)
        if frac >= 1.0 - args.warmdown_frac:
            return max((1.0 - frac) / max(args.warmdown_frac, 1e-9), args.min_lr)
        return 1.0

    effective_warmup_steps = args.warmup_steps
    has_diffusion = args.diffusion_loss_weight > 0.0 and args.diffusion_aux_prob > 0.0
    if args.skip_quant and args.screen_warmup_steps >= 0:
        min_compile_paths = 2 if has_diffusion else 1
        effective_warmup_steps = min(args.warmup_steps, max(args.screen_warmup_steps, min_compile_paths))
        if effective_warmup_steps != args.warmup_steps:
            log0(f"screen_mode:warmup_steps {args.warmup_steps}->{effective_warmup_steps}")
    if effective_warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(effective_warmup_steps):
            zero_grad_all()
            for _micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    if has_diffusion and warmup_step == 0:
                        warmup_use_diffusion = True
                    elif has_diffusion and warmup_step == 1:
                        warmup_use_diffusion = False
                    else:
                        warmup_use_diffusion = has_diffusion and random.random() < args.diffusion_aux_prob
                    warmup_loss = model(x, y, warmup_use_diffusion)
                (warmup_loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if effective_warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == effective_warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{effective_warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = ShuffledSequenceLoader(args, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    training_time_ms = 0.0
    stop_after_step: int | None = None
    diffusion_cutoff_logged = False
    diffusion_startup_logged = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            t_eval = time.perf_counter()
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        step_wall_t0 = time.perf_counter()
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for _micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                before_start = (
                    args.diffusion_start_frac > 0.0
                    and max_wallclock_ms is not None
                    and elapsed_ms < args.diffusion_start_frac * max_wallclock_ms
                )
                after_stop = not (
                    args.diffusion_stop_frac >= 1.0
                    or (max_wallclock_ms is not None and elapsed_ms < args.diffusion_stop_frac * max_wallclock_ms)
                )
                diffusion_active = not before_start and not after_stop
                if not before_start and not diffusion_startup_logged and args.diffusion_start_frac > 0.0:
                    log0(f"diffusion_startup:step:{step} elapsed_ms:{elapsed_ms:.0f} frac:{args.diffusion_start_frac:.2f}")
                    diffusion_startup_logged = True
                if after_stop and not diffusion_cutoff_logged:
                    log0(f"diffusion_cutoff:step:{step} elapsed_ms:{elapsed_ms:.0f} frac:{args.diffusion_stop_frac:.2f}")
                    diffusion_cutoff_logged = True
                use_diffusion = (
                    diffusion_active
                    and args.diffusion_loss_weight > 0.0
                    and args.diffusion_aux_prob > 0.0
                    and random.random() < args.diffusion_aux_prob
                )
                loss = model(x, y, use_diffusion)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer_muon.launch_reduce_scatters()
        replicated_reduce_ms = 0.0
        replicated_reduce_calls = 0
        replicated_reduce_bytes = 0
        if distributed:
            t_replicated_reduce = time.perf_counter()
            for p in replicated_params:
                if p.grad is not None:
                    replicated_reduce_calls += 1
                    replicated_reduce_bytes += p.grad.numel() * p.grad.element_size()
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            replicated_reduce_ms = 1000.0 * (time.perf_counter() - t_replicated_reduce)
        optimizer_tok.step()
        optimizer_scalar.step()
        optimizer_diffusion.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        comm_profiler.record_replicated(replicated_reduce_ms, replicated_reduce_calls, replicated_reduce_bytes)
        comm_profiler.record_muon(optimizer_muon.drain_comm_stats())
        zero_grad_all()
        step += 1
        comm_profiler.record_step(1000.0 * (time.perf_counter() - step_wall_t0))
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
        comm_profiler.maybe_log(step, log0, force=stop_after_step is not None)
    comm_profiler.maybe_log(step, log0, force=True)
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args,
        compiled_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_train val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "latent_proj" not in k}
    excluded_aux = sum(int(t.numel()) for k, t in full_state_dict.items() if "latent_proj" in k)
    if excluded_aux > 0:
        log0(f"export_excluding_training_only_params:{excluded_aux}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    if args.skip_quant:
        log0("skip_quant:enabled — skipping post-training quantization pipeline")
        if distributed:
            dist.destroy_process_group()
        return
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    log0("gptq:building non-banked model for Hessian collection...")
    hessian_model = _HessianGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        train_seq_len=args.train_seq_len,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        skip_gates_enabled=args.skip_gates_enabled,
        parallel_start_layer=args.parallel_start_layer,
        recurrence_depth=args.recurrence_depth,
        recurrence_start=args.recurrence_start,
        recurrence_end=args.recurrence_end,
        weight_share=args.weight_share,
        swigelu=args.swigelu,
        swigelu_hidden_dim=args.swigelu_hidden_dim,
    ).to(device).bfloat16()
    for module in hessian_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(hessian_model)
    hessian_model.load_state_dict(
        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
        strict=False,
    )
    calib_loader = ShuffledSequenceLoader(args, rank, world_size, device)
    log0(
        f"gptq:generating {args.gptq_calib_batches} AR self-gen calibration batches "
        f"(prompt_len={prompt_len} seq_len={selfgen_seq_len} batch_size={args.selfgen_batch_size} "
        f"temp={args.selfgen_temperature:.2f})..."
    )
    t_selfgen = time.perf_counter()
    calib_batches = generate_ar_selfgen_calibration_batches(
        base_model,
        calib_loader,
        args,
        device,
        grad_accum_steps,
        num_batches=args.gptq_calib_batches,
    )
    log0(f"gptq:generated {len(calib_batches)} AR calibration batches in {time.perf_counter() - t_selfgen:.1f}s")
    _t_hessian_start = time.perf_counter()
    hessians = collect_hessians_from_token_batches(hessian_model, calib_batches, device)
    _t_hessian_elapsed = time.perf_counter() - _t_hessian_start
    log0(f"gptq:collected hessians for {len(hessians)} layers in {_t_hessian_elapsed:.1f}s")
    del calib_batches
    del hessian_model
    torch.cuda.empty_cache()
    _t_quant_start = time.perf_counter()
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, args, hessians=hessians)
    _t_quant_elapsed = time.perf_counter() - _t_quant_start
    log0(f"gptq:quantized {len(quant_meta)} layers in {_t_quant_elapsed:.1f}s")
    target_mb = float(os.environ.get("TARGET_MB", "15.999"))
    code_bytes_est = len(code.encode("utf-8"))
    prune_keys = []
    prune_idx = []
    all_errors = []
    for name, info in quant_meta.items():
        if not (isinstance(info, str) and info == "gptq (int6)"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = q.abs() == 1
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                order = errors.argsort()
                prune_keys.append(qk)
                prune_idx.append(flat_idx[order])
                all_errors.append(errors[order])
    total_candidates = sum(len(e) for e in all_errors)
    if total_candidates > 0:
        global_errors = torch.cat(all_errors)
        global_order = global_errors.argsort()
        offsets = []
        off = 0
        for errors in all_errors:
            offsets.append(off)
            off += len(errors)
        group_ids = torch.zeros(total_candidates, dtype=torch.long)
        for gi, errors in enumerate(all_errors):
            group_ids[offsets[gi]:offsets[gi] + len(errors)] = gi
        sorted_group_ids = group_ids[global_order]
        sorted_local_pos = torch.zeros(total_candidates, dtype=torch.long)
        for gi in range(len(prune_keys)):
            mask = sorted_group_ids == gi
            sorted_local_pos[mask] = torch.arange(mask.sum())

        def _apply_prune(n: int) -> dict[str, Tensor]:
            tmp = {k: v.clone() for k, v in quant_result.items()}
            if n <= 0:
                return tmp
            n = min(n, total_candidates)
            sel_groups = sorted_group_ids[:n]
            sel_local = sorted_local_pos[:n]
            for gi in range(len(prune_keys)):
                mask = sel_groups == gi
                if not mask.any():
                    continue
                local_positions = sel_local[mask]
                flat_indices = prune_idx[gi][local_positions]
                tmp[prune_keys[gi]].view(-1)[flat_indices] = 0
            return tmp

        def _measure(tmp: dict[str, Tensor]) -> int:
            buf = io.BytesIO()
            torch.save({"w": tmp, "m": quant_meta}, buf)
            return len(_compress_quant_payload(buf.getvalue(), args.compressor)) + code_bytes_est

        no_sz = _measure(_apply_prune(0))
        target_bytes = int(target_mb * 1_000_000)
        log0(f"selective_prune: {total_candidates} ±1 candidates, unpruned={no_sz / 1e6:.2f}MB target={target_mb}MB")
        if no_sz <= target_bytes:
            log0("selective_prune: already fits, no pruning needed")
        else:
            full_sz = _measure(_apply_prune(total_candidates))
            log0(f"selective_prune: full ±1 prune={full_sz / 1e6:.2f}MB")
            if full_sz > target_bytes:
                log0("selective_prune: even full prune not enough, applying all")
                quant_result = _apply_prune(total_candidates)
            else:
                lo, hi = 0, total_candidates
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz = _measure(_apply_prune(mid))
                    if sz <= target_bytes:
                        hi = mid
                    else:
                        lo = mid + 1
                log0(
                    f"selective_prune: pruning {lo}/{total_candidates} ±1 values "
                    f"({100 * lo / total_candidates:.1f}%) to fit {target_mb}MB"
                )
                quant_result = _apply_prune(lo)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress_quant_payload(quant_raw, args.compressor)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model mixed-quant+{args.compressor}: {quant_file_bytes} bytes")
        log0(f"Total submission size mixed-quant+{args.compressor}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress_quant_payload(quant_blob_disk, args.compressor)),
        map_location="cpu",
    )
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        train_seq_len=args.train_seq_len,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        skip_gates_enabled=args.skip_gates_enabled,
        diffusion_loss_weight=0.0,
        diffusion_aux_prob=0.0,
        parallel_start_layer=args.parallel_start_layer,
        recurrence_depth=args.recurrence_depth,
        recurrence_start=args.recurrence_start,
        recurrence_end=args.recurrence_end,
        weight_share=args.weight_share,
        swigelu=args.swigelu,
        swigelu_hidden_dim=args.swigelu_hidden_dim,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for module in eval_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_eval,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
