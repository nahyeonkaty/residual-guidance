"""Residual guidance (RG) for prompt switches in chunk-wise autoregressive video diffusion. Inference only, no training.

When the prompt changes at chunk k, the KV cache built under the old prompt keeps the new chunks in the old scene. For every chunk j
of the new prompt's interval (from k up to the next switch) we take the previous chunk's clean output s_{j-1}, re-noise it to t*, and predict it once more under
the current prompt with the cached context hidden (only the chunk's own tokens are attended):

    z   = add_noise(s_{j-1}, eps_k, t*)
    r_j = x0(z | current prompt, no context) - s_{j-1}

r_j is added to the x0 prediction of chunk j at denoising step 1 (x 1.0) and step 2 (x 0.5). Defaults (paper): t* = 850,
step weights (1.0, 0.5), one eps per switch shared by the chunks of its interval. Cost: one extra forward per
guided chunk. Relative RoPE inside a chunk does not depend on its absolute position, so the restore pass runs at chunk j's position.

Works with the Wan2.1-based chunk-wise repos that share `wan/modules/causal_model.py` (Self Forcing, Causal Forcing, LongLive, Infinity-RoPE).
"""
import contextlib
from dataclasses import dataclass

import torch


@dataclass
class RGConfig:
    t: int = 850                          # nominal re-noising timestep t* (warped like the denoising steps)
    step_weights: tuple = (1.0, 0.5)      # weight of r at denoising steps 1, 2, ...
    eps_seed: int = 0                     # every chunk from a switch to the next one is guided; eps of the switch at chunk k: CPU generator seeded eps_seed + k


_HIDE = [False]


def patch_attention(causal_model_module):
    """Wrap the self-attention of `wan.modules.causal_model`: inside `context_hidden()` only the current chunk's keys
    (the last q.shape[1]) are attended."""
    if getattr(causal_model_module, "_rg_patched", False):
        return
    orig = causal_model_module.attention

    def attention(q, k, v, *a, **kw):
        n = q.shape[1]
        if _HIDE[0] and k.shape[1] > n:
            k, v = k[:, -n:], v[:, -n:]
        return orig(q, k, v, *a, **kw)
    causal_model_module.attention = attention
    causal_model_module._rg_patched = True


@contextlib.contextmanager
def context_hidden():
    _HIDE[0] = True
    try:
        yield
    finally:
        _HIDE[0] = False


def switch_chunks(chunk_prompt):
    return [b for b in range(1, len(chunk_prompt)) if chunk_prompt[b] != chunk_prompt[b - 1]]


def reset_crossattn(pipe):
    for blk in pipe.crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False


@torch.no_grad()
def rollout(pipe, noise, conds, chunk_prompt, rg=None, init_caches=None, on_switch=None, on_chunk=None):
    """Chunk-wise 4-step rollout of `pipe` (an upstream CausalInferencePipeline / InteractiveCausalInferencePipeline) with a
    prompt per chunk. conds: {name: conditional_dict}; chunk_prompt: one name per chunk. At a switch `on_switch(pipe, output,
    start_frame, cond)` runs (default: only the cross-attention cache is rebuilt from the new prompt; LongLive: its KV recache).
    rg: RGConfig or None (plain rollout). The step noise comes from the global RNG (torch.randn_like), as upstream.
    on_chunk(b): called at the start of every chunk, after the switch handling (Infinity-RoPE's scene-cut flag, timing in speed.py)."""
    fpb = pipe.num_frame_per_block
    bsz, n_frames = noise.shape[:2]
    assert n_frames == fpb * len(chunk_prompt)
    init_caches(pipe)
    gen, sch = pipe.generator, pipe.generator.get_scheduler()
    tab = torch.cat((sch.timesteps.cpu(), torch.tensor([0.0])))
    switches = switch_chunks(chunk_prompt)
    output = torch.zeros_like(noise)
    prev = None                                           # s_{j-1}: the previous chunk's clean output
    for b, name in enumerate(chunk_prompt):
        f0 = b * fpb
        if b in switches:
            (on_switch or (lambda p, *_: reset_crossattn(p)))(pipe, output, f0, conds[name])
        if on_chunk is not None:
            on_chunk(b)
        cond = conds[name]
        cur = dict(kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=f0 * pipe.frame_seq_length)
        x = noise[:, f0:f0 + fpb]
        sw = max([s for s in switches if s <= b], default=None)
        res = None
        for i, t in enumerate(pipe.denoising_step_list):
            timestep = torch.ones([bsz, fpb], device=noise.device, dtype=torch.int64) * t
            if i == 0 and rg is not None and sw is not None:     # residual of this chunk (any chunk after a switch)
                tw = tab[1000 - rg.t].to(noise.device)
                eps = torch.randn(prev.flatten(0, 1).shape, generator=torch.Generator("cpu").manual_seed(rg.eps_seed + sw)).to(prev.device, prev.dtype)
                z = sch.add_noise(prev.flatten(0, 1), eps, tw * torch.ones([fpb], device=noise.device, dtype=torch.long)).unflatten(0, prev.shape[:2])
                with context_hidden():
                    _, xz = gen(noisy_image_or_video=z, conditional_dict=cond, timestep=torch.ones_like(timestep) * tw.to(timestep.dtype), **cur)
                res = xz.float() - prev.float()
            _, x0 = gen(noisy_image_or_video=x, conditional_dict=cond, timestep=timestep, **cur)
            if res is not None and i < len(rg.step_weights) and rg.step_weights[i]:
                x0 = (x0.float() + rg.step_weights[i] * res).to(x0.dtype)
            if i < len(pipe.denoising_step_list) - 1:
                nt = pipe.denoising_step_list[i + 1]
                x = sch.add_noise(x0.flatten(0, 1), torch.randn_like(x0.flatten(0, 1)),
                                  nt * torch.ones([bsz * fpb], device=noise.device, dtype=torch.long)).unflatten(0, x0.shape[:2])
        output[:, f0:f0 + fpb] = x0
        prev = x0.detach().clone()
        gen(noisy_image_or_video=x0, conditional_dict=cond, timestep=torch.ones_like(timestep) * pipe.args.context_noise, **cur)   # clean cache pass
    return output
