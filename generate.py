#!/usr/bin/env python
"""Generate prompt-switch videos with or without residual guidance.

  python generate.py --backbone longlive --repo /path/to/LongLive --ckpt_dir /path/to/longlive_models \
      --prompts prompts/ab_pairs.json --method rg --out out/ll_rg

--backbone   self_forcing | causal_forcing | longlive | infinity_rope. The upstream repo is used unmodified: it is put on sys.path and the
             working directory is set to --workdir (default --repo), which must hold `wan_models/Wan2.1-T2V-1.3B` as in the repo's README.
--prompts    {"items": [{"id", "seed", "prompts": [...], "switch_chunks": [...], "n_chunks"}]}: prompt q is used from
             switch_chunks[q - 1] on (3 latent frames per chunk, 16 fps).
--method     naive (the prompt just changes) | rg (residual guidance).  --recache: LongLive's KV recache at every switch
             (the LongLive setting of the paper, combined with naive or rg).  --scene_cut: Infinity-RoPE's RoPE cut at every switch
             (upstream "[..s#]"; its KV flush runs at every switch anyway).
Noise: global seed (set_seed(seed), initial noise on the GPU, step noise torch.randn_like as upstream); the RG eps of the switch at
chunk k from a CPU generator seeded 100000 + 100 seed + k.        -> <out>/<id:03d>.mp4 (+ .json)"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
device, DTYPE, FPB = torch.device("cuda"), torch.bfloat16, 3
LATENT = (16, 60, 104)
noise_frames = [0]


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["self_forcing", "causal_forcing", "longlive", "infinity_rope"])
    ap.add_argument("--repo", required=True, help="upstream repo root (LongLive: the folder with configs/longlive_interactive_inference.yaml)")
    ap.add_argument("--workdir", default=None, help="folder holding wan_models/ (default --repo)")
    ap.add_argument("--ckpt", default=None, help="Self / Causal Forcing / Infinity-RoPE generator checkpoint")
    ap.add_argument("--ckpt_dir", default=None, help="LongLive: folder with longlive_base.pt and lora.pt")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--scene_cut", action="store_true", help="Infinity-RoPE: RoPE scene cut at every switch")
    ap.add_argument("--t", type=int, default=850)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    return ap


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_forcing(args):
    """Self Forcing / Causal Forcing: upstream CausalInferencePipeline, EMA generator of the DMD checkpoint (Self Forcing) /
    generator (Causal Forcing chunk-wise)."""
    from pipeline import CausalInferencePipeline
    cfgs = {"self_forcing": ("configs/self_forcing_dmd.yaml", "generator_ema"), "causal_forcing": ("configs/causal_forcing_dmd_chunkwise.yaml", "generator")}
    cfg_path, key = cfgs[args.backbone]
    cfg = OmegaConf.merge(OmegaConf.load(f"{args.repo}/configs/default_config.yaml"), OmegaConf.load(f"{args.repo}/{cfg_path}"))
    pipe = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)[key]
    pipe.generator.load_state_dict({k.replace("_fsdp_wrapped_module.", ""): v for k, v in sd.items()})
    del sd
    pipe = pipe.to(dtype=DTYPE).to(device).eval()

    def init_caches(p):
        p.kv_cache1 = p.crossattn_cache = None
        n = noise_frames[0] * p.frame_seq_length
        p.kv_cache1 = [{"k": torch.zeros([1, max(32760, n), 12, 128], dtype=DTYPE, device=device),
                        "v": torch.zeros([1, max(32760, n), 12, 128], dtype=DTYPE, device=device),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device)} for _ in range(p.num_transformer_blocks)]
        p._initialize_crossattn_cache(1, dtype=DTYPE, device=device)
    return pipe, init_caches, None, None


def load_longlive(args):
    """LongLive-1.3B as upstream interactive_inference.py (configs/longlive_interactive_inference.yaml, base weights + LoRA, not
    merged); the T5 encoder stays on the CPU and is streamed to the GPU (upstream's DynamicSwapInstaller)."""
    import peft
    from pipeline import InteractiveCausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    from utils.memory import DynamicSwapInstaller
    from utils.wan_wrapper import WanTextEncoder
    cfg = OmegaConf.load(f"{args.repo}/configs/longlive_interactive_inference.yaml")
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False          # build the fp32 T5 on the CPU
    try:
        te = WanTextEncoder()
    finally:
        torch.cuda.is_available = real
    pipe = InteractiveCausalInferencePipeline(cfg, device=device, text_encoder=te)
    pipe.generator.load_state_dict(torch.load(f"{args.ckpt_dir}/longlive_base.pt", map_location="cpu", weights_only=False)["generator"])
    pipe.generator.model = configure_lora_for_model(pipe.generator.model, model_name="generator", lora_config=cfg.adapter, is_main_process=False)
    lora = torch.load(f"{args.ckpt_dir}/lora.pt", map_location="cpu", weights_only=False)
    peft.set_peft_model_state_dict(pipe.generator.model, lora.get("generator_lora", lora))
    pipe.is_lora_enabled = True
    pipe = pipe.to(dtype=DTYPE)
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=device)
    pipe.generator.to(device=device).eval()
    pipe.vae.to(device=device)

    def init_caches(p):
        p.kv_cache1 = p.crossattn_cache = None
        torch.cuda.empty_cache()
        la = getattr(p.args.model_kwargs, "local_attn_size", -1)
        size = la * p.frame_seq_length if la != -1 else noise_frames[0] * p.frame_seq_length
        p._initialize_kv_cache(1, dtype=DTYPE, device=device, kv_cache_size_override=size)
        p._initialize_crossattn_cache(batch_size=1, dtype=DTYPE, device=device)
        p.generator.model.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)

    def recache(p, output, f0, cond):
        p._recache_after_switch(output, f0, cond)
    return pipe, init_caches, recache if args.recache else None, None


def load_infinity_rope(args):
    """Infinity-RoPE as upstream inference.sh: configs/self_forcing_dmd.yaml merged into default_config.yaml, EMA generator of --ckpt
    (checkpoints/ema_model.pt), bf16, T5 on the GPU. At every switch upstream's KV flush (the sink frame and the last two cached frames are
    kept, fresh cross-attention cache); with --scene_cut the first chunk of the new prompt also gets its RoPE cut (upstream "[..s#]")."""
    from pipeline import CausalInferencePipeline
    cfg = OmegaConf.merge(OmegaConf.load(f"{args.repo}/configs/default_config.yaml"), OmegaConf.load(f"{args.repo}/configs/self_forcing_dmd.yaml"))
    pipe = CausalInferencePipeline(cfg, device=device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["generator_ema"]
    pipe.generator.load_state_dict({k.replace("_fsdp_wrapped_module.", ""): v for k, v in sd.items()})
    del sd
    pipe = pipe.to(dtype=DTYPE)
    pipe.text_encoder.to(device=device)
    pipe.generator.to(device=device)
    pipe.vae.to(device=device)

    def init_caches(p):
        p.kv_cache1 = p.crossattn_cache = None
        p._initialize_kv_cache(batch_size=1, dtype=DTYPE, device=device)
        p._initialize_crossattn_cache(batch_size=1, dtype=DTYPE, device=device)

    def kv_flush(p, output, f0, cond):          # upstream CausalInferencePipeline.inference: kv_flush
        for c, x in zip(p.kv_cache1, p.crossattn_cache):
            x["is_init"] = False
            c["k"][:, 1560:4680] = c["k"][:, -3120:]
            c["v"][:, 1560:4680] = c["v"][:, -3120:]
            c["local_end_index"] = torch.tensor([4680], dtype=torch.long, device=device)
            c["scene_cut"] = args.scene_cut

    def chunk_start(p, b, is_switch):           # the RoPE cut holds only for the first chunk of a prompt
        if not is_switch:
            for c in p.kv_cache1:
                c["scene_cut"] = False
    return pipe, init_caches, kv_flush, chunk_start


class Generator:
    """The upstream repo on sys.path (working directory --workdir), the backbone loaded and the attention patched."""

    def __init__(self, args):
        self.args = args
        sys.path.insert(0, HERE)
        sys.path.insert(0, os.path.abspath(args.repo))
        os.chdir(os.path.abspath(args.workdir or args.repo))
        torch.set_grad_enabled(False)
        import wan.modules.causal_model as cm
        import residual_guidance as RG
        self.RG = RG
        RG.patch_attention(cm)
        load = {"longlive": load_longlive, "infinity_rope": load_infinity_rope}.get(args.backbone, load_forcing)
        self.pipe, self.init_caches, self.on_switch, self.chunk_start = load(args)
        self._txt = {}

    def enc(self, text):
        if text not in self._txt:
            self._txt[text] = {k: v.to(device) for k, v in self.pipe.text_encoder(text_prompts=[text]).items()}
        return self._txt[text]

    def prepare(self, it):
        """per-chunk prompt names and the encoded prompts of an item"""
        bounds = [0, *it["switch_chunks"], it["n_chunks"]]
        chunk_prompt = [str(q) for q in range(len(it["prompts"])) for _ in range(bounds[q], bounds[q + 1])]
        return chunk_prompt, {str(q): self.enc(p) for q, p in enumerate(it["prompts"])}

    def rollout(self, it, method, chunk_prompt, conds, on_chunk=None):
        """latents of an item (seeded as in the paper)"""
        noise_frames[0] = it["n_chunks"] * FPB
        rg = self.RG.RGConfig(t=self.args.t, eps_seed=100000 + 100 * it["seed"]) if method == "rg" else None
        set_seed(it["seed"])
        noise = torch.randn([1, it["n_chunks"] * FPB, *LATENT], device=device, dtype=DTYPE)
        sw = set(self.RG.switch_chunks(chunk_prompt))

        def chunk_hook(b):
            if self.chunk_start is not None:
                self.chunk_start(self.pipe, b, b in sw)
            if on_chunk is not None:
                on_chunk(b)
        return self.RG.rollout(self.pipe, noise, conds, chunk_prompt, rg=rg, init_caches=self.init_caches, on_switch=self.on_switch, on_chunk=chunk_hook)

    def decode(self, lat):
        video = (self.pipe.vae.decode_to_pixel(lat, use_cache=False) * 0.5 + 0.5).clamp(0, 1)
        self.pipe.vae.model.clear_cache()
        return video


def load_items(args):
    items = json.load(open(os.path.abspath(args.prompts)))["items"]
    return [it for it in items if int(it["id"]) in args.ids] if args.ids else items


def main():
    ap = get_parser()
    ap.add_argument("--method", default="rg", choices=["naive", "rg"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_latents", action="store_true")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    items = load_items(args)
    G = Generator(args)
    from torchvision.io import write_video
    os.makedirs(args.out, exist_ok=True)
    for it in items:
        base = f"{args.out}/{int(it['id']):03d}"
        if os.path.exists(base + ".json"):
            continue
        lat = G.rollout(it, args.method, *G.prepare(it))
        if args.save_latents:
            torch.save(lat[0].to(torch.float16).cpu(), base + ".pt")
        video = G.decode(lat)
        write_video(base + ".mp4", (video[0].permute(0, 2, 3, 1).float() * 255).round().clamp(0, 255).to(torch.uint8).cpu(), fps=16, options={"crf": "18"})
        json.dump({**it, "backbone": args.backbone, "method": args.method, "recache": args.recache, "scene_cut": args.scene_cut, "t": args.t},
                  open(base + ".json", "w"))
        print(f"[done] {base}.mp4", flush=True)


if __name__ == "__main__":
    main()
