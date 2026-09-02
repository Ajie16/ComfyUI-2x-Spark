"""Ulysses sequence-parallel MiniMax-H3 DiT loader for ComfyUI.

Drop-in replacement for UNETLoader that additionally spawns world_size-1 worker
processes (one per extra GPU) and routes every diffusion-model forward through
a sequence-parallel path. Mathematically equivalent to single-GPU sampling:
same latents, same video/audio output.

Single-node (two local GPUs):
    python main.py --cuda-device 0,1 --highvram

Two DGX Sparks (one GPU each): leave peer_host=spark-2 and master_ip at the
CX-7 fabric address of this ComfyUI box (spark-1). Rank 1 is started over SSH.
"""

import logging
import sys

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.sd
import comfy.utils
import folder_paths

from . import spark_cluster
from . import sp_group
from . import sp_vae

sp_vae.enable_stack_tiling()

try:
    from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: F401
except ImportError:
    raise RuntimeError(
        "buqi-minimax-h3-multigpu requires a ComfyUI build that ships the "
        "MiniMax-H3 model (comfy.ldm.minimax.model); update ComfyUI to >= 0.30.0"
    )


def resolve_devices(devices, world):
    if devices and devices.strip() and devices.strip().lower() != "auto":
        picked = [d.strip() for d in devices.split(",")]
        if len(picked) != world:
            raise ValueError(f'devices "{devices}" lists {len(picked)} GPUs, world_size is {world}')
        return picked
    import os
    env = os.environ.get("MINIMAX_SP_DEVICES")
    if env:
        picked = [d.strip() for d in env.split(",")]
        if len(picked) < world:
            raise ValueError(f"MINIMAX_SP_DEVICES lists {len(picked)} devices, need {world}")
        return picked[:world]
    return [str(i) for i in range(world)]


class MiniMaxH3SPUNETLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPUNETLoader",
            display_name="MiniMax H3 Multi-GPU Loader (Ulysses SP)",
            category="advanced/multigpu",
            description="Loads the MiniMax-H3 DiT and shards the packed sequence across "
                        "world_size GPUs (Ulysses all-to-all). Numerically equivalent to "
                        "single-GPU sampling.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]),
                io.Int.Input("world_size", default=2, min=1, max=8,
                             tooltip="Ranks to shard across. Must divide the 56 attention heads "
                                     "(valid: 1, 2, 4, 7, 8). Two Sparks use 2."),
                io.String.Input("devices", default="auto",
                                tooltip='Local CUDA ids when peer_host is empty, e.g. "0,1". '
                                        'Ignored for the remote rank when peer_host is set.'),
                io.String.Input("peer_host", default=spark_cluster.DEFAULT_PEER_HOST,
                                tooltip='SSH host for rank 1 (spark-2). Empty / "local" = spawn '
                                        'workers on this machine (needs extra local GPUs).'),
                io.String.Input("master_ip", default=spark_cluster.DEFAULT_MASTER_IP,
                                tooltip="Rank-0 fabric IP for NCCL/gloo rendezvous. "
                                        "spark-1 CX-7 address."),
                io.Combo.Input("lora_name", options=["none"] + folder_paths.get_filename_list("loras"),
                               tooltip="Applied on every rank (Comfy LoRA nodes after this loader "
                                       "only patch rank 0 and will desync Ulysses). Use none to skip."),
                io.Float.Input("lora_strength", default=1.0, min=-2.0, max=2.0, step=0.05),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype, world_size, devices="auto",
                peer_host="", master_ip="", lora_name="none", lora_strength=1.0) -> io.NodeOutput:
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        lora_name = "" if (not lora_name or lora_name == "none") else lora_name
        if lora_name:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, lora_strength, 0)
            del lora_sd
        if world_size <= 1:
            return io.NodeOutput(model)

        if sys.platform == "win32":
            raise RuntimeError("multi-GPU SP needs NCCL, which is not available on native Windows; "
                               "run ComfyUI inside WSL2 or on Linux")

        heads = model.model.diffusion_model.blocks[0].attn.heads
        if heads % world_size:
            raise ValueError(f"world_size {world_size} must divide the {heads} attention heads "
                             "(valid: 1, 2, 4, 7, 8)")

        peer_host = (peer_host or "").strip()
        master_ip = (master_ip or spark_cluster.DEFAULT_MASTER_IP).strip()
        cross_node = peer_host.lower() not in ("", "local", "none")
        if cross_node:
            picked = ["0"] * world_size
            if torch.cuda.device_count() < 1:
                raise ValueError("no local CUDA device for rank 0")
        else:
            picked = resolve_devices(devices, world_size)
            if torch.cuda.device_count() < world_size:
                raise ValueError(f"world_size is {world_size} but only {torch.cuda.device_count()} GPUs "
                                 "are visible; start ComfyUI with e.g. --cuda-device 0,1 "
                                 "or set peer_host to the other Spark")

        sp_kw = dict(
            peer_host=peer_host if cross_node else "",
            master_ip=master_ip if cross_node else "127.0.0.1",
            lora_name=lora_name, lora_strength=lora_strength,
        )
        sp_group.get_group(world_size, unet_name, weight_dtype, picked, **sp_kw)

        model = model.clone()
        dit = model.get_model_object("diffusion_model")

        def sp_entry(x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
            group = sp_group.get_group(world_size, unet_name, weight_dtype, picked, **sp_kw)
            return group.forward(dit, x, timestep, context, transformer_options, minimax_payload)

        # patch _forward instead of adding a DIFFUSION_MODEL wrapper: a wrapper that
        # never calls executor() short-circuits the chain and silently disables other
        # wrappers on the same hook. forward() resolves _forward per call, so this
        # keeps the chain intact with the sharded path as its inner call.
        model.add_object_patch("diffusion_model._forward", sp_entry)
        logging.info(f"[minimax_sp] sequence parallel enabled, world_size={world_size} devices={picked}")
        return io.NodeOutput(model)


class MiniMaxH3SPVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPVAEDecode",
            display_name="MiniMax H3 Multi-GPU VAE Decode",
            category="advanced/multigpu",
            description="Drop-in VAEDecode that spreads the video VAE's temporal chunks "
                        "across the GPUs of a running MiniMax H3 SP group. Falls back to "
                        "the stock decode when that is not possible. Costs each worker the "
                        "video VAE weights (~5 GB) on top of the DiT.",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, samples, vae) -> io.NodeOutput:
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        group = sp_group.active_group()
        images = None
        if group is None:
            logging.info("[minimax_sp] no SP group running, decoding the VAE on one GPU")
        elif group.world > 1 and sp_vae.is_supported(vae):
            images = group.vae_decode(vae, latent)
            if images is None:
                logging.info("[minimax_sp] too few temporal chunks to shard, "
                             "decoding the VAE on one GPU")
        if images is None:
            images = vae.decode(latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return io.NodeOutput(images)


class MiniMaxSPExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SPUNETLoader, MiniMaxH3SPVAEDecode]


async def comfy_entrypoint() -> ComfyExtension:
    return MiniMaxSPExtension()
