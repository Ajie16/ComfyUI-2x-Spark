# Two-Spark Ulysses SP for ComfyUI MiniMax H3

Stock `buqi-minimax-h3-multigpu` assumes extra GPUs on the same box. This
checkout adds a **cross-node** rank: ComfyUI on this Spark (rank 0) SSH-starts
a DiT worker on the peer Spark (rank 1) and they Ulysses-shard the packed
sequence over CX-7 RoCEv2.

Text encoder, VAE, and the Comfy sampler stay on rank 0. Only the DiT (and
optional VAE-decode shard) runs on both cards.

## Layout

| Rank | Host | Role |
|---|---|---|
| 0 | spark-1 (this ComfyUI; OS `spark-0d97`) | UI, TE, VAE, DiT, sampler |
| 1 | spark-2 (peer; OS `spark-ac8f`) | DiT replica, Ulysses worker |

Fabric: `enP2p1s0f0np0`, HCA `roceP2p1s0f0`, GID 3.
Rendezvous: `tcp://10.100.65.2:29511`.

## One-time sync

Paths must be identical on both boxes. From this machine:

```bash
custom_nodes/buqi-minimax-h3-multigpu/scripts/sync-peer.sh
```

Copies ComfyUI code + `.venv` + the Ref2VA INT8 DiT + turbo LoRA to spark-2.
The vLLM 2x containers must be **down** first — they hold the GPUs.

## Workflow

Replace `Load Diffusion Model` / `UNETLoader` with **MiniMax H3 Multi-GPU Loader (Ulysses SP)**:

- `unet_name`: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `weight_dtype`: `default` (INT8 ConvRot is in the file)
- `world_size`: `2`
- `peer_host`: `spark-2`
- `master_ip`: `10.100.65.2`
- `lora_name`: `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (or `none`)
- `lora_strength`: `1.0`

Do **not** put a Comfy LoRA loader after this node: that would patch rank 0
only and desync the all-to-alls. LoRA is applied on both ranks here.

Then the stock H3 graph: CLIP, VAE, `MiniMax H3 Reference to Video`, sampler,
`VAEDecode` (or the bundled multi-GPU VAE decode).

## Launch

```bash
cd /home/xujie/workspace/ComfyUI
source .venv/bin/activate
python main.py --listen 0.0.0.0 --highvram
```

First prompt of a session SSH-starts the worker and loads ~20 GiB on spark-2
(a few minutes). Later prompts reuse the group until ComfyUI exits.

Worker log: `/tmp/minimax_sp_worker1.log` on spark-2.

## Stop vLLM

```bash
cd /home/xujie/workspace/dgx-spark-minimax-h3/MiniMax-H3-2x-DGX-Spark
make down
```
