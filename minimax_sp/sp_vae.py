"""Shard the MiniMax-H3 video VAE decode across the sequence-parallel group.

decode_temporal() walks the latent one temporal chunk at a time and each chunk
only reads a slice of z, so the chunks are independent and can be spread over the
ranks that already hold the group. Only `_adaptive_decode` is farmed out: every
spatial tile blend, temporal blend and canvas write stays in the official code
path on rank 0, which is fed the precomputed chunks.

Workers are handed the VAE weights over NCCL rather than a filename, so the
sharded decode always uses exactly the checkpoint the user loaded.

Pinned to ComfyUI 0.30.0's comfy/ldm/minimax/vae.py: chunk_plan and
prepare_latent mirror the padding, chunk bounds and denormalization of
decode()/decode_temporal() and must be re-checked when that file changes.

Spatial 256px tiles are batched into one decoder forward per temporal chunk
when H3_VAE_STACK_TILING is unset/1 (same as the vLLM 2x path). Set it to 0
to restore one-forward-per-tile. Do not raise tile_size to 512: the decoder
was trained on 256px tiles and that produces grid artifacts.
"""

import logging
import os

import torch

_STACK_PATCHED = False


def stack_tiling_enabled():
    return os.environ.get("H3_VAE_STACK_TILING", "1").lower() not in ("0", "false", "no", "off")


def is_supported(vae):
    fsm = getattr(vae, "first_stage_model", None)
    return all(hasattr(fsm, a) for a in
               ("_adaptive_decode", "tokens_chunk_size", "token_overlap", "token_drop",
                "vae_ratio", "vae_ratio_t", "latents_mean", "latents_std"))


def chunk_plan(fsm, z_len):
    """Mirror decode_temporal's padding and per-chunk token bounds."""
    pseudo = z_len + fsm.token_drop
    pad_tokens = 0
    remainder = pseudo % fsm.tokens_chunk_size
    if remainder != 0:
        pad_tokens = fsm.tokens_chunk_size - remainder
        pseudo += pad_tokens
    num_chunks = pseudo // fsm.tokens_chunk_size - int(fsm.token_drop > 0)
    if num_chunks < 1:
        pad_tokens += fsm.tokens_chunk_size
        num_chunks += 1

    padded_len = z_len + pad_tokens
    bounds = []
    for i in range(num_chunks):
        t0 = i * fsm.tokens_chunk_size
        t1 = min(t0 + fsm.tokens_chunk_size + fsm.token_overlap, padded_len)
        bounds.append((t0, t1))
    return pad_tokens, bounds


def prepare_latent(fsm, z, pad_tokens):
    """Apply decode()'s denormalization then decode_temporal()'s tail padding."""
    mean = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
    std = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
    z = z * std + mean
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    return z


def chunk_shape(fsm, z, t0, t1):
    """Pixel shape _adaptive_decode returns for one chunk: 4 frames per token, 16x spatial."""
    return (z.shape[0], 3, (t1 - t0) * fsm.vae_ratio_t,
            z.shape[3] * fsm.vae_ratio, z.shape[4] * fsm.vae_ratio)


def decode_local(fsm, z_prepared, bounds, rank, world):
    """Decode the chunks this rank owns. Round-robin keeps the load even."""
    out = {}
    for i, (t0, t1) in enumerate(bounds):
        if i % world == rank:
            out[i] = fsm._adaptive_decode(z_prepared[:, :, t0:t1])
    return out


def _decode_spatial_tiles(vae, z, y_idx, y_len, x_idx, x_len):
    """Run the spatial-tile decoder, optionally stacking tiles on the batch dim."""
    tiles = []
    for i_pos, i_len in zip(y_idx, y_len):
        zi, zl = i_pos // vae.vae_ratio, i_len // vae.vae_ratio
        for j_pos, j_len in zip(x_idx, x_len):
            zj, zw = j_pos // vae.vae_ratio, j_len // vae.vae_ratio
            tiles.append(z[..., zi:zi + zl, zj:zj + zw])
    if not tiles:
        return tiles
    if stack_tiling_enabled() and len({t.shape for t in tiles}) == 1:
        batch = tiles[0].shape[0]
        stacked = vae._decode_pixels(torch.cat(tiles, dim=0))
        return list(stacked.unflatten(0, (len(tiles), batch)).unbind(0))
    return [vae._decode_pixels(t) for t in tiles]


def _tiled_decode_maybe_stacked(self, z):
    """Comfy tiled_decode with optional stacked spatial forwards.

    Blend / canvas write is identical to comfy/ldm/minimax/vae.py; only the
    per-tile `_decode_pixels` calls are fused when every tile has the same
    shape (H3's 256px split_tiles always does on 864x480 / 832x480).
    """
    height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)
    pix = _decode_spatial_tiles(self, z, y_idx, y_len, x_idx, x_len)

    canvas = None
    row_tails = []
    out_y = 0
    k = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        new_tails = []
        left_tail = None
        out_x = 0
        for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
            tile = pix[k]
            k += 1
            if i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i]:, :].clone())
            next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
            if i > 0:
                tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            left_tail = next_left_tail
            if i < len(y_idx) - 1:
                tile = tile[..., :-y_overlap[i], :]
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
        row_tails = new_tails
        out_y += tile.shape[-2]
    return canvas


def enable_stack_tiling():
    """Replace MiniMaxH3VideoVAE.tiled_decode on rank 0 and the worker."""
    global _STACK_PATCHED
    if _STACK_PATCHED:
        return
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    MiniMaxH3VideoVAE.tiled_decode = _tiled_decode_maybe_stacked
    _STACK_PATCHED = True
    logging.info("[minimax_sp] VAE stack_tiling=%s (H3_VAE_STACK_TILING, default on)",
                 stack_tiling_enabled())
