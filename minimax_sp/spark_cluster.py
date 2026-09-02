"""Two-Spark (cross-node) rendezvous for MiniMax-H3 Ulysses SP.

The upstream plugin spawns extra ranks as local subprocesses with
CUDA_VISIBLE_DEVICES. A DGX Spark has one GPU, so the second rank has to
live on the peer box and join over the CX-7 RoCEv2 link.

Environment (all optional; the loader node can override):

  H3_SP_PEER_HOST     SSH host for rank 1 (default: spark-2)
  H3_SP_MASTER_IP     Rank-0 fabric IP used as NCCL/gloo rendezvous
                      (default: 10.100.65.2, this machine's CX-7)
  H3_SP_PORT          TCP rendezvous port (default: 29511)
  NCCL_SOCKET_IFNAME  default enP2p1s0f0np0
  NCCL_IB_HCA         default roceP2p1s0f0
  NCCL_IB_GID_INDEX   default 3
"""

from __future__ import annotations

import os
import shlex
import subprocess

DEFAULT_PEER_HOST = "spark-2"
DEFAULT_MASTER_IP = "10.100.65.2"
DEFAULT_PORT = 29511
DEFAULT_IFACE = "enP2p1s0f0np0"
DEFAULT_HCA = "roceP2p1s0f0"
DEFAULT_GID = "3"


def apply_nccl_env() -> None:
    os.environ.setdefault("NCCL_SOCKET_IFNAME", DEFAULT_IFACE)
    os.environ.setdefault("NCCL_IB_HCA", DEFAULT_HCA)
    os.environ.setdefault("NCCL_IB_GID_INDEX", DEFAULT_GID)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ["NCCL_SOCKET_IFNAME"])
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "PHB")


def cluster_defaults():
    peer = os.environ.get("H3_SP_PEER_HOST", DEFAULT_PEER_HOST).strip()
    master = os.environ.get("H3_SP_MASTER_IP", DEFAULT_MASTER_IP).strip()
    port = int(os.environ.get("H3_SP_PORT", str(DEFAULT_PORT)))
    return peer, master, port


def find_comfy_root(start=None) -> str:
    """Walk up from this file until ComfyUI's main.py + comfy/ package."""
    d = os.path.dirname(os.path.abspath(start or __file__))
    while True:
        if os.path.isfile(os.path.join(d, "main.py")) and os.path.isdir(os.path.join(d, "comfy")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"cannot locate ComfyUI root walking up from {start or __file__}")
        d = parent


def spawn_ssh_worker(
    *,
    peer_host: str,
    rank: int,
    world: int,
    port: int,
    master_ip: str,
    unet_name: str,
    weight_dtype: str,
    comfy_root: str,
    script: str,
    log_path: str,
    lora_name: str = "",
    lora_strength: float = 1.0,
) -> subprocess.Popen:
    """Start rank >= 1 on the peer Spark. Paths must match this machine."""
    python = os.path.join(comfy_root, ".venv", "bin", "python")
    if not os.path.isfile(python):
        raise RuntimeError(f"ComfyUI venv python missing on rank 0: {python}")
    pre = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", peer_host,
         f"test -x {shlex.quote(python)} && test -f {shlex.quote(script)}"],
        capture_output=True, text=True)
    if pre.returncode != 0:
        raise RuntimeError(
            f"peer {peer_host} missing {python} or {script} "
            f"(rc={pre.returncode} stderr={pre.stderr.strip()!r}). "
            f"comfy_root={comfy_root}")
    user_site = os.path.expanduser("~/.local/lib/python3.12/site-packages")
    iface = os.environ.get("NCCL_SOCKET_IFNAME", DEFAULT_IFACE)
    hca = os.environ.get("NCCL_IB_HCA", DEFAULT_HCA)
    gid = os.environ.get("NCCL_IB_GID_INDEX", DEFAULT_GID)
    remote = (
        f"mkdir -p {shlex.quote(os.path.dirname(log_path))} ; "
        f"export CUDA_VISIBLE_DEVICES=0 "
        f"PYTHONPATH={shlex.quote(user_site)} "
        f"NCCL_SOCKET_IFNAME={shlex.quote(iface)} "
        f"NCCL_IB_HCA={shlex.quote(hca)} "
        f"NCCL_IB_GID_INDEX={shlex.quote(gid)} "
        f"GLOO_SOCKET_IFNAME={shlex.quote(iface)} "
        f"NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0 "
        f"PYTHONUNBUFFERED=1 ; "
        f"cd {shlex.quote(comfy_root)} && "
        f"exec {shlex.quote(python)} {shlex.quote(script)} "
        f"--rank {rank} --world {world} --port {port} "
        f"--master {shlex.quote(master_ip)} "
        f"--unet {shlex.quote(unet_name)} "
        f"--weight-dtype {shlex.quote(weight_dtype)} "
        f"--lora {shlex.quote(lora_name)} "
        f"--lora-strength {lora_strength} "
        f"--comfy-root {shlex.quote(comfy_root)} "
        f">> {shlex.quote(log_path)} 2>&1"
    )
    log = open(log_path + ".ssh", "w")
    return subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", peer_host, remote],
        stdout=log,
        stderr=log,
    )
