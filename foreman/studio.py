"""Deployment configuration for the Mac Studio (M5 Ultra, 256 GB) running mlx-serve.

Model names are the ids mlx-serve's --model-dir discovery gives the packs under
~/.mlx-serve/models ('<org>/<repo>'). A pack loaded by absolute path is registered under its
directory basename instead; the adapter maps both back to these names.

Sizes are planning numbers: the on-disk size of each pack plus headroom for activations and
KV cache (LTX's VAE decode peaks around 20 GB on top of its weights). Load times are rough SSD
estimates. Measure both on the Studio before trusting the planner with tighter margins.

Configuration sources, later ones winning: defaults below, $FOREMAN_HOME/config.json (or the
file given with --config), environment variables (FOREMAN_*), command-line flags.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass

from .policy import ModelSpec, Policy

ENGINE = "mlx-serve"
BRAIN_PRIMARY = "ddalcu/Qwen3.8-27B-MLX-Serve-4bit"                    # vision (qwen3_5 + vision_config)
BRAIN_FALLBACK = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"    # text only
KOKORO = "ddalcu/Kokoro-82M-MLX-Serve"                                  # speech
ACE_STEP = "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit"                # music
LTX = "ddalcu/LTX-2.5-MLX-Serve-4bit"                                   # video (+ audio)
# MiniMax-H3 derived; its license excludes US users. Never in the catalog, and the adapter
# refuses anything matching it even if someone adds it.
FORBIDDEN = ("FastH3", "MiniMax-H3")

OTHER_TENANTS_GB = 40.0  # other inference services on the host, outside mlx-serve (never touched)
OS_RESERVE_GB = 16.0
HEADROOM_GB = 40.0    # kept free on top of everything planned


def catalog(video_exclusive: bool = True) -> dict:
    """video_exclusive=True unloads the brain while LTX renders, so every render exercises the
    brain-return path. A co-resident small brain (False) is the production-friendlier choice:
    the 27B vision brain fits beside LTX here, reviews need no reload, and mlx-serve's GenYield
    lets it answer between denoise steps."""
    specs = (
        ModelSpec(BRAIN_PRIMARY, ENGINE, 24, 25, "brain", vision=True),
        ModelSpec(BRAIN_FALLBACK, ENGINE, 74, 70, "brain"),
        ModelSpec(KOKORO, ENGINE, 1, 5, "tts"),
        ModelSpec(ACE_STEP, ENGINE, 8, 15, "music"),
        ModelSpec(LTX, ENGINE, 60, 45, "video", evicts_brain=video_exclusive),
    )
    return {s.name: s for s in specs}


def policy(**overrides) -> Policy:
    base = Policy(
        host_memory_gb=256.0,
        host_reserve_gb=OS_RESERVE_GB + OTHER_TENANTS_GB + HEADROOM_GB,
        text_ladder=(BRAIN_PRIMARY, BRAIN_FALLBACK),
        vision_brain=BRAIN_PRIMARY,
        co_resident_brain=BRAIN_PRIMARY,
        brain_job_expected_s=60.0,   # hard timeout 150 s for one brain decision
        unload_foreign=False,        # the user's own models in mlx-serve are never unloaded
    )
    return dataclasses.replace(base, **overrides)


def _home() -> str:
    return os.path.expanduser(os.environ.get("FOREMAN_HOME", "~/.foreman"))


@dataclass
class Config:
    home: str = ""
    db_path: str = ""
    log_dir: str = ""
    artifacts_dir: str = ""
    base_url: str = "http://127.0.0.1:11234"
    model_dir: str = "~/.mlx-serve/models"
    video_exclusive: bool = True
    min_headroom_gb: float = HEADROOM_GB
    max_resident_models: int = 3      # mlx-serve's --max-resident-models (default 3)
    tick_s: float = 1.0
    watchdog_s: float = 300.0         # a tick stuck this long: exit and let launchd restart gpud
    stall_after: int | None = None    # fault injection (FOREMAN_FAULT_STALL_AFTER)
    stall_kinds: tuple | None = None  # ... limited to these job kinds (FOREMAN_FAULT_KINDS)
    progress_timeout_s: float | None = None  # override Policy.progress_timeout_s, the lease's no-progress
                                             # floor (FOREMAN_PROGRESS_TIMEOUT_S; default 120 s)
    log_level: str = "INFO"

    def policy(self) -> Policy:
        overrides = {}
        if self.progress_timeout_s:
            overrides["progress_timeout_s"] = float(self.progress_timeout_s)
        return policy(**overrides)

    def finish(self) -> "Config":
        self.home = os.path.expanduser(self.home or _home())
        self.db_path = os.path.expanduser(self.db_path or os.path.join(self.home, "foreman.db"))
        self.log_dir = os.path.expanduser(self.log_dir or os.path.join(self.home, "logs"))
        self.artifacts_dir = os.path.expanduser(self.artifacts_dir or os.path.join(self.home, "artifacts"))
        self.model_dir = os.path.expanduser(self.model_dir)
        return self


_ENV = {
    "FOREMAN_DB": ("db_path", str), "FOREMAN_LOG_DIR": ("log_dir", str),
    "FOREMAN_ARTIFACTS": ("artifacts_dir", str), "FOREMAN_MLX_URL": ("base_url", str),
    "FOREMAN_MODEL_DIR": ("model_dir", str), "FOREMAN_MIN_HEADROOM_GB": ("min_headroom_gb", float),
    "FOREMAN_MAX_RESIDENT_MODELS": ("max_resident_models", int), "FOREMAN_TICK_S": ("tick_s", float),
    "FOREMAN_WATCHDOG_S": ("watchdog_s", float), "FOREMAN_FAULT_STALL_AFTER": ("stall_after", int),
    "FOREMAN_PROGRESS_TIMEOUT_S": ("progress_timeout_s", float),
    "FOREMAN_LOG_LEVEL": ("log_level", str),
}


def _bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | None = None, env=None, **overrides) -> Config:
    env = os.environ if env is None else env
    cfg = Config(home=os.path.expanduser(env.get("FOREMAN_HOME", "~/.foreman")))
    path = path or os.path.join(cfg.home, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        names = {f.name for f in dataclasses.fields(Config)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown}")
        for k, v in data.items():
            setattr(cfg, k, tuple(v) if k == "stall_kinds" and v else v)
    for var, (attr, conv) in _ENV.items():
        if env.get(var) not in (None, ""):
            setattr(cfg, attr, conv(env[var]))
    if env.get("FOREMAN_VIDEO_EXCLUSIVE") not in (None, ""):
        cfg.video_exclusive = _bool(env["FOREMAN_VIDEO_EXCLUSIVE"])
    if env.get("FOREMAN_FAULT_KINDS"):
        cfg.stall_kinds = tuple(k.strip() for k in env["FOREMAN_FAULT_KINDS"].split(",") if k.strip())
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg.finish()


def build_engine(cfg: Config, brain=None, **kw):
    from .adapters.mlx_serve import MlxServeEngine
    return MlxServeEngine(cfg.base_url, catalog(cfg.video_exclusive), name=ENGINE, model_dir=cfg.model_dir,
                          artifacts_dir=cfg.artifacts_dir, brain=brain, min_headroom_gb=cfg.min_headroom_gb,
                          max_resident_models=cfg.max_resident_models or None, stall_after=cfg.stall_after,
                          stall_kinds=cfg.stall_kinds, forbidden=FORBIDDEN, **kw)
