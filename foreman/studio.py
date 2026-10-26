"""Deployment configuration for the Mac Studio (M5 Ultra, 256 GB) running mlx-serve.

Model names are the ids mlx-serve's --model-dir discovery gives the packs under
~/.mlx-serve/models ('<org>/<repo>'). A pack loaded by absolute path is registered under its
directory basename instead; the adapter maps both back to these names.

Sizes and load times are calibrated on the Studio (2026-10-26, mlx-serve 26.10.3): the size is
the measured peak plus a modest margin, the load time the measured wall time with room to spare
(gpud gives a load 3x its load_s before it counts as failed). Measured:

    model                    resident (/api/ps size_vram)       load (wall)     plan
    Qwen3.8-27B brain        18.2 GB (load-model said 16.95)    2-6 s, up to 9  20 GB, 10 s
    Qwen3.8-Flash-Next       75.3 GB                            not measured    77 GB, 70 s
    Kokoro-82M               ~0.4 GB                            ~1 s             1 GB,  3 s
    ACE-Step 1.5 XL 4-bit    ~4 GB pack                         1-2 s            6 GB,  5 s
    LTX-2.5 4-bit            33.1 GB, 35.6 GB during a 5 s      3-5 s           40 GB, 10 s
                             1280x704 render

The earlier estimates (brain 24 GB/25 s, LTX 60 GB/45 s, Flash-Next 74 GB/70 s) were too
loose: LTX's 60 GB plus the 40 GB headroom made renders wait up to an hour while 88-110 GB were
free. The planner and the adapter's memory preflight also use the largest size mlx-serve
reported for a model since it started whenever that exceeds these numbers (Supervisor._size,
MlxServeEngine.plan_gb), so a model bigger than expected is never under-planned.

Configuration sources, later ones winning: defaults below, $FOREMAN_HOME/config.json (or the
file given with --config), environment variables (FOREMAN_*), command-line flags.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass, field

from .policy import ModelSpec, Policy, number

ENGINE = "mlx-serve"
BRAIN_PRIMARY = "ddalcu/Qwen3.8-27B-MLX-Serve-4bit"                    # vision (qwen3_5 + vision_config)
BRAIN_FALLBACK = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"    # text only
KOKORO = "ddalcu/Kokoro-82M-MLX-Serve"                                  # speech
ACE_STEP = "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit"                # music
LTX = "ddalcu/LTX-2.5-MLX-Serve-4bit"                                   # video (+ audio)
# MiniMax-H3 derived; its license excludes US users. Never in the catalog, and the adapter
# refuses anything matching it even if someone adds it.
FORBIDDEN = ("FastH3", "MiniMax-H3")
# Models whose licenses allow commercial use of their output (Kokoro Apache-2.0, ACE-Step MIT,
# LTX-2.x community license below its revenue threshold). The planner refuses to route a
# template flagged commercial to any model outside this list (fail closed).
COMMERCIAL_OK = frozenset({KOKORO, ACE_STEP, LTX})

OTHER_TENANTS_GB = 40.0  # other inference services on the host, outside mlx-serve (never touched)
OS_RESERVE_GB = 16.0
HEADROOM_GB = 40.0    # kept free on top of everything planned


def catalog(video_exclusive: bool = False) -> dict:
    """By default the 27B vision brain stays resident beside LTX (it fits here: about 18 + 36 GB),
    so reviews need no reload and mlx-serve's GenYield lets it answer between denoise steps.
    video_exclusive=True unloads the brain while LTX renders instead, which makes every render
    exercise the brain-return path (or frees memory on a smaller machine)."""
    specs = (  # sizes and load times measured on the Studio (module docstring)
        ModelSpec(BRAIN_PRIMARY, ENGINE, 20, 10, "brain", vision=True),
        ModelSpec(BRAIN_FALLBACK, ENGINE, 77, 70, "brain"),
        ModelSpec(KOKORO, ENGINE, 1, 3, "tts"),
        ModelSpec(ACE_STEP, ENGINE, 6, 5, "music"),
        ModelSpec(LTX, ENGINE, 40, 10, "video", evicts_brain=video_exclusive),
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
        idle_media_unload_s=300.0,   # idle 5 min: give media models' memory and slots back to the host
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
    video_exclusive: bool = False
    ffmpeg: str = "auto"              # path, "auto" (PATH, $FOREMAN_HOME/bin, Homebrew), or "" for none
    min_headroom_gb: float = HEADROOM_GB
    max_resident_models: int = 3      # mlx-serve's --max-resident-models (default 3)
    foreign_models: list = field(default_factory=list)  # catalog models that are always someone else's:
                                      # gpud never loads, unloads or claims them (it may use a resident brain)
    use_foreign_brains: bool = True   # a brain someone else loaded may serve gpud's brain jobs, read-only
    tick_s: float = 1.0
    watchdog_s: float = 300.0         # a tick stuck this long: exit and let launchd restart gpud
    stall_after: int | None = None    # fault injection (FOREMAN_FAULT_STALL_AFTER)
    stall_kinds: tuple | None = None  # ... limited to these job kinds (FOREMAN_FAULT_KINDS)
    progress_timeout_s: float | None = None  # override Policy.progress_timeout_s, the lease's no-progress
                                             # floor (FOREMAN_PROGRESS_TIMEOUT_S; default 120 s)
    log_level: str = "INFO"
    keep_raw_frames: bool = False     # keep frames.rgb after a successful MP4 mux (FOREMAN_KEEP_RAW_FRAMES)
    planner: bool = True              # run the planner inside gpud (FOREMAN_PLANNER); it does nothing
                                      # until an active template exists in $FOREMAN_HOME/templates
    planner_options: dict = field(default_factory=dict)  # PlannerConfig fields, e.g. {"interval_s": 1800}

    def policy(self) -> Policy:
        overrides = {"host_reserve_gb": OS_RESERVE_GB + OTHER_TENANTS_GB + float(self.min_headroom_gb),
                     "foreign_models": tuple(str(m) for m in (self.foreign_models or ())),
                     "use_foreign_brains": bool(self.use_foreign_brains)}
        if self.progress_timeout_s:
            overrides["progress_timeout_s"] = float(self.progress_timeout_s)
        return policy(**overrides)

    def planner_config(self):
        from .planner import PlannerConfig
        return PlannerConfig.from_dict(self.planner_options)

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
    "FOREMAN_LOG_LEVEL": ("log_level", str), "FOREMAN_FFMPEG": ("ffmpeg", str),
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
    for var, attr in (("FOREMAN_PLANNER", "planner"), ("FOREMAN_KEEP_RAW_FRAMES", "keep_raw_frames")):
        if env.get(var) not in (None, ""):
            setattr(cfg, attr, _bool(env[var]))
    if env.get("FOREMAN_PLANNER_INTERVAL_S") not in (None, ""):
        cfg.planner_options = dict(cfg.planner_options or {}, interval_s=float(env["FOREMAN_PLANNER_INTERVAL_S"]))
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if not isinstance(cfg.planner_options, dict):
        raise ValueError(f"{path}: planner_options must be an object of planner settings")
    if not isinstance(cfg.foreign_models, (list, tuple)) or not all(isinstance(m, str) for m in cfg.foreign_models):
        raise ValueError(f"{path}: foreign_models must be a list of model names")
    cfg.foreign_models = [catalog_model(m, f"{path}: foreign_models") for m in cfg.foreign_models]
    if number(cfg.min_headroom_gb) is None or cfg.min_headroom_gb < 0:
        raise ValueError(f"{path}: min_headroom_gb must be a number >= 0")
    usable = cfg.policy().usable_gb
    brain = catalog()[BRAIN_PRIMARY]
    if usable < brain.size_gb:
        raise ValueError(f"min_headroom_gb {cfg.min_headroom_gb:g} leaves {usable:g} GB to plan with "
                         f"(256 GB - {OS_RESERVE_GB:g} macOS - {OTHER_TENANTS_GB:g} other services - the headroom), "
                         f"less than the {brain.size_gb:g} GB brain: nothing could ever be loaded. At most "
                         f"{256.0 - OS_RESERVE_GB - OTHER_TENANTS_GB - brain.size_gb:g} GB")
    cfg.planner_config()  # fail early on an unknown or malformed planner option
    return cfg.finish()


def catalog_model(name: str, where: str = "model") -> str:
    """The catalog name for a model written the way mlx-serve shows it: the catalog name itself
    ('<org>/<repo>'), the directory basename a load by path registers (as /api/ps lists it, with
    or without ':latest'), or the pack's path. Anything else is refused: an override that names a
    model gpud never compares against would be silently ineffective."""
    names = list(catalog())
    text = str(name).strip()
    if text.endswith(":latest"):
        text = text[:-len(":latest")]
    if text in names:
        return text
    base = text.rstrip("/").rsplit("/", 1)[-1]
    hits = [n for n in names if n.rsplit("/", 1)[-1] == base]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"{where}: {name!r} is not a catalog model (known: {names}); models outside the catalog are "
                     f"never unloaded on the Studio anyway")


def find_ffmpeg(setting: str, home: str) -> str | None:
    """launchd gives gpud a minimal PATH, so "auto" also looks where per-user and Homebrew
    installs put ffmpeg."""
    if setting != "auto":
        return setting or None
    for cand in (shutil.which("ffmpeg"), os.path.join(home, "bin", "ffmpeg"),
                 "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def build_engine(cfg: Config, brain=None, **kw):
    from .adapters.mlx_serve import MlxServeEngine
    kw.setdefault("ffmpeg", find_ffmpeg(cfg.ffmpeg, cfg.home))
    return MlxServeEngine(cfg.base_url, catalog(cfg.video_exclusive), name=ENGINE, model_dir=cfg.model_dir,
                          artifacts_dir=cfg.artifacts_dir, brain=brain, min_headroom_gb=cfg.min_headroom_gb,
                          max_resident_models=cfg.max_resident_models or None, stall_after=cfg.stall_after,
                          stall_kinds=cfg.stall_kinds, forbidden=FORBIDDEN, keep_raw_frames=cfg.keep_raw_frames,
                          **kw)


def build_planner(cfg: Config, **kw):
    """The planner gpud runs: templates, goals and inbox under cfg.home, retention on cfg.artifacts_dir."""
    from .planner import Planner
    return Planner(cfg.planner_config(), home=cfg.home, artifacts_dir=cfg.artifacts_dir,
                   commercial_models=COMMERCIAL_OK, **kw)
