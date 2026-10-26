"""Owner-edited job templates, goals and the inbox the planner reads.

    $FOREMAN_HOME/templates/*.json   one template per file (schema below; examples/ has generic ones)
    $FOREMAN_HOME/goals.md           what the owner wants this month, in plain words
    $FOREMAN_HOME/inbox/*            inbound notes (briefs, requests); shown to the planner as
                                     <untrusted> data, never as instructions

A template is the only way work can exist: the planner may propose instances of ACTIVE
templates and nothing else. It cannot create, edit or activate a template, and it cannot change
a cap. Everything a proposal may vary is declared here with bounds.

Template fields (unknown keys are an error, so typos do not silently weaken a limit):

    id               "demo.narration.v1"   lowercase, digits, . _ -
    kind             speak | compose_music | render_video
    family           tts | music | video   (must match the kind)
    model            optional: a specific catalog model of that family
    consumer         who uses the output: a string or a list of strings. Proposals must name
                     one of them. Generic audiences ("anyone", "youtube", "followers") are refused
    revenue_line     what the work is for ("client_work", "internal_tooling", ...)
    value_usd        what one accepted item is worth to the owner
    prior            {"alpha": a, "beta": b}: Beta prior on P(ship); human verdicts update it
    caps             {"per_week": n, "max_stock": n, "two_week_use": n}
    params           {"seconds": {"min": 10, "max": 30, "default": 15},
                      "size": {"allowed": ["768x512"], "default": "768x512"}, ...}
                     speak: voice, speed, seed; compose_music: seconds, seed, instrumental, bpm;
                     render_video: seconds, size, steps, seed
    style_prefix     prepended to the brief to form the model prompt
    brief_max_chars  longest brief a proposal may carry (default 600, at most 2000)
    flags            {"speculative", "filler", "commercial", "private", "hard_deadline"}: booleans
    review_minutes   the owner's minutes to review one item
    active           only active templates are offered to the planner (default false)
    notes            free text for the owner (also shown to the planner)

Banned outright (a template matching any of these does not load, and a brief matching them is
rejected): platform-payout content farms, uncommissioned media with no named consumer, mass SEO
pages, AI-to-AI promotion, autonomous outreach, anything outward-facing (posting, uploading,
publishing, sending). Foreman makes files for a human to review; it never ships anything itself.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field

from .policy import number

KINDS = {"speak": "tts", "compose_music": "music", "render_video": "video"}
PARAM_TYPES = {
    "speak": {"voice": "choice", "speed": "number", "seed": "int"},
    "compose_music": {"seconds": "number", "seed": "int", "instrumental": "bool", "bpm": "int"},
    "render_video": {"seconds": "number", "size": "size", "steps": "int", "seed": "int"},
}
REQUIRED_PARAMS = {"compose_music": ("seconds",), "render_video": ("seconds", "size")}
# What the engines accept at all; a template's bounds must sit inside these.
HARD_BOUNDS = {
    ("compose_music", "seconds"): (10.0, 600.0), ("render_video", "seconds"): (0.5, 20.0),
    ("render_video", "steps"): (1, 100), ("speak", "speed"): (0.5, 2.0), ("compose_music", "bpm"): (30, 300),
    ("speak", "seed"): (0, 2 ** 31 - 1), ("compose_music", "seed"): (0, 2 ** 31 - 1),
    ("render_video", "seed"): (0, 2 ** 31 - 1),
}
FLAG_KEYS = ("speculative", "filler", "commercial", "private", "hard_deadline")
CAP_KEYS = ("per_week", "max_stock", "two_week_use")
TOP_KEYS = ("id", "kind", "family", "model", "consumer", "revenue_line", "value_usd", "prior", "caps",
            "params", "style_prefix", "brief_max_chars", "flags", "review_minutes", "active", "notes")
REQUIRED_KEYS = ("id", "kind", "family", "consumer", "revenue_line", "value_usd", "prior", "caps",
                 "review_minutes")
# Keys that would describe an outward action. Foreman never acts outside the box.
OUTWARD_KEYS = ("outward", "publish", "publish_to", "post", "post_to", "upload", "upload_to", "send",
                "send_to", "destination", "recipients", "outreach", "auto_publish", "auto_post", "webhook",
                "url", "urls")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")
MAX_BRIEF = 2000

BANNED = (
    (r"content farm", "a platform-payout content farm"),
    (r"\bpayouts?\b|\bmonetiz|\bcreator fund\b|\bstreaming royalt|\bad revenue share", "platform-payout content"),
    (r"\bmass (seo|pages?|produced|production|upload|uploads|posting|posts?)\b|\bseo (pages?|farm|spam)\b"
     r"|\bprogrammatic seo\b|\bscaled content\b", "mass SEO pages"),
    (r"\bai (to|2) ai\b|\bbots? (to|2) bots?\b", "AI-to-AI promotion"),
    (r"\boutreach\b|\bcold (e ?mails?|calls?|dms?|messages?)\b|\bmass (e ?mails?|dms?|messages?)\b|\bspam",
     "autonomous outreach"),
    (r"\bauto ?(post|posting|publish|publishing|upload|uploading|send|sending|tweet|reply|share)\b"
     r"|\b(post|upload|publish|share)(ing|ed)? (it |them |this |these )?(to|on) (youtube|tiktok|instagram|"
     r"spotify|soundcloud|twitter|facebook|linkedin|reddit|social|the web)\b", "an outward-facing action"),
    (r"\bclick ?bait\b|\bengagement (bait|farm)|\bview farm", "engagement bait"),
)
_BANNED = tuple((re.compile(p), why) for p, why in BANNED)
GENERIC_CONSUMERS = {
    "anyone", "everyone", "anybody", "public", "the public", "general public", "internet", "the internet",
    "audience", "an audience", "the audience", "viewers", "listeners", "followers", "subscribers", "fans",
    "social media", "youtube", "tiktok", "instagram", "spotify", "soundcloud", "twitter", "x", "facebook",
    "the algorithm", "none", "n/a", "na", "tbd", "todo", "unknown", "nobody", "no one", "speculative", "stock",
    "general", "market", "the market", "customers", "clients", "someone", "somebody",
}
URL_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://|\bwww\.|\bmailto:|\b[a-z0-9-]+\.(com|net|org|io|ai|dev|app|co|us|uk|de|fr|"
    r"xyz|info|biz|me|tv|ly|gg|site|online|shop|link)\b")


class TemplateError(ValueError):
    """A template file that does not load. The message names the file and every problem."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-./:]+", " ", str(text).lower())).strip()


def banned_reason(text) -> str | None:
    """Which banned class `text` falls into, or None."""
    if not isinstance(text, str) or not text:
        return None
    t = _norm(text)
    for rx, why in _BANNED:
        if rx.search(t):
            return why
    return None


def has_url(text) -> bool:
    return isinstance(text, str) and bool(URL_RE.search(text))


def generic_consumer(name) -> bool:
    return not isinstance(name, str) or len(name.strip()) < 3 or _norm(name) in GENERIC_CONSUMERS


@dataclass
class Template:
    id: str
    kind: str
    family: str
    consumers: tuple
    revenue_line: str
    value_usd: float
    alpha: float
    beta: float
    per_week: int
    max_stock: int
    two_week_use: float
    review_minutes: float
    params: dict = field(default_factory=dict)  # name -> {"type", "min", "max", "allowed", "default"}
    model: str | None = None
    style_prefix: str = ""
    brief_max_chars: int = 600
    speculative: bool = False
    filler: bool = False
    commercial: bool = False
    private: bool = False
    hard_deadline: bool = False
    active: bool = False
    notes: str = ""
    source: str = ""

    @property
    def flags(self) -> dict:
        return {k: bool(getattr(self, k)) for k in FLAG_KEYS}

    @property
    def prior_p(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def consumer_match(self, name) -> str | None:
        """The template's spelling of a proposal's consumer (case and spacing ignored)."""
        if not isinstance(name, str):
            return None
        want = _norm(name)
        for c in self.consumers:
            if _norm(c) == want:
                return c
        return None

    def defaults(self) -> dict:
        return {k: s["default"] for k, s in self.params.items() if "default" in s}

    def describe(self) -> dict:
        """What the planner is shown about the template (no file paths)."""
        bounds = {}
        for k, s in self.params.items():
            bounds[k] = {x: s[x] for x in ("min", "max", "allowed", "default") if x in s}
        return {"id": self.id, "kind": self.kind, "consumers": list(self.consumers),
                "revenue_line": self.revenue_line, "value_usd": self.value_usd,
                "review_minutes": self.review_minutes, "brief_max_chars": self.brief_max_chars,
                "flags": [k for k in FLAG_KEYS if getattr(self, k)], "params": bounds,
                "notes": self.notes[:300]}


def _suggest(key: str, options) -> str:
    close = difflib.get_close_matches(key, list(options), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _param_spec(kind: str, name: str, spec, problems: list) -> dict | None:
    typ = PARAM_TYPES[kind].get(name)
    where = f"params.{name}"
    if typ is None:
        problems.append(f"{where} is not a parameter of {kind} (allowed: {', '.join(PARAM_TYPES[kind])})"
                        + _suggest(name, PARAM_TYPES[kind]))
        return None
    if not isinstance(spec, dict):
        problems.append(f"{where} must be an object like {{\"min\": .., \"max\": .., \"default\": ..}}")
        return None
    allowed_keys = {"number": ("min", "max", "default"), "int": ("min", "max", "default"),
                    "choice": ("allowed", "default"), "size": ("allowed", "default"),
                    "bool": ("allowed", "default")}[typ]
    for k in spec:
        if k not in allowed_keys:
            problems.append(f"{where}.{k} is not allowed for a {typ} parameter (use {', '.join(allowed_keys)})"
                            + _suggest(k, allowed_keys))
    out = {"type": typ}
    if typ in ("number", "int"):
        check = _is_int if typ == "int" else (lambda v: number(v) is not None)
        for k in ("min", "max"):
            if k not in spec:
                problems.append(f"{where}.{k} is required")
            elif not check(spec[k]):
                problems.append(f"{where}.{k} must be {'an integer' if typ == 'int' else 'a number'}, "
                                f"got {spec[k]!r}")
            else:
                out[k] = spec[k]
        if "min" in out and "max" in out:
            if out["min"] > out["max"]:
                problems.append(f"{where}: min {out['min']} is above max {out['max']}")
            hard = HARD_BOUNDS.get((kind, name))
            if hard and (out["min"] < hard[0] or out["max"] > hard[1]):
                problems.append(f"{where}: bounds must lie within {hard[0]:g}..{hard[1]:g} (what the engine accepts)")
        if "default" in spec:
            d = spec["default"]
            if not check(d) or ("min" in out and d < out["min"]) or ("max" in out and d > out["max"]):
                problems.append(f"{where}.default {d!r} is not within its own bounds")
            else:
                out["default"] = d
    elif typ in ("choice", "size"):
        allowed = spec.get("allowed")
        if not isinstance(allowed, list) or not allowed or not all(isinstance(a, str) and a for a in allowed):
            problems.append(f"{where}.allowed must be a non-empty list of strings")
        else:
            if typ == "size":
                for a in allowed:
                    m = SIZE_RE.match(a)
                    if not m or not (64 <= int(m.group(1)) <= 2048 and 64 <= int(m.group(2)) <= 2048):
                        problems.append(f"{where}.allowed: {a!r} is not WIDTHxHEIGHT with sides 64..2048")
            out["allowed"] = list(allowed)
        if "default" in spec:
            if spec["default"] not in (allowed or []):
                problems.append(f"{where}.default {spec['default']!r} is not in allowed")
            else:
                out["default"] = spec["default"]
    else:  # bool
        allowed = spec.get("allowed", [True, False])
        if not isinstance(allowed, list) or not allowed or not all(isinstance(a, bool) for a in allowed):
            problems.append(f"{where}.allowed must be a list of booleans")
            allowed = [True, False]
        out["allowed"] = list(allowed)
        if "default" in spec:
            if spec["default"] not in allowed:
                problems.append(f"{where}.default must be one of {allowed}")
            else:
                out["default"] = spec["default"]
    return out


def parse_template(data, source: str = "<template>") -> Template:
    """Validate one template object. Raises TemplateError listing every problem found."""
    problems: list = []
    if not isinstance(data, dict):
        raise TemplateError(f"{source}: a template must be a JSON object, got {type(data).__name__}")
    for k in data:
        if k in OUTWARD_KEYS:
            problems.append(f"{k!r}: templates cannot describe outward actions (posting, uploading, sending); "
                            f"Foreman only makes files for a human to review")
        elif k not in TOP_KEYS:
            problems.append(f"unknown key {k!r}" + _suggest(k, TOP_KEYS))
    for k in REQUIRED_KEYS:
        if k not in data:
            problems.append(f"missing required key {k!r}")
    tid = data.get("id")
    if "id" in data and (not isinstance(tid, str) or not ID_RE.match(tid)):
        problems.append(f"id must be 2-64 characters of a-z 0-9 . _ - (starting with a letter or digit), got {tid!r}")
    kind = data.get("kind")
    if "kind" in data and kind not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")
        kind = None
    family = data.get("family")
    if kind and "family" in data and family != KINDS[kind]:
        problems.append(f"family {family!r} does not match kind {kind!r} (expected {KINDS[kind]!r})")
    model = data.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        problems.append("model must be a catalog model name")
    raw_consumers = data.get("consumer")
    consumers = [raw_consumers] if isinstance(raw_consumers, str) else raw_consumers
    if "consumer" in data:
        if not isinstance(consumers, list) or not consumers or not all(isinstance(c, str) for c in consumers):
            problems.append("consumer must be a non-empty string or a list of them: who actually uses the output")
            consumers = []
        for c in consumers:
            if generic_consumer(c):
                problems.append(f"consumer {c!r} is not a named consumer (uncommissioned media for a generic "
                                f"audience is banned)")
    revenue = data.get("revenue_line")
    if "revenue_line" in data and (not isinstance(revenue, str) or not revenue.strip()):
        problems.append("revenue_line must be a non-empty string")
    value = data.get("value_usd")
    if "value_usd" in data and (number(value) is None or value < 0):
        problems.append(f"value_usd must be a number >= 0, got {value!r}")
    prior = data.get("prior")
    alpha = beta = None
    if "prior" in data:
        if not isinstance(prior, dict) or set(prior) - {"alpha", "beta"}:
            problems.append("prior must be {\"alpha\": a, \"beta\": b}")
        else:
            alpha, beta = number(prior.get("alpha")), number(prior.get("beta"))
            if alpha is None or beta is None or alpha <= 0 or beta <= 0:
                problems.append(f"prior.alpha and prior.beta must be numbers > 0, got {prior!r}")
    caps = data.get("caps")
    capv = {}
    if "caps" in data:
        if not isinstance(caps, dict):
            problems.append("caps must be an object {per_week, max_stock, two_week_use}")
        else:
            for k in caps:
                if k not in CAP_KEYS:
                    problems.append(f"caps.{k} is not a cap (caps are {', '.join(CAP_KEYS)})" + _suggest(k, CAP_KEYS))
            for k in CAP_KEYS:
                v = caps.get(k)
                ok = _is_int(v) if k != "two_week_use" else number(v) is not None
                if k not in caps:
                    problems.append(f"caps.{k} is required")
                elif not ok or v < 0:
                    problems.append(f"caps.{k} must be a{'n integer' if k != 'two_week_use' else ' number'} >= 0, "
                                    f"got {v!r}")
                else:
                    capv[k] = v
    params = {}
    raw_params = data.get("params", {})
    if not isinstance(raw_params, dict):
        problems.append("params must be an object")
    elif kind:
        for name, spec in raw_params.items():
            s = _param_spec(kind, name, spec, problems)
            if s is not None:
                params[name] = s
        for name in REQUIRED_PARAMS.get(kind, ()):
            if name not in raw_params:
                problems.append(f"params.{name} is required for {kind}")
            elif name in params and "default" not in params[name]:
                problems.append(f"params.{name}.default is required (used when a proposal does not set it)")
    prefix = data.get("style_prefix", "")
    if not isinstance(prefix, str) or len(prefix) > 500:
        problems.append("style_prefix must be a string of at most 500 characters")
        prefix = ""
    elif has_url(prefix):
        problems.append("style_prefix must not contain URLs")
    bmax = data.get("brief_max_chars", 600)
    if not _is_int(bmax) or not 20 <= bmax <= MAX_BRIEF:
        problems.append(f"brief_max_chars must be an integer in 20..{MAX_BRIEF}, got {bmax!r}")
        bmax = 600
    flags = data.get("flags", {})
    if not isinstance(flags, dict):
        problems.append("flags must be an object of booleans")
        flags = {}
    for k, v in flags.items():
        if k in OUTWARD_KEYS:
            problems.append(f"flags.{k}: templates cannot describe outward actions; Foreman never posts, uploads, "
                            f"publishes or sends")
        elif k not in FLAG_KEYS:
            problems.append(f"flags.{k} is not a flag (flags are {', '.join(FLAG_KEYS)})" + _suggest(k, FLAG_KEYS))
        elif not isinstance(v, bool):
            problems.append(f"flags.{k} must be true or false, got {v!r}")
    review = data.get("review_minutes")
    if "review_minutes" in data and (number(review) is None or review < 0 or review > 240):
        problems.append(f"review_minutes must be a number in 0..240, got {review!r}")
    active = data.get("active", False)
    if not isinstance(active, bool):
        problems.append(f"active must be true or false, got {active!r}")
    notes = data.get("notes", "")
    if not isinstance(notes, str):
        problems.append("notes must be a string")
        notes = ""
    for label, text in (("id", tid), ("revenue_line", revenue), ("notes", notes), ("style_prefix", prefix)) + \
            tuple(("consumer", c) for c in (consumers or []) if isinstance(c, str)):
        why = banned_reason(text) if isinstance(text, str) else None
        if why:
            problems.append(f"{label} describes {why}, which is banned (no template may exist for it)")
    if problems:
        raise TemplateError(f"{source}:\n  - " + "\n  - ".join(problems))
    return Template(
        id=tid, kind=kind, family=KINDS[kind], consumers=tuple(c.strip() for c in consumers),
        revenue_line=revenue.strip(), value_usd=float(value), alpha=float(alpha), beta=float(beta),
        per_week=int(capv["per_week"]), max_stock=int(capv["max_stock"]), two_week_use=float(capv["two_week_use"]),
        review_minutes=float(review), params=params, model=model, style_prefix=prefix, brief_max_chars=bmax,
        speculative=bool(flags.get("speculative", False)), filler=bool(flags.get("filler", False)),
        commercial=bool(flags.get("commercial", False)), private=bool(flags.get("private", False)),
        hard_deadline=bool(flags.get("hard_deadline", False)), active=active, notes=notes, source=source)


def load_template_file(path: str) -> Template:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as e:
        raise TemplateError(f"{path}: not valid JSON: {e}") from None
    except OSError as e:
        raise TemplateError(f"{path}: cannot read: {e}") from None
    return parse_template(data, path)


def load_templates(directory: str | None) -> tuple:
    """({id: Template}, [error message]) for every *.json in `directory`. A file that does not
    load is reported and skipped; it never takes the others down."""
    templates: dict = {}
    errors: list = []
    if not directory or not os.path.isdir(directory):
        return templates, errors
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name.startswith("."):
            continue
        path = os.path.join(directory, name)
        try:
            t = load_template_file(path)
        except TemplateError as e:
            errors.append(str(e))
            continue
        if t.id in templates:
            errors.append(f"{path}: duplicate template id {t.id!r} (already defined in {templates[t.id].source})")
            continue
        templates[t.id] = t
    return templates, errors


def dir_signature(directory: str | None) -> tuple:
    """Cheap change detection for a directory of small files (names, sizes, mtimes)."""
    if not directory or not os.path.isdir(directory):
        return ()
    out = []
    for name in sorted(os.listdir(directory)):
        try:
            st = os.stat(os.path.join(directory, name))
        except OSError:
            continue
        out.append((name, st.st_size, st.st_mtime_ns))
    return tuple(out)


def load_goals(path: str | None, limit: int = 4000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read(limit + 1)
    except OSError:
        return ""
    return text[:limit] + ("\n[... truncated]" if len(text) > limit else "")


def load_inbox(directory: str | None, max_items: int = 10, max_chars: int = 2000) -> list:
    """Newest inbound notes first: [{"source", "text"}]. Their content is untrusted data."""
    if not directory or not os.path.isdir(directory):
        return []
    entries = []
    for name in os.listdir(directory):
        if name.startswith(".") or not name.lower().endswith((".txt", ".md", ".json")):
            continue
        path = os.path.join(directory, name)
        try:
            st = os.stat(path)
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read(max_chars + 1)
        except OSError:
            continue
        entries.append((st.st_mtime, name, text[:max_chars]))
    entries.sort(key=lambda e: (-e[0], e[1]))
    return [{"source": f"inbox/{name}", "text": text} for _, name, text in entries[:max_items]]


def model_prompt(template: Template, brief: str) -> str:
    prefix = template.style_prefix or ""
    if not prefix:
        return brief
    return prefix + ("" if prefix[-1:].isspace() else " ") + brief


def build_request(template: Template, model: str, consumer: str, brief: str, params: dict) -> dict:
    """The gpud request for one validated proposal (params already checked against the template's
    bounds and merged with its defaults)."""
    from .media_requests import music_request, speech_request, video_request
    prompt = model_prompt(template, brief)
    if template.kind == "speak":
        req = speech_request(prompt, voice=params.get("voice"), speed=params.get("speed"), model=model,
                             seed=params.get("seed"))
    elif template.kind == "compose_music":
        req = music_request(prompt, float(params["seconds"]), seed=params.get("seed"), model=model,
                            instrumental=params.get("instrumental"), bpm=params.get("bpm"))
    else:
        w, h = (int(x) for x in str(params["size"]).split("x"))
        req = video_request(prompt, float(params["seconds"]), w, h, steps=params.get("steps"),
                            seed=params.get("seed"), model=model)
    req["payload"].update(goal=f"[{template.id}] for {consumer}: {brief[:160]}", template_id=template.id,
                          consumer=consumer)
    return req
