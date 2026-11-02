"""Character bibles: the owner's recurring characters (a mascot, a human host), never in the repo.

    $FOREMAN_HOME/characters/<id>/bible.json
    $FOREMAN_HOME/characters/<id>/stills/*.png|jpg     approved stills, tagged by pose or scene
    $FOREMAN_HOME/characters/<id>/voice/reference.wav  the voice's reference recording (optional)

bible.json (unknown keys are an error, so a typo cannot silently drop a trait):

    {"id": "host-a", "name": "Host A", "kind": "human" | "mascot",
     "subject": "the host",                      how a prompt refers to the character
     "look": "plain description of the face, hair, build and outfit",
     "keep": ["traits every shot keeps", ...],
     "style": "style notes (lens, light, palette)",
     "voice": {"engine": "kokoro", "voice": "af_heart", "speed": 1.0[, "ref_wav": "voice/reference.wav"]}
            | {"engine": "qwen3_clone", "ref_wav": "voice/reference.wav", "language": "English"},
     "stills": [{"file": "stills/hero.png", "tags": ["hero"], "approved": true, "note": ""}, ...]}

Tags name what a still shows: hero (the reference the pipeline edits from and the QA compares
against), turnaround, expressions, talk (a front-facing frame for lip-sync), endcard, and any
other short scene name. The pipeline only ever uses approved stills; importing a folder records
its images unapproved unless the owner says otherwise (`characters import ... --approve`, or
`characters approve ID FILE...`). A voice of any engine may name a reference recording
(`ref_wav`): a Qwen3-TTS clone speaks from it, and a talking-take model that hears its voice from a
recording (the h3 chain) uses it whatever the engine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

from . import paths

ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
KNOWN_TAGS = ("hero", "turnaround", "expressions", "talk", "endcard")
IMAGE_EXT = (".png", ".jpg", ".jpeg")
VOICE_ENGINES = ("kokoro", "qwen3_clone")
KEYS = ("id", "name", "kind", "subject", "look", "keep", "style", "voice", "stills")
KOKORO_VOICE = re.compile(r"^[a-z]{2}_[a-z]{2,20}(,[a-z]{2}_[a-z]{2,20}){0,3}$")


class BibleError(ValueError):
    pass


def characters_dir(home: str) -> str:
    return os.path.join(os.path.expanduser(home), "characters")


class Bible:
    def __init__(self, root: str, data: dict):
        self.root = root
        self.data = data
        self.id = data["id"]

    @property
    def subject(self) -> str:
        return self.data.get("subject") or ("the mascot" if self.data.get("kind") == "mascot" else "the host")

    @property
    def voice(self) -> dict:
        return dict(self.data.get("voice") or {})

    def file(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def approved(self, tag: str | None = None) -> list:
        """Absolute paths of approved stills (with `tag`), in bible order."""
        out = []
        for s in self.data.get("stills") or []:
            if s.get("approved") is True and (tag is None or tag in s.get("tags", [])):
                out.append(self.file(s["file"]))
        return out

    def approved_tags(self) -> list:
        tags = []
        for s in self.data.get("stills") or []:
            if s.get("approved") is True:
                for t in s.get("tags", []):
                    if t not in tags:
                        tags.append(t)
        return tags

    def hero(self) -> str | None:
        found = self.approved("hero")
        return found[0] if found else None

    def ref_wav(self) -> str | None:
        """The Qwen3-TTS clone's reference (None for any other engine: Kokoro never reads it)."""
        v = self.voice
        return self.file(v["ref_wav"]) if v.get("engine") == "qwen3_clone" and v.get("ref_wav") else None

    def voice_ref(self) -> str | None:
        """The voice's reference recording, whatever the engine (None when the bible names none)."""
        v = self.voice
        return self.file(v["ref_wav"]) if v.get("ref_wav") else None

    def fingerprint(self) -> str:
        h = hashlib.sha256(json.dumps(self.data, sort_keys=True).encode())
        for p in self.approved():
            try:
                with open(p, "rb") as f:
                    h.update(hashlib.sha256(f.read()).digest())
            except OSError:
                h.update(b"missing")
        return h.hexdigest()[:16]

    def describe(self) -> dict:
        stills = self.data.get("stills") or []
        return {"id": self.id, "name": self.data.get("name"), "kind": self.data.get("kind"), "subject": self.subject,
                "look": self.data.get("look"), "keep": self.data.get("keep") or [], "style": self.data.get("style"),
                "voice": self.voice, "stills": len(stills), "approved": sum(1 for s in stills if s.get("approved")),
                "tags": self.approved_tags(), "hero": bool(self.hero())}


def _rel_file(root: str, rel, where: str, problems: list, must_exist: bool = True) -> None:
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
        problems.append(f"{where}: {rel!r} must be a path relative to the character's directory")
        return
    full = os.path.join(root, rel)
    if not paths.inside(full, root):
        problems.append(f"{where}: {rel!r} resolves outside the character's directory")
    elif must_exist and not os.path.isfile(full):
        problems.append(f"{where}: {rel!r} does not exist")


def validate(data, root: str) -> dict:
    """The bible as stored, or BibleError listing every problem."""
    problems = []
    if not isinstance(data, dict):
        raise BibleError("bible.json must be a JSON object")
    for k in data:
        if k not in KEYS:
            problems.append(f"unknown key {k!r} (keys: {', '.join(KEYS)})")
    cid = data.get("id")
    if not isinstance(cid, str) or not ID_RE.match(cid):
        problems.append(f"id must be 2-64 characters of a-z 0-9 . _ -, got {cid!r}")
    for k in ("name", "subject", "look", "style"):
        if data.get(k) is not None and (not isinstance(data[k], str) or len(data[k]) > 1500):
            problems.append(f"{k} must be a string of at most 1500 characters")
    if not isinstance(data.get("look"), str) or not data["look"].strip():
        problems.append("look is required: the plain description every prompt carries")
    if data.get("kind") not in (None, "human", "mascot"):
        problems.append("kind must be human or mascot")
    keep = data.get("keep", [])
    if not isinstance(keep, list) or not all(isinstance(x, str) and 0 < len(x) <= 200 for x in keep) or len(keep) > 20:
        problems.append("keep must be a list of at most 20 short strings")
    voice = data.get("voice")
    if not isinstance(voice, dict) or voice.get("engine") not in VOICE_ENGINES:
        problems.append(f"voice must be an object with engine one of {VOICE_ENGINES}")
    elif voice["engine"] == "kokoro":
        if not isinstance(voice.get("voice"), str) or not KOKORO_VOICE.match(voice["voice"]):
            problems.append("voice.voice must be a Kokoro voice id such as af_heart, or a blend 'am_onyx,bm_george'")
        sp = voice.get("speed", 1.0)
        if isinstance(sp, bool) or not isinstance(sp, (int, float)) or not 0.5 <= sp <= 2.0:
            problems.append("voice.speed must be a number in 0.5..2.0")
        if voice.get("ref_wav") is not None:  # the h3 chain's reference voice; Kokoro itself never reads it
            _rel_file(root, voice.get("ref_wav"), "voice.ref_wav", problems)
        extra = set(voice) - {"engine", "voice", "speed", "ref_wav"}
        if extra:
            problems.append(f"voice has unknown keys {sorted(extra)}")
    else:
        _rel_file(root, voice.get("ref_wav"), "voice.ref_wav", problems)
        if voice.get("language") is not None and (not isinstance(voice["language"], str) or len(voice["language"]) > 40):
            problems.append("voice.language must be a language name such as English")
        extra = set(voice) - {"engine", "ref_wav", "language"}
        if extra:
            problems.append(f"voice has unknown keys {sorted(extra)}")
    stills = data.get("stills", [])
    if not isinstance(stills, list):
        problems.append("stills must be a list")
        stills = []
    seen = set()
    for i, s in enumerate(stills):
        where = f"stills[{i}]"
        if not isinstance(s, dict) or set(s) - {"file", "tags", "approved", "note"}:
            problems.append(f"{where} must be {{file, tags, approved, note}}")
            continue
        _rel_file(root, s.get("file"), f"{where}.file", problems)
        if isinstance(s.get("file"), str) and not s["file"].lower().endswith(IMAGE_EXT):
            problems.append(f"{where}.file must be a PNG or JPEG")
        if s.get("file") in seen:
            problems.append(f"{where}.file {s.get('file')!r} is listed twice")
        seen.add(s.get("file"))
        tags = s.get("tags", [])
        if not isinstance(tags, list) or not tags or not all(isinstance(t, str) and TAG_RE.match(t) for t in tags):
            problems.append(f"{where}.tags must be a non-empty list of short tags (hero, turnaround, talk, ...)")
        if not isinstance(s.get("approved", False), bool):
            problems.append(f"{where}.approved must be true or false")
    if problems:
        raise BibleError("; ".join(problems))
    return data


def load_bible(home: str, cid: str, chars_dir: str | None = None) -> Bible:
    """The validated bible of character `cid` (under `chars_dir`, default $FOREMAN_HOME/characters)."""
    if not isinstance(cid, str) or not ID_RE.match(cid):
        raise BibleError(f"not a character id: {cid!r}")
    root = os.path.join(os.path.expanduser(chars_dir) if chars_dir else characters_dir(home), cid)
    path = os.path.join(root, "bible.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise BibleError(f"no character {cid!r} ({path}: {e.strerror or e})") from None
    except ValueError as e:
        raise BibleError(f"{path} is not valid JSON: {e}") from None
    data = validate(data, root)
    if data["id"] != cid:
        raise BibleError(f"{path}: id {data['id']!r} does not match its directory {cid!r}")
    return Bible(root, data)


def list_bibles(home: str) -> list:
    """[(id, Bible or None, error or None)] for every directory under characters/."""
    root = characters_dir(home)
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if name.startswith(".") or not os.path.isdir(os.path.join(root, name)):
            continue
        try:
            out.append((name, load_bible(home, name), None))
        except BibleError as e:
            out.append((name, None, str(e)))
    return out


def guess_tags(filename: str) -> list:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    words = re.split(r"[^a-z0-9]+", stem)
    tags = []
    for w in words:
        w = {"expression": "expressions", "turn": "turnaround", "talking": "talk", "end": "endcard",
             "endcard": "endcard"}.get(w, w)
        if w in KNOWN_TAGS and w not in tags:
            tags.append(w)
    if not tags:
        scene = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-")[:32] or "other"
        tags = [scene if TAG_RE.match(scene) else "other"]
    return tags


def import_folder(home: str, src: str, cid: str, name: str | None = None, kind: str | None = None,
                  look: str | None = None, subject: str | None = None, voice: str | None = None,
                  speed: float | None = None, ref_wav: str | None = None, language: str | None = None,
                  approve: bool = False, overwrite: bool = False) -> Bible:
    """Copy a folder of images (and its bible.json, if it has one) into characters/<cid>/ and write
    a validated bible. Images are tagged from their file names; nothing is approved unless
    `approve` (or the folder's own bible.json says so)."""
    if not ID_RE.match(cid or ""):
        raise BibleError(f"character id must be 2-64 characters of a-z 0-9 . _ -, got {cid!r}")
    src = os.path.expanduser(src)
    if not os.path.isdir(src):
        raise BibleError(f"{src} is not a directory")
    root = os.path.join(characters_dir(home), cid)
    if os.path.exists(os.path.join(root, "bible.json")) and not overwrite:
        raise BibleError(f"character {cid!r} exists already ({root}); pass --overwrite to replace its bible")
    base = {}
    own = os.path.join(src, "bible.json")
    if os.path.isfile(own):
        try:
            with open(own, encoding="utf-8") as f:
                base = json.load(f)
        except (OSError, ValueError) as e:
            raise BibleError(f"{own}: {e}") from None
        if not isinstance(base, dict):
            raise BibleError(f"{own} must be a JSON object")
    os.makedirs(os.path.join(root, "stills"), exist_ok=True)
    stills = []
    listed = {s.get("file"): s for s in base.get("stills") or [] if isinstance(s, dict)}
    for fn in sorted(os.listdir(src)):
        full = os.path.join(src, fn)
        if fn.startswith(".") or not fn.lower().endswith(IMAGE_EXT) or not os.path.isfile(full) or os.path.islink(full):
            continue
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", fn)
        shutil.copyfile(full, os.path.join(root, "stills", safe))
        prior = listed.get(fn) or listed.get(f"stills/{fn}") or {}
        stills.append({"file": f"stills/{safe}", "tags": prior.get("tags") or guess_tags(fn),
                       "approved": bool(approve or prior.get("approved") is True), "note": prior.get("note", "")})
    for sub in ("stills",):  # images already under stills/ in the source folder
        d = os.path.join(src, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                full = os.path.join(d, fn)
                if fn.startswith(".") or not fn.lower().endswith(IMAGE_EXT) or os.path.islink(full):
                    continue
                safe = re.sub(r"[^A-Za-z0-9._-]+", "-", fn)
                shutil.copyfile(full, os.path.join(root, "stills", safe))
                prior = listed.get(f"stills/{fn}") or listed.get(fn) or {}
                if any(s["file"] == f"stills/{safe}" for s in stills):
                    continue
                stills.append({"file": f"stills/{safe}", "tags": prior.get("tags") or guess_tags(fn),
                               "approved": bool(approve or prior.get("approved") is True), "note": prior.get("note", "")})
    if not stills:
        raise BibleError(f"{src} holds no PNG or JPEG stills")
    data = {k: base[k] for k in KEYS if k in base and k not in ("id", "stills")}
    data["id"] = cid
    for k, v in (("name", name), ("kind", kind), ("look", look), ("subject", subject)):
        if v is not None:
            data[k] = v
    data.setdefault("name", cid)
    if ref_wav:
        os.makedirs(os.path.join(root, "voice"), exist_ok=True)
        shutil.copyfile(os.path.expanduser(ref_wav), os.path.join(root, "voice", "reference.wav"))
        data["voice"] = {"engine": "qwen3_clone", "ref_wav": "voice/reference.wav", "language": language or "English"}
    elif voice:
        data["voice"] = {"engine": "kokoro", "voice": voice, "speed": float(speed or 1.0)}
    elif isinstance(base.get("voice"), dict) and (base["voice"].get("engine") == "qwen3_clone"
                                                  or base["voice"].get("ref_wav") is not None):
        rel = base["voice"].get("ref_wav")
        bad = []
        _rel_file(src, rel, "the folder's bible.json voice.ref_wav", bad)  # never a file outside the folder
        if bad:
            raise BibleError("; ".join(bad))
        ref = os.path.join(src, rel)
        if os.path.isfile(ref) and not os.path.islink(ref):
            os.makedirs(os.path.join(root, "voice"), exist_ok=True)
            shutil.copyfile(ref, os.path.join(root, "voice", "reference.wav"))
            data["voice"] = dict(base["voice"], ref_wav="voice/reference.wav")
    data.setdefault("voice", {"engine": "kokoro", "voice": "af_heart", "speed": 1.0})
    data["stills"] = stills
    data = validate(data, root)
    paths.atomic_json(os.path.join(root, "bible.json"), data)
    return Bible(root, data)


def approve(home: str, cid: str, files, tags=None, approved: bool = True) -> Bible:
    """Mark stills approved (or not), optionally replacing their tags."""
    b = load_bible(home, cid)
    names = {os.path.basename(f) for f in files}
    hit = 0
    for s in b.data["stills"]:
        if os.path.basename(s["file"]) in names or s["file"] in files:
            s["approved"] = bool(approved)
            if tags:
                s["tags"] = list(tags)
            hit += 1
    if not hit:
        raise BibleError(f"none of {sorted(names)} is a still of {cid!r}")
    validate(b.data, b.root)
    paths.atomic_json(os.path.join(b.root, "bible.json"), b.data)
    return b
