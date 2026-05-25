"""
Dialogue Data Loader
File: dialogue_loader.py

Loads agent dialogue from generated_agent_dialogue_{name}.json files.

JSON Structure:
[
  {
    "setting": ["...scene description..."],
    "emotion": "Uncomfortable",
    "topic": ["Discussing hobbies"],       # can also be a plain string
    "location": "Social Media Live Session",
    "background": ".",
    "source": "seed_dialogue_0",
    "dialogue": [
      {
        "role": "Aria Hartley",
        "action": "(speaking)",            # "(speaking)" | "(thinking)"
        "content": "..."
      },
      ...
    ]
  },
  ...
]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class DialogueTurn:
    role: str
    action: str         # "(speaking)" or "(thinking)"
    content: str

    @property
    def is_speaking(self) -> bool:
        return "(speaking)" in self.action.lower()

    @property
    def is_thinking(self) -> bool:
        return "(thinking)" in self.action.lower()

    def __repr__(self) -> str:
        return f"[{self.role}] {self.action}: {self.content[:60]}..."


@dataclass
class DialogueScene:
    source: str
    emotion: str
    location: str
    background: str
    setting: list[str]
    topic: list[str]                    # normalised to list[str]
    dialogue: list[DialogueTurn] = field(default_factory=list)

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def speakers(self) -> list[str]:
        """Unique speaker roles in order of first appearance."""
        seen: dict[str, None] = {}
        for t in self.dialogue:
            seen.setdefault(t.role, None)
        return list(seen)

    def turns_by(self, role: str) -> list[DialogueTurn]:
        """All turns for a given speaker (case-insensitive)."""
        role_lower = role.lower()
        return [t for t in self.dialogue if t.role.lower() == role_lower]

    def spoken_turns(self) -> list[DialogueTurn]:
        """Only '(speaking)' turns — excludes inner monologue."""
        return [t for t in self.dialogue if t.is_speaking]

    def thinking_turns(self) -> list[DialogueTurn]:
        """Only '(thinking)' turns."""
        return [t for t in self.dialogue if t.is_thinking]

    def __repr__(self) -> str:
        return (
            f"DialogueScene(source={self.source!r}, emotion={self.emotion!r}, "
            f"location={self.location!r}, turns={len(self.dialogue)}, "
            f"speakers={self.speakers})"
        )


# ─── Loader ───────────────────────────────────────────────────────────────────

class DialogueLoader:
    """
    Load dialogue scenes from  generated_agent_dialogue_{name}.json.

    Parameters
    ----------
    data_dir : str | Path
        Directory that contains the JSON files.
        Defaults to the current working directory.
    """

    FILE_PATTERN = "generated_agent_dialogue_{name}.json"

    def __init__(self, data_dir: str | Path = ".") -> None:
        self.data_dir = Path(data_dir)

    # ── public API ────────────────────────────────────────────────────────────

    def load(
        self,
        name: str,
        *,
        emotion_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
        topic_filter: Optional[str] = None,
        speaker_filter: Optional[str] = None,
    ) -> list[DialogueScene]:
        """
        Load and optionally filter scenes for a given agent name.

        Parameters
        ----------
        name            : Agent identifier used in the filename, e.g. "Aria-Hartley".
        emotion_filter  : Keep only scenes whose emotion matches (case-insensitive).
        location_filter : Keep only scenes whose location contains the string.
        topic_filter    : Keep only scenes whose topic list contains the string.
        speaker_filter  : Keep only scenes that include a given speaker role.

        Returns
        -------
        List of DialogueScene objects.
        """
        path = self._resolve_path(name)
        raw: list[dict] = self._read_json(path)
        scenes = [self._parse_scene(item) for item in raw]

        if emotion_filter:
            scenes = [s for s in scenes if s.emotion.lower() == emotion_filter.lower()]
        if location_filter:
            scenes = [s for s in scenes if location_filter.lower() in s.location.lower()]
        if topic_filter:
            scenes = [s for s in scenes if any(topic_filter.lower() in t.lower() for t in s.topic)]
        if speaker_filter:
            scenes = [s for s in scenes if speaker_filter.lower() in [sp.lower() for sp in s.speakers]]

        return scenes

    def iter_turns(
        self,
        name: str,
        *,
        speaking_only: bool = False,
        thinking_only: bool = False,
        **filter_kwargs,
    ) -> Iterator[tuple[DialogueScene, DialogueTurn]]:
        """
        Iterate over (scene, turn) pairs for every dialogue turn.

        Yields
        ------
        (DialogueScene, DialogueTurn)
        """
        scenes = self.load(name, **filter_kwargs)
        for scene in scenes:
            for turn in scene.dialogue:
                if speaking_only and not turn.is_speaking:
                    continue
                if thinking_only and not turn.is_thinking:
                    continue
                yield scene, turn

    def available_names(self) -> list[str]:
        """Return agent names inferred from JSON files in data_dir."""
        prefix = "generated_agent_dialogue_"
        return [
            p.stem[len(prefix):]
            for p in self.data_dir.glob(f"{prefix}*.json")
        ]

    # ── internals ─────────────────────────────────────────────────────────────

    def _resolve_path(self, name: str) -> Path:
        filename = self.FILE_PATTERN.format(name=name)
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Dialogue file not found: {path}\n"
                f"Available names: {self.available_names() or '(none found)'}"
            )
        return path

    @staticmethod
    def _read_json(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array at top level, got {type(data).__name__}")
        return data

    @staticmethod
    def _parse_scene(raw: dict) -> DialogueScene:
        # Normalise 'topic' — may be str or list[str]
        topic_raw = raw.get("topic", [])
        topic: list[str] = [topic_raw] if isinstance(topic_raw, str) else list(topic_raw)

        # Normalise 'setting' — may be str or list[str]
        setting_raw = raw.get("setting", [])
        setting: list[str] = [setting_raw] if isinstance(setting_raw, str) else list(setting_raw)

        turns = [
            DialogueTurn(
                role=t.get("role", ""),
                action=t.get("action", ""),
                content=t.get("content", ""),
            )
            for t in raw.get("dialogue", [])
        ]

        return DialogueScene(
            source=raw.get("source", ""),
            emotion=raw.get("emotion", ""),
            location=raw.get("location", ""),
            background=raw.get("background", ""),
            setting=setting,
            topic=topic,
            dialogue=turns,
        )


# ─── Quick-start helper ───────────────────────────────────────────────────────

def load_dialogue(name: str, data_dir: str | Path = ".") -> list[DialogueScene]:
    """Convenience wrapper: load all scenes for *name* from *data_dir*."""
    return DialogueLoader(data_dir).load(name)
