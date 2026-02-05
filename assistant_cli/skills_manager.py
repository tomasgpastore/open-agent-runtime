from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "to",
    "use",
    "with",
    "when",
    "your",
}


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path
    skill_md_path: Path


@dataclass(frozen=True)
class Skill:
    metadata: SkillMetadata
    content: str


class SkillManager:
    def __init__(
        self,
        skill_dirs: Sequence[Path],
        max_per_turn: int = 3,
        max_chars: int = 8000,
    ) -> None:
        self._skill_dirs = [Path(path).expanduser() for path in skill_dirs]
        self._max_per_turn = max(1, int(max_per_turn))
        self._max_chars = max(1000, int(max_chars))
        self._skills: dict[str, SkillMetadata] = {}
        self._refresh_errors: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self._skills = {}
        self._refresh_errors = []

        for root in self._skill_dirs:
            root_path = root.expanduser()
            if not root_path.exists():
                continue
            for skill_md in root_path.rglob("SKILL.md"):
                if not skill_md.is_file():
                    continue
                try:
                    content = skill_md.read_text(encoding="utf-8")
                except OSError as exc:
                    self._refresh_errors.append(f"{skill_md}: {exc}")
                    continue

                metadata = self._parse_metadata(skill_md, content)
                if not metadata:
                    continue

                key = metadata.name.lower()
                if key in self._skills:
                    LOGGER.warning("Duplicate skill name '%s' at %s", metadata.name, skill_md)
                    continue
                self._skills[key] = metadata

    def list_skills(self) -> list[SkillMetadata]:
        return sorted(self._skills.values(), key=lambda item: item.name.lower())

    def list_skill_dirs(self) -> list[Path]:
        return [path.expanduser() for path in self._skill_dirs]

    def refresh_errors(self) -> list[str]:
        return list(self._refresh_errors)

    def available_skills_prompt(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "(none)"

        lines = ["<available_skills>"]
        for skill in skills:
            description = self._truncate(skill.description, 200)
            lines.append("  <skill>")
            lines.append(f"    <name>{skill.name}</name>")
            if description:
                lines.append(f"    <description>{description}</description>")
            lines.append(f"    <location>{skill.skill_md_path}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def activation_prompt(self, user_text: str) -> str | None:
        if not user_text:
            return None

        matched = self.match_skills(user_text)
        if not matched:
            return None

        total_chars = 0
        blocks: list[str] = []
        for meta in matched:
            content = self._load_skill_content(meta)
            if not content:
                continue
            if total_chars + len(content) > self._max_chars:
                break
            total_chars += len(content)
            blocks.append(
                "\n".join(
                    [
                        f"<skill name=\"{meta.name}\" path=\"{meta.skill_md_path}\">",
                        content.strip(),
                        "</skill>",
                    ]
                )
            )

        if not blocks:
            return None

        header = "Activated skills (full instructions):"
        return "\n\n".join([header, *blocks])

    def match_skills(self, user_text: str) -> list[SkillMetadata]:
        text = user_text.lower()
        tokens = self._tokenize(text)
        scored: list[tuple[int, SkillMetadata]] = []

        for meta in self._skills.values():
            name_lower = meta.name.lower()
            if f"${name_lower}" in text:
                scored.append((100, meta))
                continue
            if name_lower in text:
                scored.append((75, meta))
                continue

            name_tokens = self._tokenize(name_lower)
            desc_tokens = self._tokenize(meta.description)
            if not name_tokens and not desc_tokens:
                continue

            name_hits = len(tokens & name_tokens)
            desc_hits = len(tokens & desc_tokens)
            score = (name_hits * 2) + desc_hits
            if name_hits == 0 and desc_hits == 0:
                continue
            if score < 2:
                continue
            scored.append((score, meta))

        scored.sort(key=lambda item: (-item[0], item[1].name.lower()))
        return [item[1] for item in scored[: self._max_per_turn]]

    def get_skill(self, query: str) -> Skill | None:
        if not query:
            return None
        key = query.strip().lower()
        meta = self._skills.get(key)
        if meta is None:
            meta = self._match_by_path_or_folder(key)
        if meta is None:
            return None

        content = self._load_skill_content(meta)
        if not content:
            return None

        return Skill(metadata=meta, content=content)

    def _match_by_path_or_folder(self, key: str) -> SkillMetadata | None:
        for meta in self._skills.values():
            if meta.path.name.lower() == key:
                return meta
            if str(meta.path).lower() == key:
                return meta
            if str(meta.skill_md_path).lower() == key:
                return meta
        return None

    def _load_skill_content(self, meta: SkillMetadata) -> str | None:
        try:
            return meta.skill_md_path.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Failed to read SKILL.md for %s: %s", meta.name, exc)
            return None

    def _parse_metadata(self, skill_md: Path, content: str) -> SkillMetadata | None:
        frontmatter = self._extract_frontmatter(content)
        name = (frontmatter.get("name") or "").strip()
        description = (frontmatter.get("description") or "").strip()

        skill_dir = skill_md.parent
        if not name:
            name = skill_dir.name

        if not name:
            self._refresh_errors.append(f"{skill_md}: missing name")
            return None

        return SkillMetadata(
            name=name,
            description=description,
            path=skill_dir.resolve(),
            skill_md_path=skill_md.resolve(),
        )

    def _extract_frontmatter(self, content: str) -> dict[str, str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}

        frontmatter_lines: list[str] = []
        index = 1
        while index < len(lines):
            line = lines[index]
            if line.strip() == "---":
                break
            frontmatter_lines.append(line)
            index += 1
        else:
            return {}

        data: dict[str, str] = {}
        for line in frontmatter_lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                data[key] = value
        return data

    def _tokenize(self, text: str) -> set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token and token not in STOPWORDS
        }
        return tokens

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."


__all__ = ["Skill", "SkillManager", "SkillMetadata"]
