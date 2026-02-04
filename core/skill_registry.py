import os
from typing import Dict, List, Optional, Iterable, Tuple


class SkillRegistry:
    def __init__(self, skills_dirs: Iterable[str]):
        self.skills_dirs = [d for d in skills_dirs if d]
        self._skills: Dict[str, str] = {}
        self._paths: Dict[str, str] = {}
        self._keywords: Dict[str, List[str]] = {}
        self._scan()

    def _scan(self) -> None:
        self._skills.clear()
        self._paths.clear()
        self._keywords.clear()
        for skills_dir in self.skills_dirs:
            if not os.path.isdir(skills_dir):
                continue
            for entry in sorted(os.listdir(skills_dir)):
                if entry in self._paths:
                    continue
                skill_path = os.path.join(skills_dir, entry)
                if not os.path.isdir(skill_path):
                    continue
                skill_file = os.path.join(skill_path, "SKILL.md")
                if not os.path.exists(skill_file):
                    continue
                summary, keywords = self._read_metadata(skill_file)
                self._skills[entry] = summary
                self._paths[entry] = skill_path
                self._keywords[entry] = keywords

    def list_skills(self) -> Dict[str, str]:
        return dict(self._skills)

    @staticmethod
    def resolve_enabled(
        available: Dict[str, str],
        defaults: List[str],
        include: List[str],
        exclude: List[str],
    ) -> List[str]:
        if defaults:
            enabled = [name for name in defaults if name in available]
        else:
            enabled = list(available.keys())

        for name in include:
            if name in available and name not in enabled:
                enabled.append(name)

        blocked = set(exclude or [])
        return [name for name in enabled if name not in blocked]

    def get_summary(self, name: str) -> Optional[str]:
        return self._skills.get(name)

    def get_keywords(self, name: str) -> List[str]:
        return list(self._keywords.get(name, []))

    def _read_metadata(self, skill_file: str) -> Tuple[str, List[str]]:
        summary = ""
        keywords: List[str] = []
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("keywords:") or lower.startswith("tags:"):
                    raw = line.split(":", 1)[1].strip()
                    for part in raw.split(","):
                        kw = part.strip()
                        if kw:
                            keywords.append(kw)
                if not summary:
                    if line.startswith("#"):
                        summary = line.lstrip("#").strip()
                    elif lower.startswith("summary:"):
                        summary = line.split(":", 1)[1].strip()
                    else:
                        summary = line
        except Exception:
            return "", []
        return summary, keywords

    def build_skills_prompt(self, enabled: List[str]) -> str:
        if not enabled:
            return ""
        lines = ["\n\nAvailable Skills:"]
        for name in enabled:
            summary = self._skills.get(name, "")
            if summary:
                lines.append(f"- {name}: {summary}")
            else:
                lines.append(f"- {name}")
        lines.append("Auto-trigger: matching skill names or Keywords/Tags may be injected automatically.")
        lines.append("Usage: call `use_skill` to load details; set include_references=true if needed.")
        return "\n".join(lines)

    def match_skills(self, text: str, enabled: List[str], max_matches: int = 2) -> List[str]:
        if not text:
            return []
        lowered = text.lower()
        matches: List[str] = []
        for name in enabled:
            if name.lower() in lowered:
                matches.append(name)
            else:
                for kw in self._keywords.get(name, []):
                    if kw.lower() in lowered:
                        matches.append(name)
                        break
            if len(matches) >= max_matches:
                break
        return matches

    def read_skill(self, name: str, include_references: bool = False, max_ref_lines: int = 200) -> str:
        if name not in self._paths:
            return f"Error: skill '{name}' not found."
        skill_file = os.path.join(self._paths[name], "SKILL.md")
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:
            return f"Error: failed to read skill '{name}'."

        if not include_references:
            return content

        ref_dir = os.path.join(self._paths[name], "references")
        if not os.path.isdir(ref_dir):
            return content

        ref_chunks: List[str] = []
        for entry in sorted(os.listdir(ref_dir)):
            ref_path = os.path.join(ref_dir, entry)
            if not os.path.isfile(ref_path):
                continue
            try:
                with open(ref_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                snippet = "".join(lines[:max_ref_lines]).rstrip()
                ref_chunks.append(f"\n\n[Reference: {entry}]\n{snippet}")
            except Exception:
                continue

        if not ref_chunks:
            return content

        return content + "".join(ref_chunks)
