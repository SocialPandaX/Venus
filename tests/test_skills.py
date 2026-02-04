import os
import tempfile
import unittest

from core.skill_registry import SkillRegistry


class SkillRegistryTests(unittest.TestCase):
    def _make_skill(self, skills_dir: str, name: str, content: str, references: dict | None = None) -> None:
        skill_dir = os.path.join(skills_dir, name)
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(content)
        if references:
            ref_dir = os.path.join(skill_dir, "references")
            os.makedirs(ref_dir, exist_ok=True)
            for filename, text in references.items():
                with open(os.path.join(ref_dir, filename), "w", encoding="utf-8") as f:
                    f.write(text)

    def test_scan_and_summary_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = os.path.join(tmp, "skills")
            os.makedirs(skills_dir, exist_ok=True)
            self._make_skill(skills_dir, "foo", "# Foo Skill\nDetails...\n")

            registry = SkillRegistry([skills_dir])
            skills = registry.list_skills()
            self.assertIn("foo", skills)
            self.assertEqual(skills["foo"], "Foo Skill")

    def test_summary_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = os.path.join(tmp, "skills")
            os.makedirs(skills_dir, exist_ok=True)
            self._make_skill(skills_dir, "bar", "Summary: Bar summary\nMore...\n")

            registry = SkillRegistry([skills_dir])
            skills = registry.list_skills()
            self.assertEqual(skills["bar"], "Bar summary")

    def test_resolve_enabled(self):
        available = {"a": "A", "b": "B", "c": "C"}
        defaults = ["a", "b"]
        include = ["c"]
        exclude = ["b"]
        result = SkillRegistry.resolve_enabled(available, defaults, include, exclude)
        self.assertEqual(result, ["a", "c"])

    def test_priority_project_overrides_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = os.path.join(tmp, "project_skills")
            global_dir = os.path.join(tmp, "global_skills")
            os.makedirs(project_dir, exist_ok=True)
            os.makedirs(global_dir, exist_ok=True)
            self._make_skill(project_dir, "foo", "# Project Foo\n")
            self._make_skill(global_dir, "foo", "# Global Foo\n")

            registry = SkillRegistry([project_dir, global_dir])
            skills = registry.list_skills()
            self.assertEqual(skills["foo"], "Project Foo")
            content = registry.read_skill("foo")
            self.assertIn("Project Foo", content)
            self.assertNotIn("Global Foo", content)

    def test_match_skills_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = os.path.join(tmp, "skills")
            os.makedirs(skills_dir, exist_ok=True)
            self._make_skill(skills_dir, "demo", "# Demo\nKeywords: alpha, beta\n")

            registry = SkillRegistry([skills_dir])
            matches = registry.match_skills("need beta support", ["demo"])
            self.assertEqual(matches, ["demo"])

    def test_read_skill_with_references_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = os.path.join(tmp, "skills")
            os.makedirs(skills_dir, exist_ok=True)
            self._make_skill(
                skills_dir,
                "baz",
                "Baz Skill\n",
                references={"ref.txt": "line1\nline2\n"},
            )

            registry = SkillRegistry([skills_dir])
            content = registry.read_skill("baz", include_references=True, max_ref_lines=1)
            self.assertIn("Baz Skill", content)
            self.assertIn("line1", content)
            self.assertNotIn("line2", content)


if __name__ == "__main__":
    unittest.main()
