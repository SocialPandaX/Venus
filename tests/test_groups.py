import json
import os
import tempfile
import unittest

from core.group_config import resolve_agent_group


class GroupConfigTests(unittest.TestCase):
    def test_resolve_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "agent_configs")
            os.makedirs(config_dir, exist_ok=True)
            groups_path = os.path.join(config_dir, "groups.json")
            with open(groups_path, "w", encoding="utf-8") as f:
                json.dump({"default": ["a", "b"]}, f)

            result = resolve_agent_group("default", config_dir)
            self.assertEqual(result, ["a", "b"])

    def test_fallback_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "agent_configs")
            os.makedirs(config_dir, exist_ok=True)
            for name in ["alpha", "beta"]:
                with open(os.path.join(config_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                    f.write("{}")

            result = resolve_agent_group("missing", config_dir)
            self.assertEqual(result, ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
