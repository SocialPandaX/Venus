import json
import os
from typing import Dict, List


def _scan_agents(config_dir: str) -> List[str]:
    if not os.path.isdir(config_dir):
        return []
    names: List[str] = []
    for entry in os.listdir(config_dir):
        if not entry.endswith(".json"):
            continue
        if entry == "groups.json":
            continue
        names.append(os.path.splitext(entry)[0])
    return sorted(dict.fromkeys(names))


def load_group_config(config_dir: str) -> Dict[str, List[str]]:
    group_path = os.path.join(config_dir, "groups.json")
    if not os.path.exists(group_path):
        return {}
    with open(group_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    groups: Dict[str, List[str]] = {}
    for name, value in data.items():
        if isinstance(value, list):
            groups[name] = [str(v) for v in value]
    return groups


def resolve_agent_group(group_name: str, config_dir: str) -> List[str]:
    group_name = (group_name or "").strip()
    groups = load_group_config(config_dir)
    if group_name and group_name in groups and groups[group_name]:
        return groups[group_name]
    return _scan_agents(config_dir)

