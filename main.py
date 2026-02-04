import argparse
import sys
import os
import re

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.runtime import AgentRuntime
from core.console_router import run_console_router
from core.group_config import resolve_agent_group


def _slugify(value: str) -> str:
    # Keep ASCII, replace non-alnum with hyphen, collapse repeats.
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "project"


def _build_extra_env(project: str | None, container: str | None) -> dict | None:
    if container:
        return {"MCP_CONTAINER_NAME": container}
    if project:
        container_name = f"mcp-env-{_slugify(project)}"
        return {"MCP_CONTAINER_NAME": container_name}
    return None


def main():
    parser = argparse.ArgumentParser(description="Launch the Venus Multi-Agent Runtime")
    parser.add_argument("--group", default="default", help="Agent group name")
    parser.add_argument("--active", default="manager", help="Default active agent")
    parser.add_argument("--project", help="Project name (used to scope container name)")
    parser.add_argument("--container", help="Explicit container name override")
    args = parser.parse_args()

    try:
        agent_names = resolve_agent_group(args.group, "agent_configs")
        if not agent_names:
            raise ValueError("No agents found for runtime.")
        extra_env = _build_extra_env(args.project, args.container) or {}
        runtime = AgentRuntime(agent_names, extra_env=extra_env)
        runtime.start()
        if args.active and args.active in agent_names:
            runtime.set_active(args.active)
        run_console_router(runtime)
        runtime.join()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        print(f"Critical Error: {e}")


if __name__ == "__main__":
    main()
