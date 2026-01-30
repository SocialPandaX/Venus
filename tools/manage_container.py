"""
Container Manager MCP Server
This server handles the lifecycle of Docker containers used for development.
It supports creating, starting, stopping, and checking the status of containers.
"""
import json
import os
import re
import shutil
import subprocess
import time
import sys
from typing import List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Container Manager")

MCP_PROMPT = """
Container Manager MCP Server:
- ensure_container: 确保开发容器存在并正在运行。
  自动挂载的目录：
  - /workspace: 工作目录，对应宿主机的 workspace/ 文件夹
  - /template: 模板目录，对应宿主机的 template/ 文件夹（存放软著文档模板等）
  可选参数：
  - ports: 逗号分隔的端口映射列表，如 "3000:3000,8080:80"。
  - extra_volumes: 分号分隔的额外卷挂载列表，如 "/host/data:/data"。
- stop_container: 停止正在运行的容器。
- container_status: 检查容器的运行状态和是否存在。
该 Server 用于管理docker容器的生命周期。
"""

IMAGE_NAME = "mcr.microsoft.com/devcontainers/universal:latest"
# This script is in gemini_mcp_client/tools/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_HOST = os.path.join(BASE_DIR, "workspace")
WORKSPACE_CONTAINER = "/workspace"
TEMPLATE_CONTAINER = "/template"
TEMPLATE_HOST = os.path.join(BASE_DIR, "template")


def _slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "project"


def _resolve_name(name: str, project_slug: str) -> str:
    if name:
        return name
    if project_slug:
        return f"mcp-env-{_slugify(project_slug)}"
    return os.getenv("MCP_CONTAINER_NAME", "")


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_run(args: List[str], timeout: int = 15) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker"] + args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        cmd = "docker " + " ".join(args)
        return subprocess.CompletedProcess(["docker"] + args, 124, "", f"timeout after {timeout}s (cmd: {cmd})")


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _diagnose_docker_env() -> str:
    parts: list[str] = []
    for key in ("DOCKER_CONTEXT", "DOCKER_HOST", "DOCKER_CONFIG"):
        val = os.getenv(key)
        if val:
            parts.append(f"{key}={val}")
    if not parts:
        parts.append("DOCKER_CONTEXT/DOCKER_HOST/DOCKER_CONFIG not set")

    ctx_result = _docker_run(["context", "show"], timeout=15)
    if ctx_result.returncode == 0 and ctx_result.stdout.strip():
        parts.append(f"docker context show: {ctx_result.stdout.strip()}")
    elif ctx_result.returncode != 0:
        err = _first_nonempty(ctx_result.stderr, ctx_result.stdout)
        if err:
            parts.append(f"docker context show error: {err}")

    return "; ".join(parts)


def get_container_name() -> str:
    # Priority: CLI arg -> env -> fallback to empty
    if len(sys.argv) > 1 and sys.argv[1] not in ["stop"]:
        return sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[1] == "stop":
        return sys.argv[2]
    return os.getenv("MCP_CONTAINER_NAME", "")


def ensure_workspace() -> None:
    if not os.path.exists(WORKSPACE_HOST):
        os.makedirs(WORKSPACE_HOST, exist_ok=True)


def ensure_template() -> None:
    if not os.path.exists(TEMPLATE_HOST):
        os.makedirs(TEMPLATE_HOST, exist_ok=True)


def _get_container_state(container_name: str) -> Tuple[bool, bool]:
    """
    Returns (running, exists) using a single docker inspect call.
    """
    result = _docker_run(["inspect", container_name], timeout=10)
    if result.returncode != 0:
        return False, False
    
    try:
        data = json.loads(result.stdout)
        if not data:
            return False, False
        running = data[0].get("State", {}).get("Running", False)
        return running, True
    except (json.JSONDecodeError, IndexError):
        return False, False


def _start_container(container_name: str, ports: Optional[List[str]] = None, extra_volumes: Optional[List[str]] = None) -> str:
    if not _docker_available():
        return "Error: docker CLI not found. Please install Docker Desktop and ensure it is on PATH."

    ensure_workspace()
    ensure_template()

    running, exists = _get_container_state(container_name)

    if running:
        return f"Container {container_name} is already running."

    if exists:
        result = _docker_run(["start", container_name])
        if result.returncode != 0:
            return f"Error starting container {container_name}: {result.stderr.strip()}"
        return f"Started existing container {container_name}."

    run_args = [
        "run",
        "-d",
        "--name",
        container_name,
    ]
    
    if ports:
        for port in ports:
            run_args.extend(["-p", port])

    # 固定挂载 workspace 和 template
    run_args.extend([
        "-v",
        f"{WORKSPACE_HOST}:{WORKSPACE_CONTAINER}",
        "-v",
        f"{TEMPLATE_HOST}:{TEMPLATE_CONTAINER}",
    ])
    
    # 额外挂载的卷
    if extra_volumes:
        for vol in extra_volumes:
            run_args.extend(["-v", vol])

    run_args.extend([
        "-w",
        WORKSPACE_CONTAINER,
        IMAGE_NAME,
        "tail",
        "-f",
        "/dev/null",
    ])

    run_result = _docker_run(run_args, timeout=180)
    if run_result.returncode != 0:
        details = _first_nonempty(run_result.stderr, run_result.stdout)
        return f"Error creating container {container_name}: {details or 'unknown error'}"

    return f"Created and started new container {container_name} with workspace and template mounted."


def _stop_container(container_name: str) -> str:
    if not _docker_available():
        return "Error: docker CLI not found."
    _docker_run(["stop", container_name])
    return f"Stopped container {container_name} (if it was running)."


def _container_status(container_name: str) -> str:
    if not _docker_available():
        return f"name={container_name} running=False exists=False (docker not found)"
    running, exists = _get_container_state(container_name)
    return f"name={container_name} running={running} exists={exists}"


@mcp.tool()
def ensure_container(container_name: str = "", project_slug: str = "", ports: str = "", extra_volumes: str = "") -> str:
    """
    Ensure a Debian container exists and is running.
    'ports' is an optional comma-separated list of port mappings, e.g., "3000:3000,8080:80".
    'extra_volumes' is an optional semicolon-separated list of additional volume mounts, e.g., 
    "/host/path1:/container/path1;/host/path2:/container/path2".
    """
    port_list = [p.strip() for p in ports.split(",")] if ports.strip() else None
    volume_list = [v.strip() for v in extra_volumes.split(";")] if extra_volumes.strip() else None
    resolved = _resolve_name(container_name, project_slug)
    if not resolved:
        return "Error: No container name or project slug provided, and MCP_CONTAINER_NAME is not set."
    return _start_container(resolved, port_list, volume_list)


@mcp.tool()
def stop_container(container_name: str = "") -> str:
    """
    Stop a running container by name.
    """
    resolved = container_name or os.getenv("MCP_CONTAINER_NAME", "")
    if not resolved:
        return "Error: No container name provided and MCP_CONTAINER_NAME is not set."
    return _stop_container(resolved)


@mcp.tool()
def container_status(container_name: str = "", project_slug: str = "") -> str:
    """
    Return running/exists status for a container.
    """
    resolved = _resolve_name(container_name, project_slug)
    if not resolved:
        return "Error: No container name or project slug provided, and MCP_CONTAINER_NAME is not set."
    return _container_status(resolved)


if __name__ == "__main__":
    # If running under MCP (stdio pipe), start MCP server.
    # Otherwise, keep CLI behavior for manual use.
    if not sys.stdin.isatty() or "--mcp" in sys.argv:
        mcp.run()
    else:
        # CLI usage: python tools/manage_container.py <command> <name/slug>
        # e.g. python tools/manage_container.py status my-project
        if len(sys.argv) < 2:
            print("Usage: python manage_container.py <start|stop|status> [name/slug]")
            sys.exit(1)
            
        cmd = sys.argv[1].lower()
        target = sys.argv[2] if len(sys.argv) > 2 else os.getenv("MCP_CONTAINER_NAME", "")
        
        if not target and cmd in ["stop", "status", "start"]:
            print(f"Error: Command '{cmd}' requires a container name or MCP_CONTAINER_NAME environment variable.")
            sys.exit(1)
            
        if cmd == "stop":
            print(_stop_container(target))
        elif cmd == "status":
            print(_container_status(target))
        elif cmd == "start":
            print(_start_container(target))
        else:
            # Default to start if target provided without explicit command
            print(_start_container(cmd))
