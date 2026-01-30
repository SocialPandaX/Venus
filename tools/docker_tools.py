"""
Docker Tools MCP Server
"""
import os
import sys
import asyncio
import subprocess
import tempfile
import shutil
import re
import time
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Docker Tools")

MCP_PROMPT = """
Docker Tools MCP Server:
- execute_command: 执行容器内的 shell 命令。
- write_file: 将文件写入容器。
  **重要：针对长文件（源代码、长文本），为避免 JSON 截断错误：**
  1. 优先考虑将文件拆分为更小的模块。
  2. 如果必须写入长文件，请分批次调用：第一次使用 append=False 覆盖创建，后续调用使用 append=True 追加内容。
- read_file: 从容器中读取文件。
- list_directory: 列出容器内的目录。
- upload_file_to_container: 将宿主机文件上传至容器。
  **极其重要（针对 Playwright 截图）：**
  当使用 Playwright 截图后，截图保存在宿主机的临时目录。你必须从 Playwright 的返回结果中提取出那个**完整的、绝对的宿主机路径**（例如：C:\\Users\\...\\homepage.png），并将其作为 host_path 传递给此工具。
  严禁只传递文件名（如 "homepage.png"），否则会报错找不到文件。
使用时请务必提供 container_name 或 project_slug 以定位目标容器。
"""

_DEBUG_FLAG = os.getenv("DOCKER_TOOLS_DEBUG", "").strip().lower()
_DEBUG = _DEBUG_FLAG in ("1", "true", "yes", "on")

def _debug(msg: str) -> None:
    if _DEBUG:
        print(msg, file=sys.stderr, flush=True)

def _slugify(value: str) -> str:
    if not value: return "project"
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "project"

def _get_target(container_name: str, project_slug: str) -> str:
    if container_name:
        return container_name
    if project_slug:
        return f"mcp-env-{_slugify(project_slug)}"
    
    # Check for environment variable
    env_container = os.getenv("MCP_CONTAINER_NAME")
    if env_container:
        return env_container
        
    return ""

async def _exec_docker(args: List[str], timeout: int = 180) -> str:
    """The most basic and robust execution logic using asyncio."""
    docker_path = shutil.which("docker")
    if not docker_path:
        return "ERROR: 'docker' executable not found in system PATH."
    
    full_cmd = [docker_path] + args
    _debug(f"[docker_tools] exec: {' '.join(full_cmd)}")
    
    try:
        start = time.time()
        # Using asyncio.create_subprocess_exec is much more robust for MCP servers
        # especially on Windows to avoid blocking the stdio transport.
        # We explicitly set stdin to DEVNULL to avoid any inheritance issues.
        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL
        )
        
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            exit_code = process.returncode
            elapsed = time.time() - start
            _debug(f"[docker_tools] exit={exit_code} elapsed={elapsed:.2f}s")
            
            output = stdout.decode('utf-8', errors='replace') if stdout else ""
            
            if exit_code == 0:
                return output if output.strip() else f"SUCCESS (Exit Code 0)"
            else:
                return f"FAILED (Exit Code {exit_code})\nOutput:\n{output}"
                
        except asyncio.TimeoutError:
            try:
                process.kill()
            except:
                pass
            _debug(f"[docker_tools] timeout after {timeout}s")
            return f"ERROR: Command timed out after {timeout} seconds."
            
    except Exception as e:
        _debug(f"[docker_tools] exception: {type(e).__name__}: {e}")
        return f"EXCEPTION: {str(e)}"

@mcp.tool()
async def execute_command(command: str, container_name: str = "", project_slug: str = "") -> str:
    
    target = _get_target(container_name, project_slug)
    if not target:
        return "ERROR: No target container specified. Please provide 'container_name' or 'project_slug'."
    return await _exec_docker(["exec", target, "sh", "-c", command])

@mcp.tool()
async def write_file(path: str, content: str, append: bool = False, container_name: str = "", project_slug: str = "") -> str:
    """
    Write file to container.
    If append is True, appends to the existing file. Otherwise overwrites.
    """
    target = _get_target(container_name, project_slug)
    if not target:
        return "ERROR: No target container specified. Please provide 'container_name' or 'project_slug'."
    
    # Pre-create directory
    dir_path = os.path.dirname(path)
    if dir_path and dir_path != "/":
        await _exec_docker(["exec", target, "mkdir", "-p", dir_path])
    
    if append:
        # For appending, we use a temporary file and 'cat >>' in the container
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Upload temp file to a temporary location in container first
            remote_tmp = f"/tmp/mcp_append_{int(time.time())}"
            await _exec_docker(["cp", tmp_path, f"{target}:{remote_tmp}"])
            # Append in container
            res = await _exec_docker(["exec", target, "sh", "-c", f"cat {remote_tmp} >> {path} && rm {remote_tmp}"])
            if "SUCCESS" in res or not res.strip():
                return f"Successfully appended to {path} in {target}"
            return res
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Standard overwrite using docker cp
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            res = await _exec_docker(["cp", tmp_path, f"{target}:{path}"])
            if "SUCCESS" in res or not res.strip():
                return f"Successfully wrote to {path} in {target}"
            return res
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

@mcp.tool()
async def read_file(path: str, container_name: str = "", project_slug: str = "") -> str:
    """Read file from container."""
    target = _get_target(container_name, project_slug)
    if not target:
        return "ERROR: No target container specified. Please provide 'container_name' or 'project_slug'."
    return await _exec_docker(["exec", target, "cat", path])

@mcp.tool()
async def list_directory(path: str = ".", container_name: str = "", project_slug: str = "") -> str:
    """List directory in container."""
    target = _get_target(container_name, project_slug)
    if not target:
        return "ERROR: No target container specified. Please provide 'container_name' or 'project_slug'."
    return await _exec_docker(["exec", target, "ls", "-la", path])

@mcp.tool()
async def upload_file_to_container(host_path: str, container_path: str, container_name: str = "", project_slug: str = "") -> str:
    """
    Upload a file from host machine to the container. 
    Useful for moving host-generated files (like screenshots) into the workspace.
    """
    target = _get_target(container_name, project_slug)
    if not target:
        return "ERROR: No target container specified."
    
    if not os.path.exists(host_path):
        return f"ERROR: Host file not found at {host_path}"

    # Pre-create directory in container
    dir_path = os.path.dirname(container_path)
    if dir_path and dir_path != "/":
        await _exec_docker(["exec", target, "mkdir", "-p", dir_path])

    res = await _exec_docker(["cp", host_path, f"{target}:{container_path}"])
    if "SUCCESS" in res or not res.strip():
        return f"Successfully uploaded {host_path} to {target}:{container_path}"
    return res

if __name__ == "__main__":
    mcp.run()
