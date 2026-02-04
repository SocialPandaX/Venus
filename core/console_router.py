import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConsoleCommand:
    action: str
    target: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


def parse_console_input(line: str, default_target: Optional[str]) -> ConsoleCommand:
    if not line:
        return ConsoleCommand(action="noop")

    stripped = line.strip()
    if not stripped:
        return ConsoleCommand(action="noop")

    if stripped.startswith("/quit"):
        return ConsoleCommand(action="quit")
    if stripped.startswith("/list"):
        return ConsoleCommand(action="list")
    if stripped.startswith("/active"):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            return ConsoleCommand(action="error", error="Missing agent name for /active")
        return ConsoleCommand(action="active", target=parts[1].strip())
    if stripped.startswith("/broadcast"):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            return ConsoleCommand(action="error", error="Missing message for /broadcast")
        return ConsoleCommand(action="broadcast", message=parts[1].strip())

    if stripped.startswith("@"):
        match = re.match(r"@([A-Za-z0-9_-]+)\s+(.*)", stripped)
        if not match:
            return ConsoleCommand(action="error", error="Invalid @agent syntax")
        return ConsoleCommand(action="send", target=match.group(1), message=match.group(2).strip())

    return ConsoleCommand(action="send", target=default_target, message=line)


def run_console_router(runtime) -> None:
    print("\nConsole commands: @agent <msg> | /broadcast <msg> | /active <agent> | /list | /quit\n")

    while True:
        try:
            line = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            runtime.stop()
            break

        command = parse_console_input(line, runtime.get_active())

        if command.action == "noop":
            continue
        if command.action == "error":
            print(f"[Console] {command.error}")
            continue
        if command.action == "quit":
            runtime.stop()
            break
        if command.action == "list":
            print(
                f"[Console] Agents: {', '.join(runtime.list_agents())} | Active: {runtime.get_active()}"
            )
            continue
        if command.action == "active":
            if not command.target:
                print("[Console] No agent specified.")
                continue
            if runtime.set_active(command.target):
                print(f"[Console] Active agent set to {command.target}")
            else:
                print(f"[Console] Unknown agent: {command.target}")
            continue
        if command.action == "broadcast":
            runtime.broadcast(command.message or "")
            continue
        if command.action == "send":
            if not command.target:
                print("[Console] No active agent. Use /active <agent>.")
                continue
            runtime.send_user_message(command.target, command.message or "")
            continue
