import asyncio
import json
import os
import re
import uuid
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.runtime import AgentRuntime

from openai import OpenAI
from core.mcp_client import MultiServerMCPClient
from core.config import AgentConfig
from core.skill_registry import SkillRegistry

class BaseAgent:
    def __init__(
        self,
        agent_name: str,
        extra_env: Optional[dict] = None,
        runtime: Optional["AgentRuntime"] = None,
    ):
        self.name = agent_name
        self.config = AgentConfig(agent_name)
        self.extra_env = extra_env or {}
        self.runtime = runtime
        self.client_ai = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
        self.require_confirmation = self.config.require_confirmation
        self.confirmation_keywords = [k.strip() for k in (self.config.confirmation_keywords or []) if k.strip()]
        self.confirmed = False
        # Create MCP client with server config directly from agent config
        self.mcp_client = MultiServerMCPClient({"mcpServers": self.config.mcp_servers}, extra_env=self.extra_env)
        self._session_started = False
        self._messages: list[dict] | None = None
        self._openai_tools_all: list[dict] | None = None
        self._current_call_chain: list[str] | None = None
        self._skill_registry: Optional[SkillRegistry] = None
        self._skills_enabled: list[str] = []
        self._skills_prompt: str = ""
        self._skills_loaded: set[str] = set()

    def _get_mcp_prompts(self) -> str:
        """
        从本地工具文件或配置中收集 MCP 专属提示词。
        """
        prompts = []
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for name, conf in self.config.mcp_servers.items():
            # 1. 优先检查 JSON 配置中是否有 explicit prompt
            if "prompt" in conf:
                prompts.append(f"### {name} 指南\n{conf['prompt']}")
                continue
            
            # 2. 检查是否为本地 python 工具并提取 MCP_PROMPT 常量
            if conf.get("command") == "python" and conf.get("args"):
                for arg in conf["args"]:
                    if arg.endswith(".py"):
                        file_path = os.path.join(project_root, arg)
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    match = re.search(r'MCP_PROMPT\s*=\s*("""|\'\'\')(.*?)\1', content, re.DOTALL)
                                    if match:
                                        prompts.append(f"### {name} 指南\n{match.group(2).strip()}")
                            except Exception:
                                pass
                        break
        
        if not prompts:
            return ""
            
        header = "\n\n" + "="*30 + "\n你已加载以下 MCP 工具服务器，请严格遵循其特定的指令和参数规范：\n"
        return header + "\n\n".join(prompts) + "\n" + "="*30

    def _get_global_prompt(self) -> str:
        config_dir = getattr(self.config, "config_dir", "agent_configs")
        path = os.path.join(config_dir, "global.json")
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            prompt = data.get("system_prompt") or data.get("prompt") or ""
            prompt = str(prompt).strip()
            if not prompt:
                return ""
            return "\n\n" + prompt
        except Exception:
            return ""

    def _init_skills(self) -> None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_skills = os.path.join(project_root, "skills")
        codex_home = os.getenv("CODEX_HOME", os.path.expanduser("~/.codex"))
        global_skills = os.path.join(codex_home, "skills")
        registry = SkillRegistry([project_skills, global_skills])
        available = registry.list_skills()
        if not available:
            self._skill_registry = registry
            self._skills_enabled = []
            self._skills_prompt = ""
            self._skills_loaded = set()
            return

        defaults = self.config.skills_default
        include = self.config.skills_include
        exclude = self.config.skills_exclude

        enabled = SkillRegistry.resolve_enabled(available, defaults, include, exclude)

        self._skill_registry = registry
        self._skills_enabled = enabled
        self._skills_prompt = registry.build_skills_prompt(enabled)
        self._skills_loaded = set()

    def _auto_attach_skills(self, user_input: str, messages: list[dict]) -> None:
        if not self._skill_registry or not self._skills_enabled:
            return
        if not user_input:
            return
        matches = self._skill_registry.match_skills(user_input, self._skills_enabled, max_matches=2)
        for name in matches:
            if name in self._skills_loaded:
                continue
            content = self._skill_registry.read_skill(name, include_references=False)
            messages.append({
                "role": "system",
                "content": f"[Auto Skill: {name}]\n{content}"
            })
            self._skills_loaded.add(name)

    def _construct_system_prompt(self) -> str:
        """Build final system prompt with global and MCP-specific rules."""
        base_prompt = self.config.system_prompt
        global_prompt = self._get_global_prompt()
        mcp_prompts = self._get_mcp_prompts()
        return base_prompt + global_prompt + self._skills_prompt + mcp_prompts

    def _discover_agent_names(self) -> list[str]:
        config_dir = getattr(self.config, "config_dir", "agent_configs")
        if not os.path.isdir(config_dir):
            return []
        names = []
        for entry in os.listdir(config_dir):
            if not entry.endswith(".json"):
                continue
            if entry == "groups.json":
                continue
            names.append(os.path.splitext(entry)[0])
        return sorted(dict.fromkeys(names))

    def _known_agent_names(self) -> list[str]:
        if self.runtime:
            names = self.runtime.list_agents()
            if names:
                return names
        names = self._discover_agent_names()
        if names:
            return names
        return ["manager", "developer", "clerk"]

    def _local_tools(self) -> list[dict]:
        agent_names = self._known_agent_names()
        return [
            {
                "type": "function",
                "function": {
                    "name": "use_skill",
                    "description": "Load a skill's instructions by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "include_references": {"type": "boolean"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": "List available skills and summaries.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delegate",
                    "description": "Delegate a task to another agent and wait for its completion (parallel runtime only).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_agent": {
                                "type": "string",
                                "enum": agent_names,
                            },
                            "message": {"type": "string"},
                            "container": {
                                "type": "string",
                                "description": "Container override is not supported in parallel mode.",
                            },
                        },
                        "required": ["to_agent", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "handoff",
                    "description": "Set the default active agent for console input (parallel runtime only).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_agent": {
                                "type": "string",
                                "enum": agent_names,
                            },
                            "message": {
                                "type": "string",
                                "description": "Context or instructions for the next agent.",
                            },
                            "container": {
                                "type": "string",
                                "description": "Container override is not supported in parallel mode.",
                            },
                        },
                        "required": ["to_agent", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notify",
                    "description": "Send a message to another agent without waiting for a response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_agent": {
                                "type": "string",
                                "enum": agent_names,
                            },
                            "message": {"type": "string"},
                            "container": {
                                "type": "string",
                                "description": "Optional container name override (ignored in parallel mode).",
                            },
                        },
                        "required": ["to_agent", "message"],
                    },
                },
            },
        ]

    def _get_tool_output(self, result) -> str:
        """Convert MCP tool result to string."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        
        output = ""
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'type') and content.type == 'text':
                    output += content.text
                elif hasattr(content, 'type') and content.type == 'image':
                    output += "[Image Content]"
                elif isinstance(content, dict) and content.get('type') == 'text':
                    output += content.get('text', '')
                else:
                    output += str(content)
        else:
            output = str(result)
        return output

    def _format_tool_error(self, e: Exception) -> str:
        message = str(e).strip()
        if message:
            return message
        err_type = type(e).__name__
        if getattr(e, "args", None):
            return f"{err_type}: {e.args}"
        return f"{err_type} (no message)"

    def _slugify_container(self, value: str) -> str:
        value = (value or "").strip().lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        return value.strip("-") or "project"

    def _resolve_docker_target(self, function_args: dict) -> str:
        container = (function_args.get("container_name") or "").strip()
        if container:
            return container
        project_slug = (function_args.get("project_slug") or "").strip()
        if project_slug:
            return f"mcp-env-{self._slugify_container(project_slug)}"
        if isinstance(self.extra_env, dict):
            env_container = (self.extra_env.get("MCP_CONTAINER_NAME") or "").strip()
            if env_container:
                return env_container
        env_container = (os.getenv("MCP_CONTAINER_NAME") or "").strip()
        return env_container

    def _log_execute_command(self, function_args: dict) -> None:
        cmd = function_args.get("command")
        if not cmd:
            return
        print(f"[{self.config.name}] Command: {cmd}")
        target = self._resolve_docker_target(function_args)
        if target:
            print(f"[{self.config.name}] Host: docker exec {target} sh -c {cmd!r}")
        else:
            print(f"[{self.config.name}] Host: docker exec <unknown-container> sh -c {cmd!r}")

    async def _handle_delegate(self, args: dict) -> str:
        to_agent = args.get("to_agent")
        message = args.get("message", "")
        container = args.get("container")

        if to_agent not in set(self._known_agent_names()):
            return "Error: invalid to_agent."
        if not isinstance(message, str) or not message.strip():
            return "Error: message is required."

        if not self.runtime:
            return "Error: delegate requires parallel runtime."
        if container:
            return "Error: container override is not supported in parallel mode."
        if to_agent == self.config.name:
            return "Error: delegate to self is not allowed."
        if self._current_call_chain and to_agent in self._current_call_chain:
            return f"Error: delegate to '{to_agent}' is blocked (ancestor in call chain)."
        return await self.runtime.call_async(
            to_agent,
            message.strip(),
            from_agent=self.config.name,
            call_chain=self._current_call_chain,
        )

    async def _handle_handoff(self, args: dict) -> Optional[str]:
        to_agent = args.get("to_agent")
        _ = args.get("message", "")
        _ = args.get("container")

        if to_agent not in set(self._known_agent_names()):
            raise ValueError("Error: invalid to_agent.")

        print(f"\n=== Handoff: {self.config.name} -> {to_agent} ===")
        if self.runtime:
            self.runtime.set_active(to_agent)
            return f"Active agent set to {to_agent}"
        return "Error: handoff requires parallel runtime."

    async def _handle_notify(self, args: dict) -> str:
        to_agent = args.get("to_agent")
        message = args.get("message", "")
        if to_agent not in set(self._known_agent_names()):
            return "Error: invalid to_agent."
        if not isinstance(message, str) or not message.strip():
            return "Error: message is required."
        if not self.runtime:
            return "Error: notify is only supported in parallel mode."
        ok = self.runtime.notify(to_agent, message.strip(), from_agent=self.config.name)
        if not ok:
            return f"Error: agent '{to_agent}' not found."
        return f"Notified {to_agent}."

    async def _handle_use_skill(self, args: dict) -> str:
        name = args.get("name")
        include_references = bool(args.get("include_references", False))
        if not isinstance(name, str) or not name.strip():
            return "Error: name is required."
        if not self._skill_registry:
            return "Error: skills are not available."
        if name not in self._skills_enabled:
            return f"Error: skill '{name}' is not enabled."
        content = self._skill_registry.read_skill(name, include_references=include_references)
        if not content.startswith("Error:"):
            self._skills_loaded.add(name)
        return content

    async def _handle_list_skills(self) -> str:
        if not self._skill_registry:
            return "No skills available."
        if not self._skills_enabled:
            return "No skills enabled."
        summaries = []
        for name in self._skills_enabled:
            summary = self._skill_registry.get_summary(name) or ""
            if summary:
                summaries.append(f"- {name}: {summary}")
            else:
                summaries.append(f"- {name}")
        return "\n".join(summaries)

    def _check_confirmation(self, text: str) -> bool:
        if not self.confirmation_keywords:
            return False
        lowered = (text or "").strip().lower()
        # Guard against explicit negations like "不确认/不用确认"
        negations = [
            "不确认", "不用确认", "无需确认", "先不确认", "暂不确认",
            "不开始", "不用开始", "先不开始", "暂不开始", "不要开始",
            "不同意", "暂不同意"
        ]
        if any(neg in lowered for neg in negations):
            return False
        for kw in self.confirmation_keywords:
            if kw.lower() in lowered:
                return True
        return False

    def _extract_tool_json(self, content: str) -> Optional[dict]:
        if not content:
            return None
        stripped = content.strip()

        # 1. Try fenced code blocks first (standard markdown)
        fenced = re.search(r"```json\s*([\s\S]*?)\s*```", stripped, re.DOTALL)
        if not fenced:
            # Try generic code block if it looks like JSON
            fenced = re.search(r"```\s*(\{[\s\S]*?\})\s*```", stripped, re.DOTALL)
        
        if fenced:
            candidate = fenced.group(1).strip()
            try:
                obj = json.loads(self._repair_json(candidate))
                if isinstance(obj, dict): return obj
            except json.JSONDecodeError:
                pass

        # 2. If the entire content is JSON (possibly with some junk at end like <|DSML|...>)
        # We try to find the first '{' and the last '}' that form a valid object
        # Using a regex that finds the outermost braces
        brace_match = re.search(r"(\{[\s\S]*\})", stripped)
        if brace_match:
            candidate = brace_match.group(1).strip()
            # Try to parse it. If it fails, maybe there's junk AFTER the last '}'
            # We'll try to find the last '}' that makes it valid JSON
            current_candidate = candidate
            while '}' in current_candidate:
                try:
                    obj = json.loads(self._repair_json(current_candidate))
                    if isinstance(obj, dict): return obj
                except json.JSONDecodeError:
                    # Remove everything after the last '}' and try again
                    last_brace_idx = current_candidate.rfind('}')
                    if last_brace_idx <= 0: break
                    current_candidate = current_candidate[:last_brace_idx+1].strip()
        
        return None

    def _extract_tool_yaml(self, content: str) -> Optional[dict]:
        if not content:
            return None
        stripped = content.strip()
        # Try fenced yaml blocks
        fenced = re.search(r"```yaml\s*(.*?)\s*```", stripped, re.DOTALL)
        if not fenced:
            fenced = re.search(r"```yml\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced:
            block = fenced.group(1)
        else:
            # Try to find a plain "tool: ..." block
            if "tool:" not in stripped and "action:" not in stripped:
                return None
            block = stripped

        lines = block.splitlines()
        data: dict = {}
        args: dict = {}
        in_args = False
        current_key = None
        current_lines = []

        def flush_multiline() -> None:
            nonlocal current_key, current_lines
            if current_key is not None:
                args[current_key] = "\n".join(current_lines).rstrip()
                current_key = None
                current_lines = []

        for line in lines:
            if not line.strip() or line.strip().startswith("#"):
                continue
            if not in_args:
                if line.strip().startswith("tool:"):
                    data["tool"] = line.split(":", 1)[1].strip().strip("'\"")
                elif line.strip().startswith("action:"):
                    data["action"] = line.split(":", 1)[1].strip().strip("'\"")
                elif line.strip().startswith("arguments:"):
                    in_args = True
                continue

            # arguments section
            stripped_line = line.strip()
            m = re.match(r"([A-Za-z0-9_]+):\\s*(.*)$", stripped_line)
            if m:
                flush_multiline()
                key, val = m.group(1), m.group(2)
                if val == "|":
                    current_key = key
                    current_lines = []
                else:
                    args[key] = val.strip().strip("'\"")
            else:
                if current_key is not None:
                    current_lines.append(stripped_line)

        flush_multiline()
        if args:
            data["arguments"] = args
        return data if data else None

    def _extract_tool_dsml(self, content: str) -> Optional[dict]:
        """Extract tool calls from GLM's DSML XML-like format."""
        if not content or "<｜DSML｜invoke" not in content:
            return None
        
        # Simple regex-based extraction for the first invoke
        # <｜DSML｜invoke name="tool_name">
        invoke_match = re.search(r'<｜DSML｜invoke name="([^"]+)">', content)
        if not invoke_match:
            return None
        
        tool_name = invoke_match.group(1)
        args = {}
        
        # Find parameters: <｜DSML｜parameter name="key" ...>value</｜DSML｜parameter>
        params = re.findall(r'<｜DSML｜parameter name="([^"]+)"[^>]*>(.*?)</｜DSML｜parameter>', content, re.DOTALL)
        for name, value in params:
            # Try to parse value as JSON if it looks like it, otherwise use as string
            val = value.strip()
            if (val.startswith('{') and val.endswith('}')) or (val.startswith('[') and val.endswith(']')):
                try:
                    args[name] = json.loads(val)
                except:
                    args[name] = val
            else:
                args[name] = val
                
        return {"tool": tool_name, "arguments": args}

    def _extract_tool_call(self, content: str) -> Optional[dict]:
        # 1. Try DSML first (for GLM)
        dsml = self._extract_tool_dsml(content)
        if dsml:
            return dsml
            
        # 2. Try JSON
        obj = self._extract_tool_json(content)
        if obj:
            return obj
            
        # 3. Try YAML fallback
        return self._extract_tool_yaml(content)

    def _normalize_tool_call(self, obj: dict) -> Optional[tuple[str, dict]]:
        # Support {"action": "...", "args": {...}} or {"tool": "...", "arguments": {...}}
        if "action" in obj:
            name = obj.get("action")
            args = obj.get("args") or obj.get("arguments") or obj.get("input") or obj.get("params") or {}
        elif "tool" in obj:
            name = obj.get("tool")
            args = obj.get("arguments") or obj.get("args") or obj.get("input") or obj.get("params") or {}
        else:
            return None
        
        if not isinstance(name, str):
            return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            # If args is not a dict (e.g. a string for a simple command), wrap it
            if isinstance(args, (str, int, float, bool)):
                # This is a heuristic for tools that might take a single unnamed argument
                # But most MCP tools expect a dict. We'll leave it as is or return None.
                return None
            return None
        return name, args

    def _print_agent_response(self, content: str):
        """Print AI response, parsing JSON if present for better display."""
        if not content:
            return

        # Try to extract structured data using the same logic as tool extraction
        data = self._extract_tool_json(content)
        if data and isinstance(data, dict):
            # 1. Print Thoughts (labeled)
            thoughts = data.get("thoughts") or data.get("thought")
            if thoughts:
                print(f"\n[Thinking] {thoughts}")
            
            # 2. Print Message (the main response to user)
            message = data.get("message") or data.get("response") or data.get("answer")
            if message:
                print(f"\nAI: {message}")
            elif not data.get("tool") and not data.get("action"):
                # If no tool and no message, and it's a simple JSON, maybe it's just text in a field
                # If there's only one key and it's long, print it.
                if len(data) == 1:
                    val = list(data.values())[0]
                    if isinstance(val, str):
                        print(f"\nAI: {val}")
                        return
                
                # Otherwise fallback to raw if we can't find a clear message
                print(f"\nAI: {content}")
            
            # 3. Print Tool Info (if present)
            tool_name = data.get("tool") or data.get("action")
            if tool_name:
                print(f"\n[{self.name}] Planning to call: {tool_name}")
            
            return
        
        # Fallback for plain text or invalid JSON
        print(f"\nAI: {content}")

    def _repair_json(self, raw_json: str) -> str:
        """Attempt to repair common JSON errors from LLM output."""
        if not raw_json:
            return ""
        
        # 1. Fix unescaped newlines in string values
        def replace_newlines(match):
            content = match.group(0)
            # Only replace if there are actual newlines
            if '\n' in content or '\r' in content:
                return content.replace('\n', '\\n').replace('\r', '\\r')
            return content
        
        # Regex to find quoted strings, handling escaped quotes
        repaired = re.sub(r'"(?:\\.|[^"\\])*"', replace_newlines, raw_json, flags=re.DOTALL)
        
        # 2. Fix trailing commas before closing braces/brackets
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        
        return repaired

    async def _safe_chat_completion(self, messages: list, tools: list = None, retries: int = 3) -> Optional[object]:
        """Execute chat completion with retry logic and JSON output support."""
        import time
        response_format = None

        last_exception = None
        for attempt in range(retries):
            try:
                response = self.client_ai.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    tools=tools,
                    response_format=response_format
                )
                return response
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                is_conn_error = "connection" in error_str or "ssl" in error_str or "eof" in error_str
                
                if is_conn_error and attempt < retries - 1:
                    wait_time = 2 * (attempt + 1)
                    print(f"\n[Warning] API Connection error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        
        return None

    async def start_session(self) -> None:
        if self._session_started:
            return

        await self.mcp_client.connect()
        self._init_skills()
        openai_tools_all = await self.mcp_client.get_openai_tools()
        local_tools = self._local_tools()
        openai_tools_all.extend(local_tools)

        if not openai_tools_all:
            print("No tools loaded. Exiting.")
            raise RuntimeError("No tools loaded")

        self._openai_tools_all = openai_tools_all
        self._messages = [{"role": "system", "content": self._construct_system_prompt()}]
        self._session_started = True

        print(f"\n=== Agent '{self.config.name}' Started ===")
        print("Model:", self.config.model_name)
        print("Available tools:", ", ".join(self.mcp_client.tools_map.keys()))

    async def shutdown(self) -> None:
        await self.mcp_client.cleanup()
        self._session_started = False
        self._messages = None
        self._openai_tools_all = None

    async def process_user_message(self, user_input: str) -> str:
        if not self._session_started:
            await self.start_session()
        if not user_input or not user_input.strip():
            return ""

        if self.require_confirmation and not self.confirmed:
            if self._check_confirmation(user_input):
                self.confirmed = True

        messages = self._messages if self._messages is not None else []
        openai_tools_all = self._openai_tools_all or []
        self._auto_attach_skills(user_input, messages)
        messages.append({"role": "user", "content": user_input})

        while True:
            tools_for_turn = openai_tools_all
            if self.require_confirmation and not self.confirmed:
                safe_tools = {"sequentialthinking", "container_status", "read_file", "list_directory", "use_skill", "list_skills"}
                tools_for_turn = [
                    t for t in openai_tools_all
                    if t["function"]["name"] in safe_tools
                ]

            response = await self._safe_chat_completion(messages, tools=tools_for_turn)
            response_message = response.choices[0].message

            if not response_message.tool_calls:
                tool_json = self._extract_tool_call(response_message.content or "")
                normalized = self._normalize_tool_call(tool_json) if tool_json else None
                if normalized:
                    function_name, function_args = normalized
                    safe_tools = {"sequentialthinking", "container_status", "read_file", "list_directory", "use_skill", "list_skills"}
                    allowed_tools = set()
                    if (not self.require_confirmation or self.confirmed or function_name in safe_tools):
                        allowed_tools = set(self.mcp_client.tools_map.keys())
                        if not self.require_confirmation or self.confirmed:
                            local_tool_names = {t["function"]["name"] for t in self._local_tools()}
                            allowed_tools.update(local_tool_names)

                    if function_name in allowed_tools:
                        self._print_agent_response(response_message.content)
                        messages.append({"role": "assistant", "content": response_message.content})

                        tool_call_id = f"manual_{uuid.uuid4().hex}"
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [{
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args, ensure_ascii=False)
                                }
                            }]
                        })
                        print(f"\n[{self.config.name}] Tool Calls Detected (manual)")
                        print(f"[{self.config.name}] Calling: {function_name}")
                        if function_name == "execute_command" and isinstance(function_args, dict):
                            self._log_execute_command(function_args)
                        try:
                            if function_name == "delegate":
                                tool_output = await self._handle_delegate(function_args)
                            elif function_name == "handoff":
                                tool_output = await self._handle_handoff(function_args)
                                if tool_output is None:
                                    return ""
                            elif function_name == "notify":
                                tool_output = await self._handle_notify(function_args)
                            elif function_name == "use_skill":
                                tool_output = await self._handle_use_skill(function_args)
                            elif function_name == "list_skills":
                                tool_output = await self._handle_list_skills()
                            else:
                                result = await self.mcp_client.call_tool(function_name, function_args)
                                tool_output = self._get_tool_output(result)

                            display_output = (tool_output[:200] + '...') if len(tool_output) > 200 else tool_output
                            print(f"[Result] {display_output}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_output
                            })
                            continue
                        except Exception as e:
                            err_msg = self._format_tool_error(e)
                            print(f"[Error] {err_msg}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": f"Error: {err_msg}"
                            })
                            continue
                    else:
                        print(f"[{self.config.name}] Warning: Tool '{function_name}' is not recognized or not allowed in current state.")

                self._print_agent_response(response_message.content)
                messages.append(response_message)
                return response_message.content or ""

            if response_message.content:
                self._print_agent_response(response_message.content)

            messages.append(response_message)
            print(f"\n[{self.config.name}] Tool Calls Detected")

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(self._repair_json(tool_call.function.arguments))
                except json.JSONDecodeError as e:
                    print(f"[{self.config.name}] Error decoding JSON arguments for {function_name}: {e}")
                    print(f"Raw arguments: {tool_call.function.arguments}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Invalid JSON arguments provided. Please ensure arguments are valid JSON. Error: {str(e)}"
                    })
                    continue

                print(f"[{self.config.name}] Calling: {function_name}")
                if function_name == "execute_command" and isinstance(function_args, dict):
                    self._log_execute_command(function_args)

                try:
                    if function_name == "delegate":
                        tool_output = await self._handle_delegate(function_args)
                    elif function_name == "handoff":
                        tool_output = await self._handle_handoff(function_args)
                        if tool_output is None:
                            return ""
                    elif function_name == "notify":
                        tool_output = await self._handle_notify(function_args)
                    elif function_name == "use_skill":
                        tool_output = await self._handle_use_skill(function_args)
                    elif function_name == "list_skills":
                        tool_output = await self._handle_list_skills()
                    else:
                        result = await self.mcp_client.call_tool(function_name, function_args)
                        tool_output = self._get_tool_output(result)

                    display_output = (tool_output[:200] + '...') if len(tool_output) > 200 else tool_output
                    print(f"[Result] {display_output}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output
                    })

                except Exception as e:
                    err_msg = self._format_tool_error(e)
                    print(f"[Error] {err_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {err_msg}"
                    })

    async def run_queue(self, input_queue, stop_event=None) -> None:
        try:
            await self.start_session()
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                message = await asyncio.get_running_loop().run_in_executor(None, input_queue.get)
                if message is None:
                    continue
                kind = getattr(message, "kind", "user")
                if kind == "shutdown":
                    break

                content = getattr(message, "content", "")

                try:
                    if kind == "call":
                        self._current_call_chain = getattr(message, "call_chain", None)
                    else:
                        self._current_call_chain = None
                    result = await self.process_user_message(content)
                    if kind == "call" and self.runtime:
                        self.runtime.complete_call(getattr(message, "call_id", None), result or "")
                except Exception as e:
                    if kind == "call" and self.runtime:
                        self.runtime.complete_call(getattr(message, "call_id", None), f"Error: {e}")
                finally:
                    self._current_call_chain = None
        finally:
            await self.shutdown()
            print(f"\n[{self.name}] Connections closed.")

