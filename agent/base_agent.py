import asyncio
import json
import os
import re
import sys
import traceback
import uuid
from typing import Optional

from openai import OpenAI
from core.mcp_client import MultiServerMCPClient
from core.config import AgentConfig

class AgentHandoff(Exception):
    """Signal to switch control to another agent."""
    def __init__(self, target_agent: str, message: str, container: Optional[str] = None):
        self.target_agent = target_agent
        self.message = message
        self.container = container

class BaseAgent:
    def __init__(self, agent_name: str, extra_env: Optional[dict] = None):
        self.name = agent_name
        self.config = AgentConfig(agent_name)
        self.extra_env = extra_env or {}
        self.client_ai = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url
        )
        self.require_confirmation = self.config.require_confirmation
        self.confirmation_keywords = [k.strip() for k in (self.config.confirmation_keywords or []) if k.strip()]
        self.confirmed = False
        # Create MCP client with server config directly from agent config
        self.mcp_client = MultiServerMCPClient({"mcpServers": self.config.mcp_servers}, extra_env=self.extra_env)

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

    def _construct_system_prompt(self) -> str:
        """构建最终的系统提示词，包含 Agent 基础提示词和 MCP 动态提示词。"""
        base_prompt = self.config.system_prompt
        mcp_prompts = self._get_mcp_prompts()
        return base_prompt + mcp_prompts

    def _local_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "delegate",
                    "description": "Delegate a task to another agent and wait for its completion. Use this when you need a sub-task done but intend to resume control afterwards.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_agent": {
                                "type": "string",
                            "enum": ["manager", "developer", "clerk"]
                            },
                            "message": {"type": "string"},
                            "container": {
                                "type": "string",
                                "description": "Optional container name override for the delegated agent."
                            }
                        },
                        "required": ["to_agent", "message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "handoff",
                    "description": "Transfer control to another agent completely. Use this when your current phase is done and the next agent should take over the interaction with the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_agent": {
                                "type": "string",
                            "enum": ["manager", "developer", "clerk"]
                            },
                            "message": {
                                "type": "string",
                                "description": "Context or instructions for the next agent."
                            },
                            "container": {
                                "type": "string",
                                "description": "Optional container name override for the next agent."
                            }
                        },
                        "required": ["to_agent", "message"]
                    }
                }
            }
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

        if to_agent not in {"manager", "developer", "clerk"}:
            return "Error: invalid to_agent. Must be 'manager', 'developer', or 'clerk'."
        if not isinstance(message, str) or not message.strip():
            return "Error: message is required."

        env = dict(self.extra_env)
        if container:
            env["MCP_CONTAINER_NAME"] = container

        print(f"\n=== Delegate: {self.config.name} -> {to_agent} ===")
        print(f"Message: {message}\n")
        
        target = BaseAgent(to_agent, extra_env=env)
        result = await target.run_once(message.strip())
        
        print(f"\n=== Delegate Finished: {to_agent} -> {self.config.name} ===")
        print(f"Response: {result}\n")
        
        return result or "[Delegate completed with no response]"

    async def _handle_handoff(self, args: dict) -> None:
        to_agent = args.get("to_agent")
        message = args.get("message", "")
        container = args.get("container")
        
        if to_agent not in {"manager", "developer", "clerk"}:
            raise ValueError("Error: invalid to_agent. Must be 'manager', 'developer', or 'clerk'.")
            
        print(f"\n=== Handoff: {self.config.name} -> {to_agent} ===")
        raise AgentHandoff(to_agent, message, container)

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

    async def run_once(self, user_input: str) -> str:
        """Run a single user input to completion (tools + final response), then exit."""
        try:
            await self.mcp_client.connect()
            
            openai_tools_all = await self.mcp_client.get_openai_tools()
            local_tools = self._local_tools()
            openai_tools_all.extend(local_tools)

            if not openai_tools_all:
                print("No tools loaded. Exiting.")
                return ""
            
            messages = [
                {"role": "system", "content": self._construct_system_prompt()},
                {"role": "user", "content": user_input},
            ]

            print(f"\n=== Agent '{self.config.name}' Started ===")
            print("Model:", self.config.model_name)
            print("Available tools:", ", ".join(self.mcp_client.tools_map.keys()))

            if self.require_confirmation and self._check_confirmation(user_input):
                self.confirmed = True

            while True:
                tools_for_turn = openai_tools_all
                if self.require_confirmation and not self.confirmed:
                    safe_tools = {"sequentialthinking", "container_status", "read_file", "list_directory"}
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
                        safe_tools = {"sequentialthinking", "container_status", "read_file", "list_directory"}
                        if self.require_confirmation and not self.confirmed and function_name not in safe_tools:
                            self._print_agent_response(response_message.content)
                            messages.append(response_message)
                            return response_message.content or ""
                        # Allow only known tools
                        allowed_tools = set(self.mcp_client.tools_map.keys())
                        if not self.require_confirmation or self.confirmed:
                            if "delegate" in {t["function"]["name"] for t in self._local_tools()}:
                                allowed_tools.add("delegate")
                                allowed_tools.add("handoff")
                        
                        if function_name in allowed_tools:
                            # Synthesize a tool call
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
                                    await self._handle_handoff(function_args)
                                    return None # Should not reach here
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

                    self._print_agent_response(response_message.content)
                    messages.append(response_message)
                    return response_message.content or ""

                messages.append(response_message)
                print(f"\n[{self.config.name}] Tool Calls Detected")
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(self._repair_json(tool_call.function.arguments))
                    
                    print(f"[{self.config.name}] Calling: {function_name}")
                    if function_name == "execute_command" and isinstance(function_args, dict):
                        self._log_execute_command(function_args)
                    
                    try:
                        if function_name == "delegate":
                            tool_output = await self._handle_delegate(function_args)
                        elif function_name == "handoff":
                            await self._handle_handoff(function_args)
                            return None # Should not reach here
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
        finally:
            await self.mcp_client.cleanup()
            print(f"\n[{self.name}] Connections closed.")

    async def run(self, initial_message: Optional[str] = None) -> Optional[AgentHandoff]:
        try:
            await self.mcp_client.connect()
            
            openai_tools_all = await self.mcp_client.get_openai_tools()
            local_tools = self._local_tools()
            openai_tools_all.extend(local_tools)

            if not openai_tools_all:
                print("No tools loaded. Exiting.")
                return
            
            messages = [
                {"role": "system", "content": self._construct_system_prompt()}
            ]

            if initial_message:
                print(f"\n[System] Incoming handoff message: {initial_message}")
                # messages.append({"role": "user", "content": f"[Handoff Message] {initial_message}"})
                # Note: We rely on the loop below to add this as the first user message.

            print(f"\n=== Agent '{self.config.name}' Started ===")
            print("Model:", self.config.model_name)
            print("Available tools:", ", ".join(self.mcp_client.tools_map.keys()))
            print("Type 'quit' or 'exit' to stop.")
            
            pending_input = initial_message
            while True:
                try:
                    if pending_input:
                        user_input = pending_input
                        pending_input = None
                        print(f"\nUser (from handoff): {user_input}")
                    else:
                        prompt = "\nUser: "
                        if self.require_confirmation and not self.confirmed:
                            prompt = "\nUser: "
                        user_input = input(prompt)
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    if not user_input.strip():
                        continue

                    if self.require_confirmation and not self.confirmed:
                        if self._check_confirmation(user_input):
                            self.confirmed = True

                    messages.append({"role": "user", "content": user_input})

                    # Process interaction loop
                    while True:
                        tools_for_turn = openai_tools_all
                        if self.require_confirmation and not self.confirmed:
                            # Allow safe tools (thought, status check) even without confirmation
                            safe_tools = {"sequentialthinking", "container_status", "read_file", "list_directory"}
                            tools_for_turn = [
                                t for t in openai_tools_all 
                                if t["function"]["name"] in safe_tools
                            ]
                        
                        response = await self._safe_chat_completion(messages, tools=tools_for_turn)
                        response_message = response.choices[0].message
                        
                        # If no tool calls, attempt to recover from tool-call JSON
                        if not response_message.tool_calls:
                            tool_json = self._extract_tool_call(response_message.content or "")
                            normalized = self._normalize_tool_call(tool_json) if tool_json else None
                            if normalized:
                                function_name, function_args = normalized
                                safe_tools = {"sequentialthinking", "container_status", "read_file", "list_directory"}
                                allowed_tools = set()
                                if (not self.require_confirmation or self.confirmed or function_name in safe_tools):
                                    allowed_tools = set(self.mcp_client.tools_map.keys())
                                    if not self.require_confirmation or self.confirmed:
                                        if "delegate" in {t["function"]["name"] for t in self._local_tools()}:
                                            allowed_tools.add("delegate")
                                            allowed_tools.add("handoff")
                                
                                if function_name in allowed_tools:
                                    # Handle mixed content: print text content first if exists
                                    # Since we extracted a tool call from content, the content itself IS the tool call usually.
                                    # But sometimes it might be "Here is the code:\n```json...```".
                                    # We should check if there is meaningful text before the tool call block.
                                    # For simplicity, if we found a manual tool call, we assume the user might want to see the whole message if it's mixed.
                                    # However, standard behavior for tool use is usually to hide the raw JSON.
                                    # Let's try to split: Text Message -> Tool Call
                                    
                                    # 1. Add the original message as a text assistant message
                                    # If the content is purely the JSON block, we might skip this, but printing it is safer for context.
                                    self._print_agent_response(response_message.content)
                                    messages.append({"role": "assistant", "content": response_message.content})

                                    # 2. Add the synthesized tool call
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
                                            await self._handle_handoff(function_args)
                                            return None # Should not reach here
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
                                    except AgentHandoff:
                                        raise
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
                            break

                        # Handle tool calls
                        # If there is content AND tool calls (native), print content first
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
                                # Execute local tool or MCP tool
                                if function_name == "delegate":
                                    tool_output = await self._handle_delegate(function_args)
                                elif function_name == "handoff":
                                    await self._handle_handoff(function_args)
                                    return None # Should not reach here
                                else:
                                    result = await self.mcp_client.call_tool(function_name, function_args)
                                    tool_output = self._get_tool_output(result)
                                    
                                # Truncate for display
                                display_output = (tool_output[:200] + '...') if len(tool_output) > 200 else tool_output
                                print(f"[Result] {display_output}")

                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_output
                                })
                                
                            except AgentHandoff:
                                raise
                            except Exception as e:
                                err_msg = self._format_tool_error(e)
                                print(f"[Error] {err_msg}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"Error: {err_msg}"
                                })
                        
                        # Loop continues to send tool outputs back to LLM
                            
                except AgentHandoff as handoff:
                    # Log gracefully without error trace
                    print(f"\n[System] Handoff signal received: -> {handoff.target_agent}")
                    return handoff
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\nAn error occurred: {e}")
                    traceback.print_exc()

        finally:
            await self.mcp_client.cleanup()
            print(f"\n[{self.name}] Connections closed.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    agent = BaseAgent("developer")
    asyncio.run(agent.run())
