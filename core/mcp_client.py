import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Dict, Any, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MultiServerMCPClient:
    def __init__(self, mcp_config: Dict[str, Any], extra_env: Optional[Dict[str, str]] = None):
        """
        Initialize the MCP Client with a configuration dictionary.
        
        Args:
            mcp_config: Dictionary containing 'mcpServers' configuration.
            extra_env: Extra environment variables to inject into all servers.
        """
        self.config = mcp_config
        self.extra_env = extra_env or {}
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.tools_map: Dict[str, Any] = {} # tool_name -> (session, tool_info, tools_timeout, server_name)

    async def connect(self):
        """Connect to all servers defined in the config."""
        for name, server_conf in self.config.get('mcpServers', {}).items():
            print(f"Connecting to server: {name}...")
            # Prepare env
            env = os.environ.copy()
            if 'env' in server_conf:
                for k, v in server_conf['env'].items():
                    # Expand ${VAR} from current environment
                    env[k] = os.path.expandvars(v)
            if self.extra_env:
                env.update(self.extra_env)
            
            # Handle command and args
            command = server_conf['command']
            args = server_conf['args']

            # Special handling for python command to use current interpreter
            if command == "python":
                command = sys.executable

            params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            
            try:
                connect_timeout = float(server_conf.get("connect_timeout", 10.0))
                init_timeout = float(server_conf.get("init_timeout", 5.0))
                tools_timeout = float(server_conf.get("tools_timeout", 5.0))

                # Connect with timeout
                read, write = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(stdio_client(params)),
                    timeout=connect_timeout
                )
                
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                self.sessions[name] = session
                
                # List tools
                tools = await asyncio.wait_for(session.list_tools(), timeout=tools_timeout)
                for tool in tools.tools:
                    self.tools_map[tool.name] = (session, tool, tools_timeout, name)
                    print(f"  - Loaded tool: {tool.name}")
            except asyncio.TimeoutError:
                print(f"  - Failed to connect to {name}: Connection timed out")
            except Exception as e:
                print(f"  - Failed to connect to {name}: {e}")

    async def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI tool format."""
        openai_tools = []
        for name, (session, tool, _, _) in self.tools_map.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        return openai_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool on the appropriate server."""
        if tool_name not in self.tools_map:
            raise ValueError(f"Tool {tool_name} not found")
        
        session, _, tools_timeout, server_name = self.tools_map[tool_name]
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=tools_timeout
            )
            return result
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError(
                f"Tool '{tool_name}' on server '{server_name}' timed out after {tools_timeout}s"
            ) from e

    async def cleanup(self):
        await self.exit_stack.aclose()
