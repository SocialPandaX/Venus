import argparse
import asyncio
import sys
import os
import re
import time

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.base_agent import BaseAgent, AgentHandoff

def _slugify(value: str) -> str:
    # Keep ASCII, replace non-alnum with hyphen, collapse repeats.
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "project"

async def session_loop(initial_agent: str, project: str | None, container: str | None):
    current_agent_name = initial_agent
    
    # Initialize env
    extra_env = None
    if container:
        extra_env = {"MCP_CONTAINER_NAME": container}
    elif project:
        container_name = f"mcp-env-{_slugify(project)}"
        extra_env = {"MCP_CONTAINER_NAME": container_name}
        
    next_message = None

    while True:
        print(f"\n\n════════════════════════════════════════════════════════════")
        print(f" ACTIVATING AGENT: {current_agent_name.upper()}")
        print(f"════════════════════════════════════════════════════════════")
        
        agent = BaseAgent(current_agent_name, extra_env=extra_env)
        try:
            # Run the agent interactively. It returns AgentHandoff if switching, or None if quitting.
            result = await agent.run(initial_message=next_message)
            
            if isinstance(result, AgentHandoff):
                # Perform the switch
                print(f"\n[System] Handoff from {current_agent_name} -> {result.target_agent}")
                print(f"[System] Message: {result.message}\n")
                
                # Small delay to ensure previous agent's cleanup logs are flushed 
                # and resources released before starting the next one.
                time.sleep(1.0)
                
                current_agent_name = result.target_agent
                next_message = result.message
                if result.container:
                    extra_env = {"MCP_CONTAINER_NAME": result.container}
                continue
            else:
                # Normal exit (user typed quit/exit)
                print("Session ended.")
                break
                
        except Exception as e:
            print(f"Critical Error in agent loop: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Launch an MCP Agent Session with Handoff Support")
    parser.add_argument("agent", nargs="?", default="manager", help="Name of the initial agent (default: manager)")
    parser.add_argument("--project", help="Project name (used to scope container name)")
    parser.add_argument("--container", help="Explicit container name override")
    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(session_loop(args.agent, args.project, args.container))
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
