import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env from project root if present
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

class AgentConfig:
    def __init__(self, name: str, config_dir: str = "agent_configs"):
        self.name = name
        self.config_dir = config_dir
        self.config_data = self._load_config()
        self.global_data = self._load_global_config()

    def _non_empty(self, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        return value or None

    def _load_config(self) -> Dict[str, Any]:
        """Load agent-specific configuration from JSON file."""
        config_path = os.path.join(self.config_dir, f"{self.name}.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config for agent '{self.name}' not found at {config_path}")
        
        # Use utf-8-sig to tolerate BOM added by some Windows editors.
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

    def _load_global_config(self) -> Dict[str, Any]:
        global_path = os.path.join(self.config_dir, "global.json")
        if not os.path.exists(global_path):
            return {}
        try:
            with open(global_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    @property
    def system_prompt(self) -> str:
        return self.config_data.get("system_prompt", "")

    @property
    def model_name(self) -> str:
        # Priority:
        # 1. Config file "model" field (non-empty)
        # 2. {AGENT_NAME}_MODEL (e.g., MANAGER_MODEL)
        # 3. LLM_MODEL (Global default)
        config_model = self._non_empty(self.config_data.get("model", ""))
        if config_model:
            return config_model

        agent_env_model = self._non_empty(os.getenv(f"{self.name.upper()}_MODEL"))
        if agent_env_model:
            return agent_env_model

        global_env_model = self._non_empty(os.getenv("LLM_MODEL"))
        if global_env_model:
            return global_env_model

        return ""

    @property
    def mcp_servers(self) -> Dict[str, Any]:
        return self.config_data.get("mcpServers", {})

    @property
    def skills_default(self) -> list[str]:
        value = self.global_data.get("skills_default", [])
        if isinstance(value, list):
            return [str(v) for v in value]
        return []

    @property
    def skills_include(self) -> list[str]:
        value = self.config_data.get("skills_include", [])
        if isinstance(value, list):
            return [str(v) for v in value]
        return []

    @property
    def skills_exclude(self) -> list[str]:
        value = self.config_data.get("skills_exclude", [])
        if isinstance(value, list):
            return [str(v) for v in value]
        return []

    @property
    def require_confirmation(self) -> bool:
        return bool(self.config_data.get("require_confirmation", False))

    @property
    def confirmation_keywords(self) -> list[str]:
        return self.config_data.get("confirmation_keywords", [])

    @property
    def api_base_url(self) -> str:
        # Priority:
        # 1. Config file "api_base_url" field (non-empty)
        # 2. {AGENT_NAME}_API_BASE_URL
        # 3. OPENAI_BASE_URL
        config_url = self._non_empty(self.config_data.get("api_base_url", ""))
        if config_url:
            return config_url

        agent_env_url = self._non_empty(os.getenv(f"{self.name.upper()}_API_BASE_URL"))
        if agent_env_url:
            return agent_env_url

        global_env_url = self._non_empty(os.getenv("OPENAI_BASE_URL"))
        if global_env_url:
            return global_env_url

        return ""

    @property
    def api_key(self) -> str:
        # Priority:
        # 1. Config file "api_key" field (non-empty)
        # 2. {AGENT_NAME}_API_KEY
        # 3. OPENAI_API_KEY
        config_key = self._non_empty(self.config_data.get("api_key", ""))
        if config_key:
            return config_key

        agent_env_key = self._non_empty(os.getenv(f"{self.name.upper()}_API_KEY"))
        if agent_env_key:
            return agent_env_key

        global_env_key = self._non_empty(os.getenv("OPENAI_API_KEY"))
        if global_env_key:
            return global_env_key

        return ""
