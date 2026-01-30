# Venus

基于 MCP 的多 Agent 运行平台，支持任务编排、工具调用与 Agent 交接，内置容器化开发与文档处理工具。

## Requirements
- Python >= 3.13，建议使用 `uv`
- 可选：Docker Desktop（容器化工作区）
- 可选：Node.js / `npx`（运行 Playwright、sequential-thinking、brave-search 等 MCP 服务）

## Quick Start
1. 安装依赖：`uv sync`
2. 配置环境：复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY` 等变量
3. 启动：`uv run main.py [agent] [--project <name>] [--container <name>]`
   - `--project` 会生成容器名 `mcp-env-<slug>`，也可用 `--container` 指定

## Configuration
- Agent 配置：`agent_configs/<agent>.json`
- 优先级：`<agent>.json` > `{AGENT}_*` 环境变量 > 全局环境变量
- `mcpServers` 定义 MCP 服务（容器管理、Docker 工具、Docx 编辑、Playwright 等）

## Built-in MCP Tools
- `tools/manage_container.py`：创建/启动/停止容器，挂载 `workspace/` 与 `template/`
- `tools/docker_tools.py`：容器内命令执行与文件读写/上传
- `tools/docx_editor.py`：读取/生成/编辑 `.docx`

## Layout
- `agent/`：Agent 运行逻辑
- `core/`：配置加载与 MCP 客户端
- `agent_configs/`：Agent 配置
- `tools/`：MCP 工具服务
- `template/`：文档模板
- `workspace/`：任务工作区
- `main.py`：入口
