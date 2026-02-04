# Venus

基于 MCP 的多 Agent 运行平台，支持任务编排、工具调用与 Agent 协作，内置容器化开发与文档处理工具。

## Requirements
- Python >= 3.13，建议使用 `uv`
- 可选：Docker Desktop（容器化工作区）
- 可选：Node.js / `npx`（运行 Playwright、sequential-thinking、brave-search 等 MCP 服务）

## Quick Start
1. 安装依赖：`uv sync`
2. 配置环境：复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY` 等变量
3. 启动：`uv run main.py --group default --active manager`  
`--project` 会生成容器名 `mcp-env-<slug>`，也可用 `--container` 指定

## Console Commands
- 交互命令：
`@agent <message>`：发送给指定 Agent  
`/broadcast <message>`：广播给所有 Agent  
`/active <agent>`：设置默认交互 Agent  
`/list`：列出 Agent 与当前默认  
`/quit`：停止全部 Agent
- 分组配置：`agent_configs/groups.json`

## Configuration
- Agent 配置：`agent_configs/<agent>.json`
- 全局提示：`agent_configs/global.json`（对所有 Agent 生效）
- Skills 默认列表：`agent_configs/global.json` 的 `skills_default`
- Skills 增删：`agent_configs/<agent>.json` 的 `skills_include` / `skills_exclude`
- 优先级：`<agent>.json` > `{AGENT}_*` 环境变量 > 全局环境变量
- `mcpServers` 定义 MCP 服务（容器管理、Docker 工具、Docx 编辑、Playwright 等）
- 角色示例：`manager/developer/clerk` 仅为示例，可按需替换与扩展
- 测试用双人组：`alpha/beta`（见 `agent_configs/groups.json` 的 `duo`）

## Skills
- 目录结构：`skills/<name>/SKILL.md`（必需）
- 可选附加：`skills/<name>/references/` 与 `skills/<name>/assets/`
- 使用方式：调用 `use_skill` 加载技能内容（可选 `include_references=true`）
- 列表工具：调用 `list_skills` 查看可用技能与摘要
- 自动触发：输入包含技能名或 Keywords/Tags 时，会自动注入该技能内容
- 目录优先级：项目内 `skills/` 优先于全局目录 `~/.codex/skills`（可通过 `CODEX_HOME` 覆盖）

## Built-in MCP Tools
- `tools/manage_container.py`：创建/启动/停止容器，挂载 `workspace/` 与 `template/`
- `tools/docker_tools.py`：容器内命令执行与文件读写/上传
- `tools/docx_editor.py`：读取/生成/编辑 `.docx`

## Layout
- `agent/`：Agent 运行逻辑
- `core/`：配置加载与 MCP 客户端
- `agent_configs/`：Agent 配置
- `skills/`：技能包目录（每个技能一个子目录）
- `tools/`：MCP 工具服务
- `template/`：文档模板
- `workspace/`：任务工作区
- `main.py`：入口
