"""Conversational agent, built on the Claude Agent SDK (not the `anthropic` API SDK).

Auth is whatever `claude login` set up on this machine -- no ANTHROPIC_API_KEY, no
.env secret. The SDK drives the local `claude` CLI as a subprocess and already runs
its own agentic tool-use loop, so this module does not implement one: it configures
ClaudeAgentOptions once and translates each message the SDK yields into an SSE
event for the frontend.

Each browser message is a fresh call to the one-shot `query()` function rather than
a shared, long-lived `ClaudeSDKClient` -- a client left mid-response by a dropped
HTTP connection (tab closed, request aborted) corrupts that client for every
future request, since the underlying CLI session is still expecting to be
drained. `query()` per turn sidesteps that; conversation continuity across turns
is recovered via `options.resume` = the previous turn's `ResultMessage.session_id`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
)

from .tools import ALL_TOOLS

SYSTEM_PROMPT = """You are Engine-Copilot, an assistant embedded in a mechanical engineering design tool.
You help an engineer analyse 2-D shapes for external flow around a bluff body
(drag minimisation at constant area), using a trained surrogate model.

Ground rules:
- When the user asks for a prediction or a Cd estimate, CALL the flow_predict tool.
  Never invent a numeric value yourself.
- After every surrogate call, state the expected error percentage from the tool
  result alongside the Cd value -- never report Cd alone.
- Speak like an engineer: "drag coefficient", "wake", "shape coefficients" -- not
  "the number went down".
- Keep responses to 2-4 sentences unless the user explicitly asks for more detail.
- Respond in the language the user writes in (French or English).

Known model quality (from RESULTS.md -- do not fabricate other numbers):
  U-Net + continuity loss: Cd mean error 1.03%, worst case 4.2%, ~700x faster than
  the LBM solver it was trained against.

The training range is roughly Re in [10, 150]. Warn the user if they push a
Reynolds number outside that range -- the surrogate is extrapolating.
"""

_MCP_SERVER = create_sdk_mcp_server(name="engine_copilot", tools=ALL_TOOLS)
_ALLOWED_TOOLS = [f"mcp__engine_copilot__{t.name}" for t in ALL_TOOLS]


def _options(resume: str | None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"engine_copilot": _MCP_SERVER},
        allowed_tools=_ALLOWED_TOOLS,
        # These tools are safe, deterministic wrappers around our own read-only
        # endpoints (no shell, no filesystem writes) -- auto-approve them rather
        # than blocking on an interactive permission prompt.
        permission_mode="bypassPermissions",
        max_turns=6,
        resume=resume,
    )


class AgentSession:
    """Single-user local session -- one resumable Claude Agent SDK conversation."""

    def __init__(self) -> None:
        self._session_id: str | None = None

    async def chat(self, message: str) -> AsyncIterator[dict]:
        options = _options(self._session_id)

        async for msg in query(prompt=message, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield {"type": "text", "delta": block.text}
                    elif isinstance(block, ToolUseBlock):
                        yield {"type": "tool_use_start", "id": block.id, "name": block.name, "input": block.input}
            elif isinstance(msg, UserMessage):
                for block in msg.content if isinstance(msg.content, list) else []:
                    if isinstance(block, ToolResultBlock):
                        yield {"type": "tool_use_end", "id": block.tool_use_id, "result": block.content, "is_error": bool(block.is_error)}
            elif isinstance(msg, ResultMessage):
                self._session_id = msg.session_id
                yield {"type": "end", "usage": msg.usage, "is_error": msg.is_error}


agent_session = AgentSession()
