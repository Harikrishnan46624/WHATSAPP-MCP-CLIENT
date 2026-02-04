

import asyncio
import os
import uuid
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from typing import List, Dict

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage


from utils.json_history import load, append, clear
from tool_mcp.mcp_servers import MCPServerSpec
from agent.callbacks import ToolLoggingCallback


# =====================================================
# ENV
# =====================================================

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

ENABLE_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "false").lower() == "true"


# =====================================================
# COST CONFIG
# =====================================================

MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    }
}


def normalize_model_name(model: str) -> str:

    if not model:
        return ""

    if model.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"

    return model


def calculate_llm_cost(model: str, prompt: int, completion: int) -> float:

    base = normalize_model_name(model)

    pricing = MODEL_PRICING.get(base)

    if not pricing:
        return 0.0

    return round(
        (prompt * pricing["input"]) +
        (completion * pricing["output"]),
        6
    )


# =====================================================
# AGENT TRACE EXTRACTOR
# =====================================================

def extract_agent_data(response: dict) -> dict:

    messages = response.get("messages", [])

    final_ai = None
    tool_call = None
    tool_result = None

    for msg in messages:

        # Final AI response
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            final_ai = msg

        # Tool call
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]

        # Tool result
        if msg.__class__.__name__ == "ToolMessage":
            tool_result = msg

    # Token usage
    token_usage = {}

    if final_ai and final_ai.response_metadata:
        token_usage = final_ai.response_metadata.get("token_usage", {})

    # WhatsApp structured output
    whatsapp_data = {}

    if tool_result and tool_result.artifact:
        structured = tool_result.artifact.get("structured_content", {})

        whatsapp_data = {
            "messaging_product": structured.get("messaging_product"),
            "wa_id": structured.get("contacts", [{}])[0].get("wa_id"),
            "message_id": structured.get("messages", [{}])[0].get("id"),
        }

    return {
        "final_message": final_ai.content if final_ai else None,

        "model": final_ai.response_metadata.get("model_name")
        if final_ai else None,

        "tool_used": tool_call.get("name") if tool_call else None,

        "tool_arguments": tool_call.get("args") if tool_call else None,

        "whatsapp": whatsapp_data,

        "tokens": {
            "prompt": token_usage.get("prompt_tokens"),
            "completion": token_usage.get("completion_tokens"),
            "total": token_usage.get("total_tokens"),
        },

        "success": bool(whatsapp_data.get("message_id")),
    }


# =====================================================
# AGENT LOGGER
# =====================================================

def print_agent_summary(agent_data: dict):

    llm_cost = calculate_llm_cost(
        agent_data["model"],
        agent_data["tokens"]["prompt"] or 0,
        agent_data["tokens"]["completion"] or 0,
    )

    print("\n" + "=" * 60)
    print("🤖 AGENT EXECUTION SUMMARY")
    print("=" * 60)

    print(f"Model              : {agent_data['model']}")
    print(f"Tool Used          : {agent_data['tool_used']}")
    print(f"Tool Arguments     : {agent_data['tool_arguments']}")

    print("\n📨 WhatsApp Result")
    print(f"  Phone (wa_id)    : {agent_data['whatsapp'].get('wa_id')}")
    print(f"  Message ID       : {agent_data['whatsapp'].get('message_id')}")
    print(f"  Product          : {agent_data['whatsapp'].get('messaging_product')}")

    print("\n💬 Final Agent Message")
    print(f"  {agent_data['final_message']}")

    print("\n💰 Token Usage")
    print(f"  Prompt Tokens    : {agent_data['tokens']['prompt']}")
    print(f"  Completion Tokens: {agent_data['tokens']['completion']}")
    print(f"  Total Tokens     : {agent_data['tokens']['total']}")

    print("\n💵 LLM Cost (USD)")
    print(f"  ${llm_cost}")

    print("\n✅ Status")
    print("  SUCCESS" if agent_data["success"] else "  FAILED")

    print("=" * 60)


# =====================================================
# MCP CLIENT
# =====================================================

class MCPClient:

    def __init__(
        self,
        servers: List[MCPServerSpec],
        thread_id: str | None = None
    ):

        self.thread_id = thread_id or str(uuid.uuid4())
        self.servers = servers

        self.exit_stack = AsyncExitStack()

        self.sessions: List[ClientSession] = []

        self.tools = []
        self.agent = None

        # Load recent history
        self.history = load(self.thread_id, last_n=4)

    # -------------------------------------------------

    async def connect(self):

        tools = []

        for spec in self.servers:

            if spec.transport == "stdio":
                tools.extend(await self._connect_stdio(spec))

            elif spec.transport == "http":
                tools.extend(await self._connect_http(spec))

            else:
                raise ValueError(f"Unsupported transport: {spec.transport}")

        if not tools:
            raise RuntimeError("No MCP tools discovered")

        self.tools = tools

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            callbacks=[ToolLoggingCallback()],
        )

        self.agent = create_agent(
            model=llm,
            tools=self.tools,
        )

    # -------------------------------------------------

    async def _connect_stdio(self, spec: MCPServerSpec):

        params = StdioServerParameters(
            command="python" if spec.path.endswith(".py") else "node",
            args=[spec.path] + (spec.args or []),
        )

        read, write = await self.exit_stack.enter_async_context(
            stdio_client(params)
        )

        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()

        self.sessions.append(session)

        tools = await load_mcp_tools(session)

        return tools or []

    # -------------------------------------------------

    async def _connect_http(self, spec: MCPServerSpec):

        client = MultiServerMCPClient(
            {
                spec.name: {
                    "transport": "streamable_http",
                    "url": spec.url,
                    "headers": spec.headers or {},
                }
            }
        )

        return await client.get_tools()

    # -------------------------------------------------

    async def invoke(self, user_input: str) -> str:

        state = {
            "messages": [
                (h["role"], h["content"]) for h in self.history
            ] + [("user", user_input)]
        }

        result = await self.agent.ainvoke(state)

        # ================= DEBUG TRACE =================
        if ENABLE_AGENT_DEBUG:
            agent_data = extract_agent_data(result)
            print_agent_summary(agent_data)
        # ===============================================

        message = self._extract_text(result)

        now = datetime.now(timezone.utc).isoformat()

        append(self.thread_id, [
            {"role": "user", "content": user_input, "timestamp": now},
            {"role": "assistant", "content": message, "timestamp": now},
        ])

        self.history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": message},
        ])

        return message

    # -------------------------------------------------

    def _extract_text(self, result) -> str:

        if isinstance(result, dict) and "messages" in result:

            last = result["messages"][-1]

            if isinstance(last, BaseMessage):
                return last.content

            if isinstance(last, tuple):
                return last[1]

        return str(result)

    # -------------------------------------------------

    async def reset(self):

        clear(self.thread_id)
        self.history = []

    async def close(self):

        await self.exit_stack.aclose()


# =====================================================
# ENTRYPOINT
# =====================================================

async def main():

    DEFAULT_WHATSAPP_API_VERSION = "v18.0"

    servers = [
        MCPServerSpec(
            name="whatsapp",
            transport="http",
            url="https://whatsapp-mcp-server-xqt4.onrender.com/mcp",
            headers={
                "Authorization": f"Bearer {os.getenv('MCP_API_TOKEN')}",
                "x-whatsapp-phone-id": os.getenv("PHONE_NUMBER_ID"),
                "x-whatsapp-token": os.getenv("WABATOKEN"),
                "api_version": os.getenv(
                    "WHATSAPP_API_VERSION",
                    DEFAULT_WHATSAPP_API_VERSION,
                ),
            },
        )
    ]

    client = MCPClient(servers)

    await client.connect()

    print("\nMCP Client Ready (Debug =", ENABLE_AGENT_DEBUG, ")\n")

    try:

        while True:

            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                break

            response = await client.invoke(user_input)

            print("\nAssistant:")
            print(response)
            print("-" * 50)

    finally:

        await client.close()


if __name__ == "__main__":

    asyncio.run(main())


