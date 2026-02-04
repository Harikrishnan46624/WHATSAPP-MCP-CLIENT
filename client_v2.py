import asyncio
import os
import uuid
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from typing import List

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage

from agent.callbacks import ToolLoggingCallback
from utils.json_history import load, append, clear
from tool_mcp.mcp_servers import MCPServerSpec

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------

load_dotenv()

# Explicitly disable LangSmith (defensive)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# ---------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------


class MCPClient:

    def __init__(self, servers: List[MCPServerSpec], thread_id: str | None = None):
        self.thread_id = thread_id or str(uuid.uuid4())
        self.servers = servers

        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.tools = []
        self.agent = None

        self.history = load(self.thread_id, last_n=4)

    # -----------------------------------------------------------------
    async def connect(self) -> None:
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

    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    async def invoke(self, user_input: str) -> str:
        state = {
            "messages": [
                (h["role"], h["content"]) for h in self.history
            ] + [("user", user_input)]
        }

        result = await self.agent.ainvoke(state)
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

    # -----------------------------------------------------------------
    def _extract_text(self, result) -> str:
        if isinstance(result, dict) and "messages" in result:
            last = result["messages"][-1]
            if isinstance(last, BaseMessage):
                return last.content
            if isinstance(last, tuple):
                return last[1]
        return str(result)

    # -----------------------------------------------------------------
    async def reset(self):
        clear(self.thread_id)
        self.history = []

    async def close(self):
        await self.exit_stack.aclose()


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
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
                    "WHATSAPP_API_VERSION", DEFAULT_WHATSAPP_API_VERSION
                ),
            },
        )
    ]

    client = MCPClient(servers)
    await client.connect()

    print("\nMCP Client ready. Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting MCP client...")
                break

            response = await client.invoke(user_input)

            print("\nAssistant:")
            print(response)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        await client.close()



if __name__ == "__main__":
    asyncio.run(main())




