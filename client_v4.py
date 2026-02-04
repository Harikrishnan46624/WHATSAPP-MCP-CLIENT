

import asyncio
import os
import uuid
import traceback
from datetime import datetime, timezone
from contextlib import AsyncExitStack
from typing import List, Dict

from utils.env_setup import ENABLE_AGENT_DEBUG
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from agent.agent_data_extraction import extract_agent_data
from langchain_mcp_adapters.tools import load_mcp_tools
from agent.agent_logging import print_agent_summary
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import BaseMessage


from utils.json_history import load, append, clear
from tool_mcp.mcp_servers import MCPServerSpec
from agent.callbacks import ToolLoggingCallback
from prompt_library.prompt import SYSTEM_MESSAGE






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

        self.history = load(self.thread_id, last_n=4)

    # -------------------------------------------------

    async def connect(self):

        try:

            tools = []

            for spec in self.servers:

                if spec.transport == "stdio":
                    tools.extend(await self._connect_stdio(spec))

                elif spec.transport == "http":
                    tools.extend(await self._connect_http(spec))

                else:
                    raise ValueError(
                        f"Unsupported transport: {spec.transport}"
                    )

            if not tools:
                raise RuntimeError("No MCP tools discovered")

            self.tools = tools

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                callbacks=[ToolLoggingCallback()],
                max_retries=3,
                timeout=30,
            )

            self.agent = create_agent(
                model=llm,
                tools=self.tools,
            )

            print("✅ MCP Connected. Tools loaded.")

        except Exception as e:

            print("❌ MCP Connection Failed")
            traceback.print_exc()
            raise e

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

        try:

            state = {
                "messages": (
                    [SYSTEM_MESSAGE] +
                    [(h["role"], h["content"]) for h in self.history] +
                    [("user", user_input)]
                )
            }

            print("###################################")
            print("State: ", state)

            result = await self.agent.ainvoke(state)

            if ENABLE_AGENT_DEBUG:
                agent_data = extract_agent_data(result)
                print_agent_summary(agent_data)

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

        except asyncio.TimeoutError:

            return "⚠️ Request timed out. Please try again."

        except Exception as e:

            print("❌ Agent Execution Error")
            traceback.print_exc()

            return (
                "Sorry, I encountered a technical issue. "
                "Please try again shortly."
            )

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

    print("\n🚀 MCP Client Ready (Debug =", ENABLE_AGENT_DEBUG, ")\n")

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
