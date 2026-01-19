import asyncio
import os
from typing import Dict

from dotenv import load_dotenv
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


# ---------------------------------------------------------------------
# ENV + CONFIG
# ---------------------------------------------------------------------

REQUIRED_ENV_VARS = [
    "MCP_API_TOKEN",
    "PHONE_NUMBER_ID",
    "WABATOKEN",
]

DEFAULT_WHATSAPP_API_VERSION = "v18.0"
MCP_SERVER_URL = "http://127.0.0.1:2001/mcp"


def load_and_validate_env() -> Dict[str, str]:
    """Load environment variables and validate required ones."""
    load_dotenv()

    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return {
        "api_token": os.getenv("MCP_API_TOKEN"),
        "phone_number_id": os.getenv("PHONE_NUMBER_ID"),
        "access_token": os.getenv("WABATOKEN"),
        "api_version": os.getenv(
            "WHATSAPP_API_VERSION", DEFAULT_WHATSAPP_API_VERSION
        ),
    }


# ---------------------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------------------

async def main() -> None:
    env = load_and_validate_env()

    print("Environment loaded successfully")
    print("MCP API Token:", env["api_token"][:6] + "****")
    print("ACCESS TOKEN:", env["access_token"][:6] + "****")
    print("Phone Number ID:", env["phone_number_id"])
    print("WhatsApp API Version:", env["api_version"])

    # MCP client configuration
    client = MultiServerMCPClient(
        {
            "whatsapp": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
                "headers": {
                    "Authorization": f"Bearer {env['api_token']}",
                    "x-whatsapp-phone-id": env["phone_number_id"],
                    "x-whatsapp-token": env["access_token"],
                    "x-whatsapp-api-version": env["api_version"],
                },
            }
        }
    )

    # Discover tools exposed by MCP server
    tools = await client.get_tools()
    print("Available MCP tools:")
    for tool in tools:
        print(f"  - {tool.name}")

    # Create ReAct-style agent
    agent = create_agent(
        model="gpt-4o-mini",  # cost-efficient + MCP compatible
        tools=tools,
    )

    # Invoke agent
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Send WhatsApp message to 919567578391 "
                        "saying 'Hello from MCP agent"
                    ),
                }
            ]
        }
    )

    print("\nAgent Response:")
    print(response)


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
