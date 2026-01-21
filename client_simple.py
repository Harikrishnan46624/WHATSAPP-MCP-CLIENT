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
                        "Send WhatsApp message to "
                        "saying 'Hello from MCP agent"
                    ),
                }
            ]
        }
    )

    print("\nAgent Response:")
    print(response)
    agent_data = extract_agent_data(response)

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

    print("\n✅ Status")
    print("  SUCCESS" if agent_data["success"] else "  FAILED")
    print("=" * 60)







def extract_agent_data(response: dict) -> dict:
    """
    Extract maximum meaningful, production-grade information
    from a LangChain ReAct agent response.
    """

    messages = response.get("messages", [])

    final_ai_message = None
    tool_call = None
    tool_result = None

    # Walk through messages in order
    for msg in messages:
        # Final AI response (natural language)
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            final_ai_message = msg

        # Tool call intent (from AI)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]

        # Tool execution result
        if msg.__class__.__name__ == "ToolMessage":
            tool_result = msg

    # ---------------------------
    # Token usage (from final AI message)
    # ---------------------------
    token_usage = {}
    if final_ai_message and final_ai_message.response_metadata:
        token_usage = final_ai_message.response_metadata.get("token_usage", {})

    # ---------------------------
    # WhatsApp structured output
    # ---------------------------
    whatsapp_data = {}
    if tool_result and tool_result.artifact:
        structured = tool_result.artifact.get("structured_content", {})
        whatsapp_data = {
            "messaging_product": structured.get("messaging_product"),
            "wa_id": structured.get("contacts", [{}])[0].get("wa_id"),
            "message_id": structured.get("messages", [{}])[0].get("id"),
        }

    return {
        # Agent reasoning outcome
        "final_message": final_ai_message.content if final_ai_message else None,
        "model": final_ai_message.response_metadata.get("model_name")
        if final_ai_message
        else None,

        # Tool usage
        "tool_used": tool_call.get("name") if tool_call else None,
        "tool_arguments": tool_call.get("args") if tool_call else None,

        # Domain result
        "whatsapp": whatsapp_data,

        # Cost
        "tokens": {
            "prompt": token_usage.get("prompt_tokens"),
            "completion": token_usage.get("completion_tokens"),
            "total": token_usage.get("total_tokens"),
        },

        # Success signal
        "success": bool(whatsapp_data.get("message_id")),
    }







# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())


# NEED TO BE CONTINOUS CONVERSATION AND SESSION MEMORY