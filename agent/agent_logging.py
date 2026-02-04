# Agent logging module

from utils.cost_calculation import calculate_llm_cost

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