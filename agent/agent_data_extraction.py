# Agent data extraction module

def extract_agent_data(response: dict) -> dict:
    messages = response.get("messages", [])

    final_ai = None
    tool_call = None
    tool_result = None

    for msg in messages:
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            final_ai = msg

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]

        if msg.__class__.__name__ == "ToolMessage":
            tool_result = msg

    token_usage = {}

    if final_ai and final_ai.response_metadata:
        token_usage = final_ai.response_metadata.get("token_usage", {})

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
        "model": final_ai.response_metadata.get("model_name") if final_ai else None,
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