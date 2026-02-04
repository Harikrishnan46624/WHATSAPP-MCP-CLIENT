from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any


class ToolLoggingCallback(BaseCallbackHandler):

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        name = serialized.get("name", "unknown_tool")
        print(f"\n[TOOL START] {name}")
        print(f"INPUT: {input_str}")

    def on_tool_end(self, output: Any, **kwargs) -> None:
        print(f"[TOOL END] OUTPUT: {output}\n")
