from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    transport: str  # "stdio" | "http"

    # stdio
    path: Optional[str] = None
    args: Optional[List[str]] = None

    # http
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
