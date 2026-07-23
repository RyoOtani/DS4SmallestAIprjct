
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class AgentTask:
    id: str
    title: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentRun:
    user_request: str
    tasks: List[AgentTask] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "created"
    iteration: int = 0
    max_iterations: int = 5
