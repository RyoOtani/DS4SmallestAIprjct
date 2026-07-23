"""Agent Roles: Clean separation of concerns for autonomous coding."""
from .agents import (
    Task, Plan, CodeChange, Review, TestReport, Diagnosis,
    PlannerAgent, ArchitectAgent, CoderAgent,
    ReviewerAgent, TestRunnerAgent, DebuggerAgent,
    SecurityReviewerAgent,
)
from .orchestrator import RoleOrchestrator, AgentRunResult

__all__ = [
    # Types
    "Task", "Plan", "CodeChange", "Review", "TestReport", "Diagnosis",
    # Roles
    "PlannerAgent", "ArchitectAgent", "CoderAgent",
    "ReviewerAgent", "TestRunnerAgent", "DebuggerAgent",
    "SecurityReviewerAgent",
    # Orchestrator
    "RoleOrchestrator", "AgentRunResult",
]
