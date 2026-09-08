"""Execution trace. This is what the judges actually grade — the audit log of
what the agent decided and ran. Keep it rich and honest."""
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    query: str = ""
    n_images: int = 0
    modalities: list = field(default_factory=list)
    task: str = ""
    tools: list = field(default_factory=list)          # ordered tool calls
    steps: list = field(default_factory=list)          # human-readable log lines
    confidence: float | None = None
    started: float = field(default_factory=time.time)

    def log(self, msg: str):
        dt = time.time() - self.started
        self.steps.append(f"[{dt:6.2f}s] {msg}")
        return self

    def tool(self, name: str, **params):
        self.tools.append({"tool": name, "params": params})
        self.log(f"→ tool: {name}  {params if params else ''}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "n_images": self.n_images,
            "modalities": self.modalities,
            "task": self.task,
            "tools": self.tools,
            "confidence": self.confidence,
            "elapsed_s": round(time.time() - self.started, 2),
            "steps": self.steps,
        }

    def pretty(self) -> str:
        d = self.to_dict()
        lines = [
            f"TASK        : {d['task']}",
            f"QUERY       : {d['query']}",
            f"IMAGES      : {d['n_images']}  modalities={d['modalities']}",
            f"TOOLS RUN   : {', '.join(t['tool'] for t in d['tools']) or '(none)'}",
            f"CONFIDENCE  : {d['confidence']}",
            f"ELAPSED     : {d['elapsed_s']}s",
            "",
            "EXECUTION LOG:",
        ]
        lines += ["  " + s for s in d["steps"]]
        return "\n".join(lines)
