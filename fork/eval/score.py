"""Score answers against an evaluation file (fork/eval/*.json).

A held-out fact question counts as a hit when the fact's pattern appears in the
answer, or, for a "lead" fact, within its first 25 words. A control passes when
the answer mentions no fact and contains the control's expected pattern.

    from score import Eval
    ev = Eval.load("fork/eval/five-facts.json")
    ev.questions()          -> [(label, question), ...]  facts first, then controls
    ev.score(answers)       -> {"fact_hits", "by_fact", "controls_ok", ...}
                               answers: {label: answer text}
"""
import json
import re
from pathlib import Path


class Eval:
    def __init__(self, data: dict):
        self.data = data
        self.system_prompt = data["system_prompt"]
        self.max_output_tokens = data["max_output_tokens"]
        self.facts = data["facts"]
        self.controls = data["controls"]
        self._pattern = {f: re.compile(v["pattern"], re.I) for f, v in self.facts.items()}
        self._ignored = set(data.get("control_ignores_facts", []))

    @classmethod
    def load(cls, path) -> "Eval":
        return cls(json.loads(Path(path).read_text()))

    def questions(self):
        out = [(f"fact:{f}:{i}", q) for f, v in self.facts.items() for i, q in enumerate(v["held"])]
        out += [(f"ctrl:{i}", c["q"]) for i, c in enumerate(self.controls)]
        return out

    def hit(self, fact: str, answer: str) -> bool:
        text = " ".join(answer.split()[:25]) if self.facts[fact]["lead"] else answer
        return bool(self._pattern[fact].search(text))

    def control_ok(self, index: int, answer: str) -> bool:
        if any(p.search(answer) for f, p in self._pattern.items() if f not in self._ignored):
            return False
        pattern = self.controls[index]["pattern"]
        return pattern is None or bool(re.search(pattern, answer, re.I))

    def score(self, answers: dict) -> dict:
        by_fact = {f: sum(self.hit(f, answers.get(f"fact:{f}:{i}", ""))
                          for i in range(len(v["held"]))) for f, v in self.facts.items()}
        controls = [self.control_ok(i, answers[f"ctrl:{i}"])
                    for i in range(len(self.controls)) if f"ctrl:{i}" in answers]
        return {"fact_hits": sum(by_fact.values()), "by_fact": by_fact,
                "fact_questions": sum(len(v["held"]) for v in self.facts.values()),
                "controls_ok": sum(controls), "controls_answered": len(controls),
                "control_results": controls}
