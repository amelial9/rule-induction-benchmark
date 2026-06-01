"""
env.py - Rule-induction benchmark for Mesocosm.
 
The agent sees labeled examples  word -> number  produced by a single hidden
rule, then predicts the number for ONE new query word. The rule is never shown.
 
Difficulty selects which tier of rules is in play:
    easy   -> tier 1 only   (length, vowel count, consonants, first-letter index)
    medium -> tiers 1-2
    hard   -> tiers 1-3      (letter-position sum mod 10, etc.)
Set it with the RULE_DIFFICULTY env var (default: medium):
    RULE_DIFFICULTY=hard mesocosm run local
 
Fairness guarantee: the chosen rule is the ONLY rule in the active pool that
fits the training examples, and the example outputs vary (>=2 distinct values),
so every puzzle is uniquely determinable and actually demonstrates the rule.
"""
 
from __future__ import annotations
 
import os
import random
import re
from typing import Any, Optional
 
from bench_common.env_sdk.base import BaseEnv, StepResult
 
VOWELS = set("aeiou")
 
 
# --- the hidden rules -------------------------------------------------------
# rule_id -> (human description, tier, fn(word) -> int).
# Descriptions are for analysis/showcase only; never shown to the agent.
 
def _vowels(w: str) -> int:
    return sum(c in VOWELS for c in w)
 
 
def _consonants(w: str) -> int:
    return sum(c.isalpha() and c not in VOWELS for c in w)
 
 
def _double_pairs(w: str) -> int:
    return sum(w[i] == w[i + 1] for i in range(len(w) - 1))
 
 
def _first_index(w: str) -> int:
    return ord(w[0]) - ord("a") + 1
 
 
def _second_half(w: str) -> int:
    return sum(c >= "n" for c in w)  # letters n..z
 
 
def _letter_sum_mod10(w: str) -> int:
    return sum(ord(c) - ord("a") + 1 for c in w) % 10
 
 
RULES = {
    # tier 1 - obvious once you see it
    "length":          ("number of letters",                           1, len),
    "vowels":          ("number of vowels",                            1, _vowels),
    "consonants":      ("letters minus vowels (consonants)",           1, _consonants),
    "first_index":     ("alphabet position of the first letter (a=1)", 1, _first_index),
    # tier 2 - needs a second look
    "unique":          ("number of distinct letters",                  2, lambda w: len(set(w))),
    "double_pairs":    ("count of adjacent repeated letters",          2, _double_pairs),
    "len_plus_vowels": ("letters plus vowels",                         2, lambda w: len(w) + _vowels(w)),
    "repeats":         ("letters minus distinct letters (repeats)",    2, lambda w: len(w) - len(set(w))),
    # tier 3 - genuinely sneaky
    "second_half":     ("count of letters from n..z",                  3, _second_half),
    "letter_sum_mod10":("sum of letter positions, mod 10",             3, _letter_sum_mod10),
    "vowels_x2":       ("number of vowels, doubled",                   3, lambda w: _vowels(w) * 2),
}
 
TIERS = {"easy": {1}, "medium": {1, 2}, "hard": {1, 2, 3}}
 
WORDS = [
    "cat", "dog", "hi", "ox", "sun", "moon", "tree", "fish", "bird", "frog",
    "apple", "banana", "elephant", "tiger", "mango", "lemon", "grape", "melon",
    "ocean", "river", "cloud", "stone", "glass", "bread", "honey", "sugar",
    "balloon", "coffee", "kettle", "mirror", "puzzle", "rabbit", "yellow",
    "pillow", "summer", "winter", "letter", "bottle", "ladder", "pepper",
    "umbrella", "octopus", "diamond", "machine", "kitchen", "morning", "journey",
    "violin", "rocket", "planet", "garden", "candle", "window", "silver",
    "forest", "desert", "island", "valley", "meadow", "thunder",
]
 
 
# --- pure helpers (testable without the SDK) --------------------------------
 
def parse_int(text: Any) -> Optional[int]:
    """First integer found in the agent's free-text answer, else None."""
    m = re.search(r"-?\d+", str(text))
    return int(m.group()) if m else None
 
 
def consistent_rules(pool: list[str], examples: list[tuple[str, int]]) -> list[str]:
    """Rule ids in the pool that match every (word, value) example."""
    return [rid for rid in pool if all(RULES[rid][2](w) == v for w, v in examples)]
 
 
def sample_episode(rng: random.Random, pool: list[str], num_train: int,
                   max_resample: int = 300):
    """Pick a rule + words so the rule is the unique fit and outputs vary."""
    rid = train = query = None
    for _ in range(max_resample):
        rid = rng.choice(pool)
        fn = RULES[rid][2]
        words = rng.sample(WORDS, num_train + 1)
        train = [(w, fn(w)) for w in words[:num_train]]
        query = words[num_train]
        if len({v for _w, v in train}) >= 2 and consistent_rules(pool, train) == [rid]:
            return rid, train, query
    return rid, train, query  # rare fallback: accept last draw
 
 
def build_prompt(train: list[tuple[str, int]], query: str) -> str:
    examples = "\n".join(f"  {w}  ->  {v}" for w, v in train)
    return (
        "Each word below maps to a number by a single hidden rule.\n"
        "Work out the rule from the examples, then apply it.\n"
        "Reply with just one non-negative integer.\n\n"
        f"Examples:\n{examples}\n\n"
        f"Now predict:  {query}  ->  ?"
    )
 
 
# --- the environment --------------------------------------------------------
# NOTE: class name MyEnv matches the generated adapter.py import. If you rename
# it, update the import in auxiliary/adapter.py too.
 
class MyEnv(BaseEnv):
    def __init__(self) -> None:
        self.difficulty = os.environ.get("RULE_DIFFICULTY", "medium")
        if self.difficulty not in TIERS:
            self.difficulty = "medium"
        self.num_train = int(os.environ.get("RULE_NUM_TRAIN", "6"))
        self._rng = random.Random()
        self._rule_id: Optional[str] = None
        self._query: Optional[str] = None
        self._answer: Optional[int] = None
 
    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._rng.seed(seed)
        difficulty = params.get("difficulty", self.difficulty)
        difficulty = difficulty if difficulty in TIERS else "medium"
        num_train = int(params.get("num_train", self.num_train))
 
        pool = [rid for rid, (_d, tier, _f) in RULES.items() if tier in TIERS[difficulty]]
        rid, train, query = sample_episode(self._rng, pool, num_train)
 
        self._rule_id = rid
        self._query = query
        self._answer = RULES[rid][2](query)
 
        return {
            "prompt": build_prompt(train, query),
            "examples": [[w, v] for w, v in train],
            "query": query,
        }
 
    # Remap the agent's free text into an int before step() sees it.
    def parse_action(self, action: Any) -> Any:
        return parse_int(action)
 
    def step(self, action: Any) -> StepResult:
        if self._query is None:
            raise RuntimeError("Call reset() before step()")
 
        # Robust to either a pre-parsed int (via parse_action) or raw text.
        guess = action if isinstance(action, int) else parse_int(action)
        correct = guess == self._answer
 
        return StepResult(
            observation={"result": "done"},
            reward=1.0 if correct else 0.0,
            terminated=True,
            truncated=False,
            info={
                "correct": str(correct),
                "given_answer": str(guess),
                "expected": str(self._answer),
                "query": str(self._query),
                "rule_id": str(self._rule_id),
                "rule_description": RULES[self._rule_id][0],
                "difficulty": self.difficulty,
            },
        )
