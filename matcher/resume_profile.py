"""
profile.py -- structured resume/skill profile input for the Phase 2 matcher.

A deliberately small, checker-`Profile`-style data shape -- not full NLP
resume parsing. `raw_text` is extracted mechanically (resume_extract.py);
`skills` is a manually curated list, honestly derived by reading that
extracted text (the same "read it and write down what's really there"
discipline Phase 0's hand-labeled eval set used) -- never invented.
See NEXT-BUILD-SPEC.md Scope-in section 1 / Scope-out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class ResumeProfile:
    name: str
    raw_text: str
    skills: List[str] = field(default_factory=list)
    experience_level: str = ""

    @staticmethod
    def from_dict(d: dict) -> "ResumeProfile":
        return ResumeProfile(
            name=d.get("name", ""),
            raw_text=d.get("raw_text", ""),
            skills=list(d.get("skills", [])),
            experience_level=d.get("experience_level", ""),
        )

    @staticmethod
    def load(path: str) -> "ResumeProfile":
        with open(path, "r", encoding="utf-8") as f:
            return ResumeProfile.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw_text": self.raw_text,
            "skills": self.skills,
            "experience_level": self.experience_level,
        }
