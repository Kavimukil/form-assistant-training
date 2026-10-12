"""Teacher adapter (G4).

Hard rules:
  * the teacher NEVER produces labels; it may only paraphrase a *known* semantic target;
  * with no provider/credentials configured it falls back to deterministic paraphrase
    templates (``lexicon`` teacher_generated frames) -- no external API is ever called;
  * a paraphrase is rejected unless target preservation can be proven.
"""
from __future__ import annotations

import os

from form_sft.generation import lexicon as lex

PROVIDER_ENV = "FORM_SFT_TEACHER_PROVIDER"


class TeacherRejected(RuntimeError):
    """Raised when target preservation cannot be proven for a teacher row."""


class TeacherAdapter:
    mode = "deterministic_paraphrase_no_credentials"

    def __init__(self) -> None:
        if os.environ.get(PROVIDER_ENV):
            # Preserve the future stub: a real provider is intentionally not wired in Phase 0.
            raise NotImplementedError("external teacher providers are not implemented in Phase 0")

    def frame(self, specialization_mode: str, variant: int) -> str:
        table = lex.ACTIVE_FRAMES if specialization_mode == "active_field" else lex.REACHABLE_FRAMES
        return table["teacher_generated"][variant]

    @staticmethod
    def require_preserved(proof_ok: bool, slot_id: str) -> None:
        if not proof_ok:
            raise TeacherRejected(f"{slot_id}: target preservation not provable; row rejected")
