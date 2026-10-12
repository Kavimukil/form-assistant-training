"""G3/G4 facade. Semantic truth MUST precede surface realization.

Pipeline for one slot (see ``form_sft.dataset.builder``):
    sample = generate_semantic_truth(slot, seed)      # canonical Operation + SemanticContext
    text   = realize_row(sample, slot, seed)          # realize -> correct -> style -> ASR -> finalize
The semantic target is never modified after step 1 (asserted by the builder).
"""
from form_sft.generation.realize import (  # noqa: F401
    apply_asr_noise, apply_correction, prove_target_preserved, realize_row, realize_utterance,
)
from form_sft.generation.semantic import generate_semantic_truth  # noqa: F401
