"""Sampler for historical normalized session path fragments."""

from __future__ import annotations

import numpy as np

from synth_shadow.models.session_path_model import SessionBlock


class PathSampler:
    """Sample session-matching historical blocks."""

    def __init__(self, blocks: list[SessionBlock], seed: int) -> None:
        if not blocks:
            raise ValueError("PathSampler requires at least one historical session block.")
        self.blocks = blocks
        self.rng = np.random.default_rng(seed)

    def sample(self, session: str) -> SessionBlock:
        candidates = [block for block in self.blocks if block.session == session]
        if not candidates:
            candidates = self.blocks
        return candidates[int(self.rng.integers(0, len(candidates)))]
