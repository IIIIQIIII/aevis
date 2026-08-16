import numpy as np

from aevis.library import VerifiedProgramLibrary
from aevis.search import cross_entropy_program_search


def test_verified_library_proposes_complete_programs():
    lib = VerifiedProgramLibrary(capacity=8)
    for i in range(4):
        lib.add(
            fingerprint=np.array([i, i + 1], dtype=np.float32),
            program=np.ones((2, 3), dtype=np.float32) * i,
            score=float(i),
        )

    candidates, mean, std = lib.propose(np.array([1.5, 2.5], dtype=np.float32))
    assert candidates
    assert mean.shape == (2, 3)
    assert std.shape == (2, 3)
    assert np.all(std > 0)


def test_search_uses_whole_program_evaluator():
    rng = np.random.default_rng(0)

    def evaluate(program):
        score = -float(np.square(program - 0.25).mean())
        verified = score > -0.02
        return score, verified, 1

    result = cross_entropy_program_search(
        evaluate,
        program_shape=(2, 2),
        rng=rng,
        interaction_budget=100,
        population=12,
        elite_count=4,
        generations=6,
    )
    assert result.interactions <= 100
    assert result.program.shape == (2, 2)
