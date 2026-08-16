import numpy as np

from experiments.atari.common import AtariLatentProgram, AtariProgramSpec
from aevis.search import cross_entropy_program_search


def test_atari_latent_program_without_ale():
    spec = AtariProgramSpec(latent_dim=8, pooled_size=6, basis_seed=1)
    model = AtariLatentProgram(n_actions=6, frame_stack=4, spec=spec)
    obs = np.zeros((4, 84, 84), dtype=np.uint8)
    obs[:, 20:40, 30:60] = 255
    latent = np.ones(8, dtype=np.float32) * 0.1

    action = model.action(latent, obs)
    fingerprint = model.fingerprint(obs)

    assert 0 <= action < 6
    assert fingerprint.shape == (36,)


def test_variable_cost_search_counts_overshoot():
    rng = np.random.default_rng(0)

    def evaluate(program):
        return float(program[0]), False, 7

    result = cross_entropy_program_search(
        evaluate=evaluate,
        program_shape=(2,),
        rng=rng,
        interaction_budget=10,
        population=4,
        elite_count=2,
        generations=1,
    )

    # Two 7-cost whole-program evaluations are actually executed. The search
    # may overshoot a variable-cost budget, but must never undercount it.
    assert result.interactions == 14
