import sys

import numpy as np
import pytest

jumanji = pytest.importorskip("jumanji")
import jax  # noqa: E402

sys.path.insert(0, "experiments")
from envs import get_env  # noqa: E402
from accteams.mappo import Config, _kernel_masks, make_trainer  # noqa: E402


def test_kernel_masks_are_proper_coalitions():
    m = np.asarray(_kernel_masks(jax.random.PRNGKey(0), 5, 2000))
    sizes = m.sum(1)
    assert sizes.min() >= 1 and sizes.max() <= 4
    # the Shapley kernel puts more mass on sizes 1 and n-1 than on the middle
    assert (sizes == 1).mean() > (sizes == 2).mean()


@pytest.mark.parametrize("credit", ["team", "resp", "dr"])
def test_every_credit_mode_trains(credit):
    spec = get_env("lbf", 2)
    cfg = Config(num_envs=4, rollout=8, epochs=1, minibatches=2, credit=credit, critic_steps=2, window=2, horizon=2,
                 critic_width=16, hidden=16)
    train, info = make_trainer(spec, cfg)
    params, m = train(jax.random.PRNGKey(0), 2)
    assert np.all(np.isfinite(np.asarray(m["loss"])))
    if credit != "team":
        assert np.all(np.isfinite(np.asarray(m["aux_loss"])))
