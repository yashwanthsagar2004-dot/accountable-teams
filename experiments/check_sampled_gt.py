"""Check that permutation-sampled ground truth (critic_budget --gt sampled) matches exact 2^n ground truth at n = 4.
z = (sampled - exact) / se should have mean ~0 and sd ~1. 11 Nov 2026 CPU run: 48 values, mean -0.18, sd 1.01.
Run from the repository root."""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path[:0] = ["src", "experiments"]
from critic_budget import sampled_shapley, direct_permutation_audit, take
from accteams.branching import all_masks, make_branch_fn, shapley_from_mask_values
from envs import collect, get_env
spec = get_env("lbf", 4); N = spec.n
branch = make_branch_fn(spec.step, spec.policy, spec.reference_action, horizon=10, gamma=0.99, window=20)
run = jax.jit(jax.vmap(branch, in_axes=(None, None, 0, 0)))
states, obs, last = collect(spec, jax.random.PRNGKey(3), 40)
live = np.where(spec.live(states) & ~last)[0][:12]
M = jnp.asarray(all_masks(N)); z = []
for j, idx in enumerate(live):
    s, o = take(states, idx), take(obs, idx)
    a = spec.policy(o, jax.random.PRNGKey(100 + j))
    R = 400
    m = jnp.asarray(np.repeat(all_masks(N), R, 0)); k = jnp.tile(jax.random.split(jax.random.PRNGKey(j), R), (2 ** N, 1))
    ex = shapley_from_mask_values(np.asarray(run(s, a, m, k)).reshape(2 ** N, R).mean(1), N)
    sp, se = sampled_shapley(run, s, a, N, 64, 16, jax.random.PRNGKey(10 + j), np.random.default_rng(j))
    au, _ = direct_permutation_audit(run, s, a, N, 4, jax.random.PRNGKey(50 + j), np.random.default_rng(99 + j))
    ok = se > 1e-9
    z += list(((sp - ex)[ok] / se[ok]))
    print(j, "exact", np.round(ex, 3), "sampled", np.round(sp, 3), "se", np.round(se, 3), "audit32", np.round(au, 3))
z = np.array(z); print("z-scores: n", len(z), "mean", z.mean().round(2), "sd", z.std().round(2), "|z|>2.58:", (abs(z) > 2.58).mean().round(3))
