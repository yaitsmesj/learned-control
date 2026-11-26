"""Hand-checked numbers from the theory manual, as tests. If these fail, the maths in ppo.py changed."""
# [permanent practice] Unit tests pin the formulas; training curves test the whole system.
import torch

from rl.ppo import compute_gae, ppo_policy_loss


def test_gae_three_steps():
    # TD errors 0.5, -0.2, 0.3 with gamma=0.99, lambda=0.95 (manual Part 4 worked example).
    # Choose values and rewards so the deltas come out exactly: V=0 everywhere, rewards = deltas.
    r = torch.tensor([[0.5], [-0.2], [0.3]])
    v = torch.zeros(3, 1)
    adv, ret = compute_gae(r, v, torch.zeros(1), torch.zeros(3, 1), 0.99, 0.95)
    assert torch.allclose(adv.squeeze(), torch.tensor([0.5771, 0.0822, 0.3000]), atol=1e-3)
    assert torch.allclose(ret, adv)  # returns = advantages + values, and values are 0


def test_termination_zeroes_the_future():
    # A terminated step must ignore the next value; a non-terminated one must use it.
    r = torch.tensor([[1.0]]); v = torch.tensor([[0.0]]); nv = torch.tensor([10.0])
    adv_term, _ = compute_gae(r, v, nv, torch.tensor([[1.0]]), 0.9, 0.95)
    adv_cont, _ = compute_gae(r, v, nv, torch.tensor([[0.0]]), 0.9, 0.95)
    assert adv_term.item() == 1.0
    assert abs(adv_cont.item() - (1.0 + 0.9 * 10.0)) < 1e-6


def test_clipped_objective_table():
    # Manual Part 5 table: (advantage, ratio) -> value PPO uses.
    cases = [(2.0, 1.1, 2.2), (2.0, 1.5, 2.4), (-2.0, 0.5, -1.6), (-2.0, 1.5, -3.0)]
    for adv, ratio, used in cases:
        new_logp = torch.log(torch.tensor([ratio])); old_logp = torch.zeros(1)
        loss, _ = ppo_policy_loss(new_logp, old_logp, torch.tensor([adv]), 0.2)
        assert abs(-loss.item() - used) < 1e-5, (adv, ratio)


def test_timeout_does_not_double_bootstrap():
    # Episode times out at t=0. The time-out bootstrap (gamma * V(final_obs) = 0.9 * 10) is already
    # in the reward. values[1] = 50 belongs to the NEXT episode and must not leak into step 0.
    rewards = torch.tensor([[1.0 + 0.9 * 10.0], [0.0]])
    values = torch.tensor([[0.0], [50.0]])
    done = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(rewards, values, torch.tensor([0.0]), done, gamma=0.9, lam=1.0)
    assert abs(adv[0, 0].item() - 10.0) < 1e-6  # 1 + 9, not 1 + 9 + 0.9 * 50


def test_iqm_drops_the_tails():
    from rl.iqm import iqm
    assert iqm([0, 10, 10, 10, 10, 10, 10, 1000]) == 10.0
