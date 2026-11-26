"""SAC from scratch for continuous-action Gymnasium tasks (Pendulum-v1). One file, on purpose.

Run:  python rl/sac.py --seed 1
Out:  runs/sac_Pendulum-v1_s1/progress.csv and model.pt
"""
# [permanent practice] Same single-file layout as ppo.py, so the two read side by side.
import argparse
import csv
import copy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=20_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5_000)   # random actions first, to seed the buffer
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=1e-3)
    p.add_argument("--policy-every", type=int, default=2)          # delayed actor updates (as in CleanRL)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out", default="runs")
    return p.parse_args()


class QNet(nn.Module):
    """Q(s, a): state AND action in, one number out."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Actor(nn.Module):
    """Squashed Gaussian: outputs mean and log-std, samples u, returns a = tanh(u) scaled to the action box."""
    def __init__(self, obs_dim, act_dim, low, high):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.register_buffer("scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def forward(self, s):
        h = self.body(s)
        log_std = torch.tanh(self.log_std(h))                      # keep log-std in a sane range
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return self.mu(h), log_std

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)                        # reparameterisation: gradient flows through u
        y = torch.tanh(u)
        a = y * self.scale + self.bias
        logp = torch.distributions.Normal(mu, std).log_prob(u)
        logp -= torch.log(self.scale * (1 - y.pow(2)) + 1e-6)      # tanh + scaling change of variables
        return a, logp.sum(-1), torch.tanh(mu) * self.scale + self.bias


class Replay:
    """Fixed-size ring buffer of (s, a, r, s', terminated)."""
    def __init__(self, size, obs_dim, act_dim):
        self.s = np.zeros((size, obs_dim), np.float32); self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32); self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32); self.size, self.i, self.n = size, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k):
        j = np.random.randint(0, self.n, k)
        t = lambda x: torch.as_tensor(x[j])
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


def soft_update(target, live, tau):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), live.parameters()):
            tp.mul_(1 - tau).add_(tau * p)


def evaluate(actor, env_id, episodes, seed):
    """Deterministic policy: the squashed mean, no sampling."""
    env = gym.make(env_id)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                _, _, a = actor.sample(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, term, trunc, _ = env.step(a.numpy())
            ret += r
            done = term or trunc
        rets.append(ret)
    env.close()
    return float(np.mean(rets))


def main():
    a = parse_args()
    run = f"sac_{a.env}_s{a.seed}"
    os.makedirs(f"{a.out}/{run}", exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    env = gym.make(a.env)
    env.action_space.seed(a.seed)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high

    actor = Actor(od, ad, low, high)
    q1, q2 = QNet(od, ad), QNet(od, ad)
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)              # target copies start identical
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=a.q_lr)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=a.policy_lr)
    target_entropy = -float(ad)
    log_alpha = torch.zeros(1, requires_grad=True)
    a_opt = torch.optim.Adam([log_alpha], lr=a.q_lr)
    buf = Replay(a.buffer_size, od, ad)

    f = open(f"{a.out}/{run}/progress.csv", "w", newline="")
    log = csv.writer(f)
    log.writerow(["step", "eval_return", "train_return", "q_loss", "actor_loss", "alpha", "q_mean", "sps"])
    obs, _ = env.reset(seed=a.seed)
    ep_ret, recent, t0 = 0.0, [], time.time()
    q_loss = pi_loss = torch.tensor(float("nan")); qmean = float("nan")

    for step in range(1, a.total_steps + 1):
        # ---- 1. Act once, store once ----
        if step <= a.learning_starts:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act, _, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32))
            act = act.numpy()
        nobs, r, term, trunc, _ = env.step(act)
        buf.add(obs, act, r, nobs, float(term))      # store TERMINATED only: a time-out still has a future
        ep_ret += r
        obs = nobs
        if term or trunc:
            recent.append(ep_ret); ep_ret = 0.0
            obs, _ = env.reset()

        if step > a.learning_starts:
            s, ac, rw, s2, d = buf.sample(a.batch_size)
            alpha = log_alpha.exp().detach()
            # ---- 2. Target y, frozen (no gradient) ----
            with torch.no_grad():
                a2, logp2, _ = actor.sample(s2)                  # fresh next action from the CURRENT actor
                q_next = torch.min(q1_t(s2, a2), q2_t(s2, a2)) - alpha * logp2
                y = rw + a.gamma * (1 - d) * q_next
            # ---- 3. Both critics regress to y ----
            q1v, q2v = q1(s, ac), q2(s, ac)
            q_loss = F.mse_loss(q1v, y) + F.mse_loss(q2v, y)
            q_opt.zero_grad(); q_loss.backward(); q_opt.step()
            qmean = q1v.mean().item()
            # ---- 4-5. Actor and alpha (every `policy-every` steps, that many times) ----
            if step % a.policy_every == 0:
                for _ in range(a.policy_every):
                    pa, logp, _ = actor.sample(s)
                    pi_loss = (alpha * logp - torch.min(q1(s, pa), q2(s, pa))).mean()
                    pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()
                    with torch.no_grad():
                        _, logp, _ = actor.sample(s)
                    a_loss = (-log_alpha.exp() * (logp + target_entropy)).mean()
                    a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            # ---- 6. Targets creep towards the live critics ----
            soft_update(q1_t, q1, a.tau); soft_update(q2_t, q2, a.tau)

        if step % a.eval_every == 0:
            ev = evaluate(actor, a.env, a.eval_episodes, seed=10_000)
            tr = float(np.mean(recent[-5:])) if recent else float("nan")
            log.writerow([step, ev, tr, q_loss.item(), pi_loss.item(), log_alpha.exp().item(), qmean,
                          int(step / (time.time() - t0))])
            f.flush()
            print(f"{run} step {step:>6} eval {ev:8.1f} train {tr:8.1f} alpha {log_alpha.exp().item():.3f} q {qmean:7.1f}")

    torch.save(actor.state_dict(), f"{a.out}/{run}/model.pt")
    f.close()
    env.close()


if __name__ == "__main__":
    main()
