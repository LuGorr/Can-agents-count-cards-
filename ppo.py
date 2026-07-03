import numpy as np
import torch
import torch.nn as nn

from env import OBS_DIM, RANK_DIM
from networks import make_net

DEVICE = "cuda"


def history_block(shoe_history, max_cards):
    # rebuilds the whole one hot card history block from scratch every call
    block = np.zeros((max_cards, RANK_DIM), dtype=np.float32)
    n = min(len(shoe_history), max_cards)
    for i in range(n):
        block[i, shoe_history[i] - 1] = 1.0
    frac = n / max_cards
    return block.reshape(-1), frac


class Buffer:
    def __init__(self):
        self.obs = []
        self.mask = []
        self.act = []
        self.logp = []
        self.val = []
        self.rew = []
        self.done = []
        self.hid = []

    def add(self, o, m, a, lp, v, r, d, h):
        self.obs.append(o)
        self.mask.append(m)
        self.act.append(a)
        self.logp.append(lp)
        self.val.append(v)
        self.rew.append(r)
        self.done.append(d)
        self.hid.append(h)

    def __len__(self):
        return len(self.obs)

    def add_last_reward(self, r):
        if len(self.rew) > 0:
            self.rew[-1] += r


class AgentPolicy:
    # wraps a net + rollout buffer + optimizer for one seat

    def __init__(
        self,
        agent_id,
        n_actions,
        arch,
        hidden=128,
        max_decks=8,
        lr=3e-4,
        device=DEVICE,
        reward_scale=1.0,
    ):
        self.agent_id = agent_id
        self.arch = arch
        self.device = device
        self.n_actions = n_actions
        self.reward_scale = reward_scale
        self.max_hist_cards = max_decks * 52
        if arch == "mlp_history":
            obs_dim = OBS_DIM + self.max_hist_cards * RANK_DIM + 1
        else:
            obs_dim = OBS_DIM
        self.net = make_net(arch, obs_dim, n_actions, hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.is_recurrent = arch == "gru"
        self.reset_episode()

    def reset_episode(self):
        self.hidden = (
            self.net.init_hidden(1, self.device) if self.is_recurrent else None
        )
        self.buf = Buffer()

    def featurize(self, obs_vec, info):
        if self.arch == "mlp_history":
            hb, frac = history_block(info["shoe_history"], self.max_hist_cards)
            return np.concatenate([obs_vec, hb, [frac]]).astype(np.float32)
        return obs_vec

    def bootstrap_value(self, obs_vec, info):
        # fresh V(s_T) for the current obs, used to boostrap truncated rollouts.
        feat = self.featurize(obs_vec, info)
        x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        legal = info["legal_actions"]
        mask = np.zeros(self.n_actions, dtype=np.float32)
        mask[legal] = 1.0
        m = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.is_recurrent:
                _, val, _ = self.net(
                    x, m, self.hidden
                )  # use the live hiden state, post last transition
            else:
                _, val = self.net(x, m)
        return float(val.item())

    def act(self, obs_vec, info, greedy=False):
        feat = self.featurize(obs_vec, info)
        legal = info["legal_actions"]
        mask = np.zeros(self.n_actions, dtype=np.float32)
        mask[legal] = 1.0
        x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.is_recurrent:
                dist, val, h2 = self.net(x, m, self.hidden)
                hidden_in = self.hidden
                self.hidden = h2.detach()
            else:
                dist, val = self.net(x, m)
                hidden_in = None
            a = dist.probs.argmax(dim=-1) if greedy else dist.sample()
            logp = dist.log_prob(a)
        return {
            "action": int(a.item()),
            "logp": float(logp.item()),
            "value": float(val.item()),
            "feat": feat,
            "mask": mask,
            "hidden_in": hidden_in.cpu().numpy() if hidden_in is not None else None,
        }

    def store(self, step_out, reward, done):
        self.buf.add(
            step_out["feat"],
            step_out["mask"],
            step_out["action"],
            step_out["logp"],
            step_out["value"],
            reward * self.reward_scale,
            done,
            step_out["hidden_in"],
        )

    def add_last_reward(self, r):
        # reward scaling keeps value loss magnitud comparable to policy loss
        self.buf.add_last_reward(r * self.reward_scale)

    def share_from(self, other):
        # share net weights + optimizer accross seats. buffer and recurrent
        self.net = other.net
        self.opt = other.opt


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgae = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


def ppo_update(
    policy,
    last_value,
    gamma=0.995,
    lam=0.95,
    clip=0.2,
    vcoef=0.5,
    ecoef=0.01,
    epochs=4,
    chunks=4,
    max_grad_norm=0.5,
):
    buf = policy.buf
    T = len(buf)
    if T < 2:
        return {}  # not enough to learn anithing from

    adv, ret = compute_gae(buf.rew, buf.val, buf.done, last_value, gamma, lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    device = policy.device
    obs = torch.tensor(np.array(buf.obs), dtype=torch.float32, device=device)
    mask = torch.tensor(np.array(buf.mask), dtype=torch.float32, device=device)
    actions = torch.tensor(buf.act, dtype=torch.long, device=device)
    old_logp = torch.tensor(buf.logp, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

    stats = {"pl": 0.0, "vl": 0.0, "ent": 0.0}
    n = 0

    if not policy.is_recurrent:
        idx = np.arange(T)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for c in np.array_split(idx, chunks):
                if len(c) == 0:
                    continue
                dist, val = policy.net(obs[c], mask[c])
                logp = dist.log_prob(actions[c])
                ratio = torch.exp(logp - old_logp[c])
                s1 = ratio * adv_t[c]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[c]
                pl = -torch.min(s1, s2).mean()
                vl = ((val - ret_t[c]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + vcoef * vl - ecoef * ent
                policy.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.net.parameters(), max_grad_norm)
                policy.opt.step()
                stats["pl"] += pl.item()
                stats["vl"] += vl.item()
                stats["ent"] += ent.item()
                n += 1
    else:
        # recurrent case, we chunk along time and carry hiden state accross
        # chunks so the gru actualy sees a sequence instead of iid steps
        bounds = np.array_split(np.arange(T), chunks)
        h0 = (
            torch.tensor(buf.hid[0], dtype=torch.float32, device=device)
            if buf.hid[0] is not None
            else policy.net.init_hidden(1, device)
        )
        for _ in range(epochs):
            hidden = h0
            for b in bounds:
                if len(b) == 0:
                    continue
                sl = slice(b[0], b[-1] + 1)
                oseq = obs[sl].unsqueeze(1)
                mseq = mask[sl].unsqueeze(1)
                dist, val, hout = policy.net(oseq, mseq, hidden.detach())
                logp = dist.log_prob(actions[sl])
                ratio = torch.exp(logp - old_logp[sl])
                s1 = ratio * adv_t[sl]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[sl]
                pl = -torch.min(s1, s2).mean()
                vl = ((val.squeeze(1) - ret_t[sl]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + vcoef * vl - ecoef * ent
                policy.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.net.parameters(), max_grad_norm)
                policy.opt.step()
                hidden = (
                    hout.detach()
                )  # carry state forward, but detach so bptt doesnt leak accross chunks
                stats["pl"] += pl.item()
                stats["vl"] += vl.item()
                stats["ent"] += ent.item()
                n += 1

    if n > 0:
        for k in stats:
            stats[k] /= n
    return stats
