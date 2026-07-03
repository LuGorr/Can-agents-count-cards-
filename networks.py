import torch
import torch.nn as nn


def masked_cat(logits, mask):
    # mask has 1 for legal actions, 0 for illegal
    neg = torch.finfo(logits.dtype).min
    logits = torch.where(mask > 0, logits, torch.full_like(logits, neg))
    return torch.distributions.Categorical(logits=logits)


class MLPNet(nn.Module):
    # used for both mlp0 and mlp_history, just different input size
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h = self.body(x)
        logits = self.pi(h)
        val = self.v(h).squeeze(-1)
        return masked_cat(logits, mask), val


class GRUNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def init_hidden(self, batch=1, device="cpu"):
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, x, mask, h):
        is_recurrent_seq = x.dim() == 3

        if not is_recurrent_seq:
            # unsqueeze a temporary time dimension to threat it as a sequence of T=1
            x = x.unsqueeze(0)  # Now (1, B, D)
            mask = mask.unsqueeze(0)  # Now (1, B, n_actions)

        T, B, D = x.shape
        z = self.proj(x.reshape(T * B, D)).reshape(T, B, -1)
        out, h2 = self.gru(z, h)
        out2 = out.reshape(T * B, -1)
        logits = self.pi(out2)
        dist = masked_cat(logits, mask.reshape(T * B, -1))

        if is_recurrent_seq:
            val = self.v(out2).squeeze(-1).reshape(T, B)
        else:
            val = self.v(out2).squeeze(-1)

        return dist, val, h2


def make_net(arch, obs_dim, n_actions, hidden=128):
    if arch in ("mlp0", "mlp_history"):
        return MLPNet(obs_dim, n_actions, hidden)
    if arch == "gru":
        return GRUNet(obs_dim, n_actions, hidden)
    raise ValueError(arch)
