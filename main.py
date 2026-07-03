import csv
import os
import shutil
import time
from multiprocessing import Pool

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from env import BlackjackEnv
from ppo import DEVICE, AgentPolicy, ppo_update

torch.set_num_threads(1)  # avoid thread oversubscription across worker processes
ARCHS = ["mlp0", "mlp_history", "gru"]
MODES = {"solo": 1, "group": 4}
SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS_PER_AGENT = 2_000_000  # env steps of experience per agent per run
EVAL_STEPS_PER_AGENT = 80_000  # env steps of evaluation per agent per run
ROLLOUT_LEN_PER_AGENT = 256  # ppo update every this many steps per agent
NUM_WORKERS = 5
SHARE_PARAMS = True  # one shared net accross seats in a combo (seats are symmetric)

# which eval metric decides the "best" seed to keep per (arch, mode), and
# which direction is better. house edge % is net profit / wagered from the
# agent's point of view, so higher (less negative, or positive) = better play.
BEST_METRIC = "house_edge_pct"
HIGHER_IS_BETTER = True

ENV_KWARGS = dict(
    num_decks=4,
    penetration=0.75,
    bet_levels=(1, 2, 3, 5, 10),
    starting_bankroll=200.0,
    max_rounds=80,
)

# dividing by the largest bet keeps rewards in roughly -1, 1.5 and prevents
# the value loss from dominating the policy loss
REWARD_SCALE = 1.0 / max(ENV_KWARGS["bet_levels"])

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
TMP_MODELS_DIR = os.path.join(MODELS_DIR, "tmp")  # per-seed checkpoints, pruned at the end
HI_LO = {
    1: -1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 0,
    8: 0,
    9: 0,
    10: -1,
    11: -1,
    12: -1,
    13: -1,
}


def save_checkpoint(policies, arch, num_seats, path):
    # save just the net weights (not optimizer/buffers) for every seat. when
    # SHARE_PARAMS is on all seats point at the same net so this is a tiny,
    # de-duplicated file; when it's off each seat gets its own entry.
    state = {
        "arch": arch,
        "num_seats": num_seats,
        "nets": {a: p.net.state_dict() for a, p in policies.items()},
    }
    torch.save(state, path)


def train_one(
    arch, num_seats, seed, total_steps_per_agent, rollout_len_per_agent, share_params
):
    env = BlackjackEnv(num_seats=num_seats, seed=seed, **ENV_KWARGS)
    policies = {
        a: AgentPolicy(
            a,
            env.action_space.n,
            arch,
            max_decks=ENV_KWARGS["num_decks"],
            reward_scale=REWARD_SCALE,
        )
        for a in env.possible_agents
    }
    if share_params and num_seats > 1:
        # use a shared net, buffers and recurrent hiden states stay per seat of course
        template = policies[env.possible_agents[0]]
        for a in env.possible_agents[1:]:
            policies[a].share_from(template)

    # adapt steps to be per agent
    total_steps = total_steps_per_agent * num_seats
    rollout_len = rollout_len_per_agent * num_seats

    obs, info = env.reset()
    agent = info["agent"]
    was_active = {a: True for a in policies}
    since_update = 0
    step = 0

    wagered = {a: 0.0 for a in policies}
    shoe_curve = []  # list of (edge_pct, rounds) per finshed shoe, for the plots later

    while step < total_steps:
        p = policies[agent]
        step_out = p.act(obs, info)
        if info["phase"] == "bet":
            wagered[agent] += env.bet_levels[step_out["action"]]

        obs, _, terminated, truncated, info = env.step(step_out["action"])
        p.store(
            step_out, reward=0.0, done=False
        )  # reward comes later, see add_last_reward below

        for a, r in info["rewards"].items():
            if r != 0.0:
                policies[a].add_last_reward(r)

        # if someone busts mid shoe we have to close their buffer out rigth away
        for a in list(policies.keys()):
            if was_active[a] and env.busted.get(a, False):
                pa = policies[a]
                if len(pa.buf) > 0:
                    pa.buf.done[-1] = True
                    if len(pa.buf) > 1:
                        ppo_update(pa, last_value=0.0)
                pa.reset_episode()
                was_active[a] = False

        step += 1
        since_update += 1

        if terminated or truncated:
            for a, p2 in policies.items():
                if was_active[a] and len(p2.buf) > 0:
                    # mark last transition terminal so GAE zeros the boostrap
                    # regardless of what last_value we pass in
                    p2.buf.done[-1] = True
                    ppo_update(p2, last_value=0.0)
                p2.reset_episode()
                was_active[a] = True
            net = np.mean(
                [env.bankrolls[a] - ENV_KWARGS["starting_bankroll"] for a in policies]
            )
            tot_w = np.mean([wagered[a] for a in policies]) or 1.0
            shoe_curve.append(
                {"edge_pct": 100.0 * net / tot_w, "rounds": env.round_num}
            )
            wagered = {a: 0.0 for a in policies}
            since_update = 0
            obs, info = env.reset()
            agent = info["agent"]
            continue

        if since_update >= rollout_len:
            # boostrap with a fresh value of the current obs, but only for the
            # seat whose turn it actualy is right now
            cur = info["agent"]
            for a, p2 in policies.items():
                if not (was_active[a] and len(p2.buf) > 1):
                    continue
                if a == cur:
                    boot = p2.bootstrap_value(obs, info)
                else:
                    # for seats that havent acted yet, fall back to the last buffered value
                    boot = p2.buf.val[-1]
                ppo_update(p2, last_value=boot)
                p2.buf = type(p2.buf)()
            since_update = 0

        agent = info["agent"]

    return env, policies, shoe_curve


def evaluate_one(env, policies, num_decks, eval_steps_per_agent):
    # frozen rollout, no learning here, we just collect win rate / edge / bet-vs-count corr
    for p in policies.values():
        p.reset_episode()
    obs, info = env.reset()
    agent = info["agent"]
    was_active = {a: True for a in policies}

    bets, counts, outcomes = [], [], []
    wagered_total = {a: 0.0 for a in policies}
    net_total = {a: 0.0 for a in policies}

    step = 0
    eval_steps = eval_steps_per_agent * len(policies)
    while step < eval_steps:
        p = policies[agent]
        if info["phase"] == "bet":
            # true count = running hi-lo count normalzed by decks left in the shoe
            running = sum(HI_LO[c] for c in info["shoe_history"])
            decks_left = max(info["cards_remaining_frac"] * num_decks, 0.5)
            tc = running / decks_left
        step_out = p.act(obs, info)
        if info["phase"] == "bet":
            amt = env.bet_levels[step_out["action"]]
            bets.append(amt)
            counts.append(tc)
            wagered_total[agent] += amt

        obs, reward, terminated, truncated, info = env.step(step_out["action"])

        for a, r in info["rewards"].items():
            if r != 0.0:
                net_total[a] += r
                outcomes.append(r)

        for a in list(policies.keys()):
            if was_active[a] and env.busted.get(a, False):
                policies[a].reset_episode()
                was_active[a] = False

        if terminated or truncated:
            for p2 in policies.values():
                p2.reset_episode()
            was_active = {a: True for a in policies}
            obs, info = env.reset()

        agent = info["agent"]
        step += 1

    outcomes = np.array(outcomes)
    win = float(np.mean(outcomes > 0)) if len(outcomes) else float("nan")
    lose = float(np.mean(outcomes < 0)) if len(outcomes) else float("nan")
    wsum = sum(wagered_total.values())
    nsum = sum(net_total.values())
    edge = 100.0 * nsum / wsum if wsum > 0 else float("nan")

    if len(bets) > 5 and np.std(bets) > 1e-6 and np.std(counts) > 1e-6:
        corr, pval = pearsonr(counts, bets)
    else:
        corr, pval = float("nan"), float("nan")  # not enough variance to say anything

    return dict(
        win_rate=win,
        lose_rate=lose,
        house_edge_pct=edge,
        n_rounds=len(outcomes),
        bet_count_corr=float(corr),
        bet_count_pval=float(pval),
        n_bets=len(bets),
    )


def run_combo(combo):
    arch, mode, nseats, seed = (
        combo  # imap_unordered only takes one arg, hence the tuple
    )
    run_id = f"{arch}_{mode}_seed_{seed}"
    t1 = time.time()
    env, policies, shoe_curve = train_one(
        arch,
        nseats,
        seed,
        TOTAL_STEPS_PER_AGENT,
        ROLLOUT_LEN_PER_AGENT,
        SHARE_PARAMS,
    )
    train_time = time.time() - t1

    ev_env = BlackjackEnv(num_seats=nseats, seed=seed + 10000, **ENV_KWARGS)
    metrics = evaluate_one(
        ev_env, policies, ENV_KWARGS["num_decks"], EVAL_STEPS_PER_AGENT
    )
    metrics["run_id"] = run_id
    metrics["arch"] = arch
    metrics["mode"] = mode
    metrics["seed"] = seed

    # every seed's weights get stashed here; main() promotes the best one per
    # (arch, mode) to results/models/ and deletes the rest once all runs are in
    ckpt_path = os.path.join(TMP_MODELS_DIR, f"{run_id}.pt")
    save_checkpoint(policies, arch, nseats, ckpt_path)
    metrics["checkpoint_path"] = ckpt_path

    edge_curve = [s["edge_pct"] for s in shoe_curve]
    return metrics, edge_curve, train_time, len(shoe_curve)


def promote_best_checkpoints(all_rows):
    # pick the best-scoring seed for every (arch, mode) combo that actually
    # ran (works fine if only "solo" or only "group" is in MODES, or even a
    # single arch), copy its weights out of tmp/, then wipe tmp/ entirely.
    best = {}  # (arch, mode) -> row
    for r in all_rows:
        key = (r["arch"], r["mode"])
        val = r.get(BEST_METRIC)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue  # can't compare nan, skip this row as a candidate
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        cur_val = cur[BEST_METRIC]
        better = val > cur_val if HIGHER_IS_BETTER else val < cur_val
        if better:
            best[key] = r

    for (arch, mode), row in best.items():
        dst = os.path.join(MODELS_DIR, f"{arch}_{mode}_best.pt")
        shutil.copyfile(row["checkpoint_path"], dst)
        print(
            f"best {arch}/{mode}: seed={row['seed']} "
            f"{BEST_METRIC}={row[BEST_METRIC]:+.3f} -> {dst}"
        )

    if not best:
        print("no runs had a valid metric to compare, no models promoted")

    shutil.rmtree(TMP_MODELS_DIR, ignore_errors=True)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    os.makedirs(TMP_MODELS_DIR, exist_ok=True)

    print("using device:", DEVICE, f"({NUM_WORKERS} worker process(es))")

    combos = [
        (arch, mode, nseats, seed)
        for arch in ARCHS
        for mode, nseats in MODES.items()
        for seed in SEEDS
    ]

    all_rows = []
    curves = {(arch, mode): [] for arch in ARCHS for mode in MODES}

    t0 = time.time()
    done = 0
    with Pool(processes=NUM_WORKERS) as pool:
        # imap_unordered streams results back as soon as whichever worker
        # finshes first, no point waiting around in order
        for metrics, edge_curve, train_time, n_shoes in pool.imap_unordered(
            run_combo, combos
        ):
            done += 1
            print(f"\n[{done}/{len(combos)}] {metrics['run_id']}:")
            print(f"  trained in {train_time:.1f}s, {n_shoes} shoes")
            print(
                f"  eval: win={metrics['win_rate']:.3f} edge={metrics['house_edge_pct']:+.2f}% "
                f"corr={metrics['bet_count_corr']:.3f} (p={metrics['bet_count_pval']:.3f})"
            )
            all_rows.append(metrics)
            curves[(metrics["arch"], metrics["mode"])].append(edge_curve)

    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")

    print()
    promote_best_checkpoints(all_rows)

    # dump raw csv (checkpoint_path is a run-local tmp file, not useful once
    # tmp/ has been cleaned up, so it's left out of the saved summary)
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [k for k in all_rows[0].keys() if k != "checkpoint_path"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: v for k, v in r.items() if k != "checkpoint_path"})
    print("wrote", csv_path)

    make_plots(all_rows, curves)


def make_plots(all_rows, curves):
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    colors = {"mlp0": "red", "mlp_history": "green", "gru": "blue"}

    # learning curves, one plot per table mode
    for mode in MODES:
        plt.figure(figsize=(7, 4.5))
        for arch in ARCHS:
            seed_curves = curves[(arch, mode)]
            if not seed_curves:
                continue
            minlen = min(
                len(c) for c in seed_curves
            )  # trim to shortest run so we can average
            arr = np.array([c[:minlen] for c in seed_curves])
            mean_c = arr.mean(axis=0)
            plt.plot(mean_c, label=arch, color=colors[arch])
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.xlabel("shoe #")
        plt.ylabel("edge % that shoe")
        plt.title(f"learning curve - {mode}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"learning_curve_{mode}.png"), dpi=130)
        plt.close()

    #  bar chart: final house edge by arch/mode
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(ARCHS))
    width = 0.35
    for i, mode in enumerate(MODES):
        means = []
        for arch in ARCHS:
            vals = [
                r["house_edge_pct"]
                for r in all_rows
                if r["arch"] == arch and r["mode"] == mode
            ]
            means.append(np.mean(vals))
        plt.bar(x + (i - 0.5) * width, means, width, label=mode)
    plt.xticks(x, ARCHS)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.ylabel("house edge %")
    plt.title("final evaluated house edge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "final_edge.png"), dpi=130)
    plt.close()

    # bar chart: bet vs true count correlation (this is basicaly our "are you
    # counting cards" detector)
    plt.figure(figsize=(7, 4.5))
    for i, mode in enumerate(MODES):
        means = []
        for arch in ARCHS:
            vals = [
                r["bet_count_corr"]
                for r in all_rows
                if r["arch"] == arch and r["mode"] == mode
            ]
            means.append(np.mean(vals))
        plt.bar(x + (i - 0.5) * width, means, width, label=mode)
    plt.xticks(x, ARCHS)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.ylabel("corr(bet size, true count)")
    plt.title("card counting fingerprint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bet_count_corr.png"), dpi=130)
    plt.close()

    print("plots saved to", plots_dir)


if __name__ == "__main__":
    main()
