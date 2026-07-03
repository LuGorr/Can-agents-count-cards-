import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MAX_SEATS = 7
RANK_DIM = 13
HAND_FEAT_DIM = RANK_DIM + 4  # histogram of ranks + total/soft/bust/ncards

# card ranks 1-13, 1=ace, 11/12/13 = J/Q/K, all worth 10 exept ace
RANKS = list(range(1, 14))

_RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


def _card_str(rank):
    # human readable card label, used only by render()
    return _RANK_NAMES.get(rank, str(rank))


# onw round obs vector size
OBS_DIM = (
    2  # phase (betting, playing)
    + MAX_SEATS  # one hot encoded vector representing which seat this agent is sitting in
    + 1  # active seats
    + 1  # tanh(bankroll)
    + 1  # tanh(currnet bet)
    + HAND_FEAT_DIM  # own hand features
    + RANK_DIM  # one hot indicator of the dealer's face up card rank (all zeros if no card is showed yet)
    + 1  # % of cards left in the shoe
    + (MAX_SEATS - 1)
    * (
        HAND_FEAT_DIM + 2
    )  # other players hand features and current bet and seat existence
)


class Shoe:
    # a shoe of num_decks decks, shuffled, with a cut card for reshuffling
    def __init__(self, num_decks=6, penetration=0.75, rng=None):
        self.num_decks = num_decks
        self.penetration = penetration
        self.rng = rng if rng is not None else random.Random()
        self.total_cards = 52 * num_decks
        self.cut_index = int(self.total_cards * (1 - penetration))
        self.cards = []
        self.reshuffle()

    def reshuffle(self):
        deck = []
        for _ in range(self.num_decks):
            deck += RANKS * 4
        self.rng.shuffle(deck)
        self.cards = deck
        self.needs_reshuffle = False

    def remaining(self):
        return len(self.cards)

    def draw(self):
        if len(self.cards) == 0:
            self.reshuffle()  # shouldn't really happen but just in case :)
        c = self.cards.pop()
        if len(self.cards) <= self.cut_index:  # apply cutting logic
            self.needs_reshuffle = True
        return c

    def maybe_reshuffle(self):
        if self.needs_reshuffle:
            self.reshuffle()


class Hand:
    def __init__(self):
        self.cards = []

    def add(self, c):
        self.cards.append(c)

    def total(self):
        # so this took me more than i like to admit...
        # we calculate the lowest possible score (aces value 1)
        t = sum(min(c, 10) for c in self.cards)
        aces = self.cards.count(1)
        # soft in blackjack means a hand in which one ace is counted as 11 (of course 2 aces as 11 = 22 we bust = illegal)
        soft = False
        if aces > 0 and t + 10 <= 21:  # can we upgrade?
            t += 10
            soft = True
        return t, soft

    def value(self):
        return self.total()[0]

    def is_bust(self):
        return self.total()[0] > 21

    def is_blackjack(self):
        return len(self.cards) == 2 and self.total()[0] == 21

    def __len__(self):
        return len(self.cards)


def _hand_feat(cards):
    hist = np.zeros(RANK_DIM, dtype=np.float32)
    for c in cards:
        hist[c - 1] += 1.0
    t = sum(min(c, 10) for c in cards)
    aces = cards.count(1)
    soft = False
    if aces > 0 and t + 10 <= 21:
        t += 10
        soft = True
    extra = np.array(
        [t / 21.0, float(soft), float(t > 21), len(cards) / 12.0], dtype=np.float32
    )
    return np.concatenate([hist, extra])


def featurize(o):
    # o is the raw obs dict, turn it into a flat float32 vector
    phase = np.array(
        [1.0, 0.0] if o["phase"] == "bet" else [0.0, 1.0],
        dtype=np.float32,  # one hot encoding
    )
    seat = np.zeros(MAX_SEATS, dtype=np.float32)
    seat[o["seat_id"]] = 1.0  # one hot encoding
    nseats = np.array(
        [o["num_seats"] / MAX_SEATS], dtype=np.float32
    )  # normalized number of seaths
    # keep bank and bets bounded with normalization and tanh
    bank = np.array([np.tanh(o["own_bankroll"] / 100.0)], dtype=np.float32)
    bet = np.array([np.tanh(o["own_bet"] / 20.0)], dtype=np.float32)
    hand = _hand_feat(o["own_hand"])  # featurize hand
    dealer = np.zeros(RANK_DIM, dtype=np.float32)
    rank = o["dealer_upcard"]
    if rank is not None:
        dealer[rank - 1] = 1.0
    rem = np.array([o["cards_remaining_frac"]], dtype=np.float32)

    ids = sorted(o["other_hands"].keys() | o["other_bets"].keys())
    others = []
    for i in range(MAX_SEATS - 1):
        if i < len(ids):
            oid = ids[i]
            hf = _hand_feat(o["other_hands"].get(oid, []))
            ob = np.array(
                [np.tanh(o["other_bets"].get(oid, 0.0) / 20.0), 1.0], dtype=np.float32
            )
        else:
            hf = np.zeros(HAND_FEAT_DIM, dtype=np.float32)
            ob = np.zeros(2, dtype=np.float32)
        others.append(np.concatenate([hf, ob]))
    others = np.concatenate(others) if others else np.zeros(0, dtype=np.float32)

    return np.concatenate(
        [phase, seat, nseats, bank, bet, hand, dealer, rem, others]
    ).astype(np.float32)


class BlackjackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_seats=1,
        num_decks=6,
        penetration=0.75,
        bet_levels=(1, 2, 3, 5, 10),
        starting_bankroll=200.0,
        max_rounds=80,
        seed=None,
    ):
        super().__init__()
        assert num_seats >= 1
        self.num_seats = num_seats
        self.num_decks = num_decks
        self.penetration = penetration
        self.bet_levels = list(bet_levels)
        self.starting_bankroll = starting_bankroll
        self.max_rounds = max_rounds
        # one agent per seat, no friends allowed :)
        self.possible_agents = [f"seat_{i}" for i in range(num_seats)]
        self.n_actions = len(self.bet_levels) + 2  # + hit, stand

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(OBS_DIM,),
            dtype=np.float32,  # high bounds set for card counts, low one for symmetry
        )

        self._rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self.shoe = Shoe(
            num_decks=self.num_decks, penetration=self.penetration, rng=self._rng
        )
        self.shoe_history = []

        self.agents = list(self.possible_agents)
        self.bankrolls = {a: self.starting_bankroll for a in self.possible_agents}
        self.busted = {a: False for a in self.possible_agents}

        self.round_num = 0
        self.hands = {}
        self.dealer_hand = Hand()
        self.dealer_hole_revealed = False  # dealer second card delayed reveal logic
        self.bets = {}
        self.rewards = {a: 0.0 for a in self.possible_agents}

        self._phase = None
        self._turn_queue = []
        self.current_agent = None
        self._just_resolved = False

        self._start_round()  # shall we start? :)

        obs = featurize(
            self._raw_obs(self.current_agent)
        )  # prepare initial observations
        info = self._info()
        return obs, info

    def step(self, action):
        agent = self.current_agent
        assert agent is not None, "step() called after episode ended, call reset()"

        legal = self._legal(agent)  # get legal action indexes
        if action not in legal:
            # be forgiving about illegal actions instead of crashing
            # (so env.action_space.sample() always works)
            action = legal[0]

        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._just_resolved = False

        if self._phase == "bet":
            self.bets[agent] = self.bet_levels[action]  # set bet
            if self._turn_queue:
                self.current_agent = self._turn_queue.pop(0)  # set new current player
            else:
                self._deal()  # if everyone finished betting deal the cards
        elif self._phase == "play":
            hit_idx = len(self.bet_levels)
            hand = self.hands[agent]
            if action == hit_idx:  # if the bot hists draw the card
                self._draw(hand, True)
                if hand.is_bust():  # handle bust
                    self._advance()
            else:
                self._advance()  # if the bot stands
        else:
            raise RuntimeError(f"bad phase {self._phase}")

        reward = (
            self.rewards.get(agent, 0.0) if self._just_resolved else 0.0
        )  # set rewards at the end of round

        terminated = False
        truncated = False
        # handle dones and prepare observations
        if self._phase == "done":
            if self.round_num > self.max_rounds:
                truncated = True
            else:
                terminated = True
            obs = featurize(self._raw_obs(agent))
        else:
            obs = featurize(self._raw_obs(self.current_agent))

        info = self._info()  # set infos
        info["rewards"] = dict(self.rewards)
        return obs, reward, terminated, truncated, info

    def render(self):
        pct_left = 100.0 * self.shoe.remaining() / self.shoe.total_cards
        lines = [
            f"round {self.round_num}/{self.max_rounds}  phase={self._phase}  "
            f"shoe: {self.shoe.remaining()}/{self.shoe.total_cards} cards left ({pct_left:.0f}%)"
        ]

        # dealer line: hide the hole card while play is still in progress
        if len(self.dealer_hand) == 0:
            lines.append("dealer: (no cards yet)")
        elif self._phase == "play" and not self.dealer_hole_revealed:
            up = _card_str(self.dealer_hand.cards[0])
            lines.append(f"dealer: [{up}, ??]")
        else:
            total, soft = self.dealer_hand.total()
            cards = ", ".join(_card_str(c) for c in self.dealer_hand.cards)
            tag = " (soft)" if soft else ""
            tag += " BUST" if total > 21 else ""
            lines.append(f"dealer: [{cards}] = {total}{tag}")

        for a in self.possible_agents:
            marker = ">> " if a == self.current_agent else "   "
            bank = self.bankrolls[a]
            bet = self.bets.get(a)
            bet_str = f"${bet:.0f}" if bet is not None else "-"

            hand = self.hands.get(a)
            if hand is not None and len(hand) > 0:
                total, soft = hand.total()
                cards = ", ".join(_card_str(c) for c in hand.cards)
                tag = " (soft)" if soft else ""
                if hand.is_blackjack():
                    tag += " BLACKJACK"
                elif total > 21:
                    tag += " BUST"
                hand_str = f"[{cards}] = {total}{tag}"
            else:
                hand_str = "(no hand yet)"

            status = "OUT (busted bankroll)" if self.busted[a] else ""
            lines.append(
                f"{marker}{a}: bankroll=${bank:.0f}  bet={bet_str}  {hand_str}  {status}"
            )

        print("\n".join(lines))

    def _active(self):
        # return the active agents
        return [a for a in self.agents if not self.busted[a]]

    def _legal(self, agent):
        # returns the legal actions for agent
        if agent is None or self.busted.get(
            agent, False
        ):  # if the player doesn't exist or has run out of money (pane e cipolle...)
            return [0]  # dummy
        if self._phase == "bet":
            bank = self.bankrolls[agent]
            l = [
                i for i, lvl in enumerate(self.bet_levels) if lvl <= bank
            ]  # allowed bets
            return l if l else [0]
        if self._phase == "play" and self.current_agent == agent:
            return [len(self.bet_levels), len(self.bet_levels) + 1]  # hit or stand
        return [0]  # dummy

    def _start_round(self):
        self.round_num += 1
        self.hands = {}
        self.dealer_hand = Hand()
        self.dealer_hole_revealed = False
        self.bets = {}

        active = self._active()
        if (
            not active or self.round_num > self.max_rounds
        ):  # handle all busted or max round reached
            self._phase = "done"
            self.current_agent = None
            return

        self._phase = "bet"  # we start with bets
        self._turn_queue = list(active)  # everyone hast to bet yet
        self.current_agent = self._turn_queue.pop(0)

    def _deal(self):
        active = [a for a in self._active() if a in self.bets]
        for _ in range(2):  # two cards each
            for a in active:
                self._draw(self.hands.setdefault(a, Hand()), True)  # agent a card
            self._draw(self.dealer_hand, len(self.dealer_hand) == 0)  # dealer card

        self._phase = "play"  # change phase
        self._turn_queue = [
            a for a in active if not self.hands[a].is_blackjack()
        ]  # set turns, if you got blackjack you are done already
        if self._turn_queue:
            self.current_agent = self._turn_queue.pop(0)  # set current player
        else:
            self._resolve()  # if everyone got a blackjack (not sure if that will ever happen haha)

    def _draw(self, hand, reveal):
        c = self.shoe.draw()
        hand.add(c)
        if reveal:
            self.shoe_history.append(
                c
            )  # keep track of the game history if card can be shown now

    def _advance(self):
        if self._turn_queue:
            self.current_agent = self._turn_queue.pop(
                0
            )  # pass the turn to the next player
        else:
            self._resolve()  # everyone is done, resolve

    def _resolve(self):
        someone_alive = any(
            not self.hands[a].is_bust() for a in self.bets if a in self.hands
        )  # is there anyone still in the game?
        if someone_alive:
            if len(self.dealer_hand) >= 2 and not self.dealer_hole_revealed:
                self.shoe_history.append(
                    self.dealer_hand.cards[1]
                )  # show hidden dealer card (put here and not in _draw to keep correct order)
                self.dealer_hole_revealed = True
            while True:
                total, soft = self.dealer_hand.total()
                if total >= 17:
                    break
                self._draw(
                    self.dealer_hand, True
                )  # draw dealer card until total >= 17 as by bleckjack rules

        # dealer hand infos
        dtotal, _ = self.dealer_hand.total()
        dbust = dtotal > 21
        dbj = self.dealer_hand.is_blackjack()

        for a, bet in self.bets.items():
            hand = self.hands[a]
            if hand.is_bust():
                payout = -bet  # busted
            elif hand.is_blackjack():
                payout = (
                    0.0 if dbj else 1.5 * bet
                )  # if dealer got blackjack draw else get bonus win
            elif dbj:
                payout = -bet  # dealer got blackjack and you didn't :((
            elif dbust or hand.value() > dtotal:
                payout = bet  # you beat the dealer :)
            elif hand.value() < dtotal:
                payout = -bet  # dealer got a better hand :(
            else:
                payout = 0.0  # draw

            # handle bankroll, rewards and busted status
            self.bankrolls[a] += payout
            self.rewards[a] = self.rewards.get(a, 0.0) + payout
            if self.bankrolls[a] < self.bet_levels[0]:
                self.busted[a] = True

        self._just_resolved = True
        self.agents = [a for a in self.agents if not self.busted[a]]
        self.shoe.maybe_reshuffle()  # shuffle if neede
        self._start_round()  # set new round

    def _raw_obs(self, agent):
        if agent is None:
            agent = self.possible_agents[0]
        own_hand = self.hands.get(agent)
        own_hand = list(own_hand.cards) if own_hand else []  # set your hand
        other_hands = {
            a: list(self.hands[a].cards)
            for a in self.agents
            if a != agent and a in self.hands
        }  # set other players hands
        other_bets = {
            a: self.bets.get(a, 0.0) for a in self.agents if a != agent
        }  # set other players bets
        dealer_up = (
            self.dealer_hand.cards[0] if len(self.dealer_hand) > 0 else None
        )  # set dealr up card
        return {
            "phase": self._phase if self._phase != "done" else "bet",
            "round_num": self.round_num,
            "seat_id": self.possible_agents.index(agent),
            "num_seats": self.num_seats,
            "own_bankroll": self.bankrolls[agent],
            "own_bet": self.bets.get(agent, 0.0),
            "own_hand": own_hand,
            "dealer_upcard": dealer_up,
            "other_hands": other_hands,
            "other_bets": other_bets,
            "shoe_history": list(self.shoe_history),
            "cards_remaining_frac": self.shoe.remaining() / self.shoe.total_cards,
        }

    def _info(self):
        agent = (
            self.current_agent
            if self.current_agent is not None
            else self.possible_agents[0]
        )
        return {
            "agent": agent,
            "legal_actions": self._legal(agent),
            "shoe_history": list(self.shoe_history),
            "cards_remaining_frac": self.shoe.remaining() / self.shoe.total_cards,
            "phase": self._phase,
        }
