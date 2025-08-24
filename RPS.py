# RPS.py — drop-in solution for freeCodeCamp's Rock–Paper–Scissors project
# Strategy: ensemble of predictors (experts) + weighted-majority meta-controller.
# Predictors include:
#   - N-gram Markov (k=1..5) on opponent history
#   - Frequency counter on opponent moves
#   - Periodicity (cycle) detection on opponent moves
#   - Mirror/shift pattern detectors (exploits bots that react to our last moves)
#
# The meta-controller updates expert weights online based on prediction loss
# and selects actions from a small best-expert set to keep behavior stable.

from collections import defaultdict, deque
import random

ACTIONS = ["R", "P", "S"]
# What beats what
BEATS = {"R": "S", "P": "R", "S": "P"}
LOSES = {v: k for k, v in BEATS.items()}

def counter(move):
    """Return the move that beats `move`."""
    if move not in BEATS:
        return random.choice(ACTIONS)
    return LOSES[move]

def most_common(d):
    """Return key with max value; deterministic tie-break by R>P>S order."""
    if not d:
        return random.choice(ACTIONS)
    best = None
    bestv = -1
    for a in ACTIONS:
        v = d.get(a, 0)
        if v > bestv:
            bestv = v
            best = a
    return best

class Expert:
    """Base class for predictors."""
    def __init__(self, name):
        self.name = name
        self.weight = 1.0
        self.last_pred = None

    def predict_opp(self, opp_hist, our_hist):
        """Return predicted opponent next move (R/P/S) or None."""
        return None

    def on_result(self, opp_move, our_move):
        """Update internal state if needed after a round."""
        pass

    def loss(self, opp_move):
        """0 if predicted correctly, else 1 (for Weighted Majority)."""
        return 0 if (self.last_pred is not None and self.last_pred == opp_move) else 1

class NGramExpert(Expert):
    """Predict using opponent N-gram (Markov) model of order k."""
    def __init__(self, k):
        super().__init__(f"ngram{k}")
        self.k = k
        self.table = defaultdict(lambda: defaultdict(int))
        self.context = deque(maxlen=k)

    def predict_opp(self, opp_hist, our_hist):
        # build table incrementally
        if len(opp_hist) > self.k:
            ctx = tuple(opp_hist[-self.k:])
            # use table entry if exists
            counts = self.table.get(ctx, None)
            if counts:
                pred = most_common(counts)
            else:
                # back-off: try shorter contexts
                pred = None
                for j in range(self.k - 1, 0, -1):
                    ctx2 = tuple(opp_hist[-j:]) if j <= len(opp_hist) else None
                    if ctx2 and ("__len__" not in self.table):  # keep linter happy
                        # build a synthetic aggregate for shorter contexts
                        # (we didn't store shorter tables; approximate by scanning)
                        agg = defaultdict(int)
                        for (c, cnts) in self.table.items():
                            if len(c) >= j and c[-j:] == ctx2:
                                for m, cnt in cnts.items():
                                    agg[m] += cnt
                        if agg:
                            pred = most_common(agg)
                            break
                if pred is None:
                    # frequency fallback
                    pred = most_common({m: opp_hist.count(m) for m in ACTIONS})
        else:
            pred = None
        self.last_pred = pred
        return pred

    def on_result(self, opp_move, our_move):
        # update table with previous context -> next move
        # Only update if we have a full context of length k
        # Build context from history tracked externally is simpler, but we can
        # re-derive by keeping a running deque.
        if hasattr(self, "_recent_opp"):
            self._recent_opp.append(opp_move)
        else:
            self._recent_opp = deque([opp_move], maxlen=self.k + 1)

        if len(self._recent_opp) == self.k + 1:
            ctx = tuple(list(self._recent_opp)[:-1])
            nxt = self._recent_opp[-1]
            self.table[ctx][nxt] += 1

class FreqExpert(Expert):
    """Predict opponent's most frequent move so far."""
    def __init__(self):
        super().__init__("freq")
        self.counts = defaultdict(int)

    def predict_opp(self, opp_hist, our_hist):
        if self.counts:
            pred = most_common(self.counts)
        else:
            pred = None
        self.last_pred = pred
        return pred

    def on_result(self, opp_move, our_move):
        self.counts[opp_move] += 1

class CycleExpert(Expert):
    """Detect periodic patterns in opponent moves and predict next."""
    def __init__(self, min_period=2, max_period=12):
        super().__init__("cycle")
        self.min_p = min_period
        self.max_p = max_period

    def predict_opp(self, opp_hist, our_hist):
        n = len(opp_hist)
        pred = None
        # Try smallest plausible period first to lock onto short cycles quickly
        for p in range(self.min_p, min(self.max_p, max(2, n // 2)) + 1):
            ok = True
            for i in range(n - p):
                if opp_hist[i] != opp_hist[i + p]:
                    ok = False
                    break
            if ok and n >= p:
                pred = opp_hist[-p]  # the pattern repeats
                break
        self.last_pred = pred
        return pred

class MirrorShiftExpert(Expert):
    """
    Detect if opponent mirrors or shifts our previous move deterministically.
    Two checks over a sliding recent window:
      - Mirror: opp[t] == our[t-1]
      - Shift (beat-last): opp[t] == counter(our[t-1])
    If confidence is high, predict accordingly.
    """
    def __init__(self, window=20):
        super().__init__("mirror_shift")
        self.window = window
        self.mirror_hits = 0
        self.shift_hits = 0
        self.total = 0
        self._our_hist = []
        self._opp_hist = []

    def predict_opp(self, opp_hist, our_hist):
        self._our_hist = our_hist
        self._opp_hist = opp_hist
        pred = None
        if len(our_hist) >= 1 and len(opp_hist) >= 1:
            # evaluate recent window
            w = min(self.window, len(our_hist))
            mirror = 0
            shift = 0
            denom = 0
            for t in range(1, w + 1):
                if t <= len(opp_hist) and t <= len(our_hist):
                    denom += 1
                    if opp_hist[-t] == our_hist[-t - 0]:  # opp[t] vs our[t]
                        mirror += 1
                    if opp_hist[-t] == counter(our_hist[-t]):
                        shift += 1
            # thresholds: >70% consistency to commit
            if denom >= 5:
                if mirror / denom > 0.7:
                    pred = our_hist[-1]  # they mirrored last time -> they will mirror our last again
                elif shift / denom > 0.7:
                    pred = counter(our_hist[-1])  # they tend to play counter(our_last)
        self.last_pred = pred
        return pred

class MetaController:
    """Weighted Majority over experts with exploration and tie-handling."""
    def __init__(self, experts, eta=0.25, explore_prob=0.02, top_k=2):
        self.experts = experts
        self.eta = eta            # learning rate for Hedge update
        self.explore_prob = explore_prob
        self.top_k = top_k
        self.round = 0

    def choose_move(self, opp_hist, our_hist):
        self.round += 1
        # Collect expert predictions
        preds = []
        for e in self.experts:
            p = e.predict_opp(opp_hist, our_hist)
            preds.append((e, p))

        # If any explicit prediction exists, score them by weights
        votes = defaultdict(float)
        for e, p in preds:
            if p in ACTIONS:
                votes[p] += e.weight

        if votes:
            # pick among top_k by cumulative weight, then counter it
            sorted_moves = sorted(ACTIONS, key=lambda m: votes.get(m, 0.0), reverse=True)
            candidate = sorted_moves[0]
            if self.top_k > 1 and len(sorted_moves) > 1:
                # small randomness among top-k to prevent being farmed
                k = min(self.top_k, len(sorted_moves))
                candidate = random.choice(sorted_moves[:k])

            # tiny exploration to avoid lock-in
            if random.random() < self.explore_prob:
                candidate = random.choice(ACTIONS)

            # we want to BEAT predicted opponent move
            return counter(candidate)
        else:
            # cold start: RPS opening book (balanced) with light bias toward P
            opening = ["P", "R", "S", "P", "R", "P"]
            if len(our_hist) < len(opening):
                return opening[len(our_hist)]
            return random.choice(ACTIONS)

    def update(self, opp_move, our_move):
        # update expert internal states and weights via multiplicative update
        for e in self.experts:
            e.on_result(opp_move, our_move)
            l = e.loss(opp_move)  # 0 or 1
            e.weight *= (self.eta ** l)

# ---- Global player state (persist across function calls) ----
def _state(reset=False):
    if reset or not hasattr(_state, "initialized"):
        _state.initialized = True
        _state.opp_hist = []
        _state.our_hist = []
        # Create an expert ensemble
        experts = [
            FreqExpert(),
            CycleExpert(min_period=2, max_period=10),
            MirrorShiftExpert(window=25),
        ]
        # Add a ladder of N-gram experts
        for k in range(1, 6):
            experts.append(NGramExpert(k))
        _state.meta = MetaController(experts, eta=0.25, explore_prob=0.015, top_k=2)
    return _state

# ---- Required entry point ----
def player(prev_play, opponent_history=[]):
    """
    freeCodeCamp calls this function once per round:
      - prev_play: opponent's last move ("R","P","S") or "" on first round
      - return: our next move ("R","P","S")
    """
    st = _state()

    # Handle reset between different play() calls (FCC test harness behavior):
    # When a new match starts, prev_play == "" and opponent_history resets externally.
    if prev_play == "" and len(opponent_history) == 0 and len(st.opp_hist) > 0:
        st = _state(reset=True)

    # Update histories if a real previous move exists
    if prev_play in ACTIONS:
        st.opp_hist.append(prev_play)

    # Decide our next move using the meta-controller
    move = st.meta.choose_move(st.opp_hist, st.our_hist)

    # Append to our history and notify experts about the round result
    if prev_play in ACTIONS:
        # We now know opp's last move AND we just chose ours for the next round.
        # For learning, attribute last result with our last played move if exists.
        # If no our move yet (first round), skip update.
        if st.our_hist:
            st.meta.update(prev_play, st.our_hist[-1])

    st.our_hist.append(move)

    # Maintain the optional external list for FCC (they pass and reuse this)
    opponent_history.append(prev_play if prev_play in ACTIONS else "")

    return move
