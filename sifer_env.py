"""
sifer_env.py
============
SiferTrustEnv — OpenEnv-compliant environment simulating a Level-1
Trust & Safety (Fraud) Analyst role for an e-commerce platform.

The agent reads synthetic platform logs and must detect and mitigate:
  Task 1 (Easy)   — Promo code abuse via bulk account creation from one IP
  Task 2 (Medium) — Coordinated fake review bombing with identical text
  Task 3 (Hard)   — Sneaker scalping with fuzzy address obfuscation

OpenEnv interface implemented:
    reset(task_level)  → Observation
    step(action)       → StepResult
    state()            → dict

All models are typed Pydantic v2 models.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# OPENENV-CORE BASE CLASS
# Import the official OpenEnv base if openenv-core is installed.
# Falls back to a plain object so the env still works standalone.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.interfaces import Environment as _OpenEnvBase
except ImportError:
    class _OpenEnvBase:  # type: ignore[no-redef]
        """Minimal no-op base used when openenv-core is not installed."""
        pass

# ---------------------------------------------------------------------------
# ─── CONSTANTS ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Legitimate background IPs and users used across all tasks as noise
LEGIT_IPS = [
    "203.0.113.10", "203.0.113.11", "203.0.113.12",
    "198.51.100.5",  "198.51.100.6",  "198.51.100.7",
    "192.0.2.20",    "192.0.2.21",    "192.0.2.22",
]
LEGIT_USERS = [f"user_{i:04d}" for i in range(1, 21)]

# ── Task 1 — Promo Code Abuse ─────────────────────────────────────────────
ABUSER_IP        = "10.0.0.77"
PROMO_CODE       = "NEWUSER50"
ABUSER_ACCOUNTS  = [f"abuser_{i:03d}" for i in range(1, 16)]   # 15 fake accounts
# Legitimate decoy — one real user who used the same promo code once
LEGIT_PROMO_USER = "user_0042"
LEGIT_PROMO_IP   = "203.0.113.50"

# ── Task 2 — Review Bombing ───────────────────────────────────────────────
BOT_REVIEW_TEXT = "Wow great item highly recommend!"
BOT_REVIEWERS   = [f"botreview_{i:03d}" for i in range(1, 51)]  # 50 bot accounts
TARGET_PRODUCT  = "PROD_9001"
LEGIT_REVIEWERS = [f"realuser_{i:03d}" for i in range(1, 6)]    # 5 real reviewers

# ── Task 3 — Sneaker Scalping ─────────────────────────────────────────────
SCALPER_ACCOUNTS = [f"scalper_{i:03d}" for i in range(1, 11)]
SCALPER_IPS      = [f"172.16.{i}.1" for i in range(1, 11)]
SCALPER_ORDERS   = [f"ORD_SC_{i:04d}" for i in range(1, 11)]
HOT_PRODUCT      = "PROD_DROP_001"

# 10 slightly different strings for the same physical address — the core challenge
SCALPER_ADDRESSES = [
    "123 Main St, Apt 4",
    "123 Main Street, #4",
    "123 Main st. (Unit 4)",
    "123 main st apt 4",
    "123 Main St Apt. 4",
    "123 Main Street Apt 4",
    "123 Main St, Unit #4",
    "123 main street, apt #4",
    "123 Main St., Apartment 4",
    "123 MAIN ST APT 4",
]


# ---------------------------------------------------------------------------
# ─── PYDANTIC MODELS — OBSERVATION ──────────────────────────────────────────
# ---------------------------------------------------------------------------

class PlatformEvent(BaseModel):
    """A single entry in the platform event log."""
    timestamp:   str            = Field(..., description="ISO-8601 datetime string")
    ip_address:  str            = Field(..., description="Source IP of the request")
    event_type:  str            = Field(..., description="account_created | review_posted | checkout_completed | page_view | login")
    user_id:     Optional[str]  = Field(None, description="User performing the action")
    order_id:    Optional[str]  = Field(None, description="Order ID for checkout events")
    details:     Dict[str, Any] = Field(default_factory=dict,
                                        description="Payload: promo_code, review_text, shipping_address, product_id, etc.")


class Observation(BaseModel):
    """Everything the agent sees at each timestep."""
    task_level:       int                 = Field(..., description="1=Easy 2=Medium 3=Hard")
    task_desc:        str                 = Field(..., description="Plain-language objective")
    events:           List[PlatformEvent] = Field(..., description="Recent platform log entries")
    revoked_ips:      List[str]           = Field(default_factory=list)
    deleted_reviews:  List[str]           = Field(default_factory=list,
                                                   description="user_ids whose reviews were deleted")
    cancelled_orders: List[str]           = Field(default_factory=list)
    step_count:       int                 = Field(0)


# ---------------------------------------------------------------------------
# ─── PYDANTIC MODELS — ACTIONS ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

class RevokeOrders(BaseModel):
    """Revoke all orders placed from a specific IP address."""
    action_type: Literal["RevokeOrders"] = "RevokeOrders"
    ip_address:  str = Field(..., description="IP whose orders should be revoked")


class DeleteBotReviews(BaseModel):
    """Delete reviews posted by a list of bot user_ids."""
    action_type: Literal["DeleteBotReviews"] = "DeleteBotReviews"
    user_ids:    List[str] = Field(..., description="user_ids whose reviews must be deleted")


class CancelOrders(BaseModel):
    """Cancel a specific list of order_ids."""
    action_type: Literal["CancelOrders"] = "CancelOrders"
    order_ids:   List[str] = Field(..., description="order_ids to cancel")


class Pass(BaseModel):
    """Do nothing — advance the clock by one step."""
    action_type: Literal["Pass"] = "Pass"


# Union type covering the full action space
Action = Union[RevokeOrders, DeleteBotReviews, CancelOrders, Pass]


# ---------------------------------------------------------------------------
# ─── PYDANTIC MODELS — REWARD & STEP RESULT ─────────────────────────────────
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value:     float            = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback:  str              = Field("")


class StepResult(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ─── SYNTHETIC LOG GENERATORS ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _ts(base: datetime, delta_seconds: float) -> str:
    """Return ISO timestamp offset from base time."""
    return (base + timedelta(seconds=delta_seconds)).isoformat()


def _make_legit_traffic(base: datetime, n: int = 20) -> List[PlatformEvent]:
    """Generate n realistic background events from legitimate IPs/users."""
    events: List[PlatformEvent] = []
    etypes = ["login", "page_view", "checkout_completed", "review_posted"]
    for _ in range(n):
        ip  = random.choice(LEGIT_IPS)
        uid = random.choice(LEGIT_USERS)
        et  = random.choice(etypes)
        det: Dict[str, Any] = {"session": str(uuid.uuid4())[:8]}
        oid = None
        if et == "checkout_completed":
            oid = f"ORD_{str(uuid.uuid4())[:8].upper()}"
            det["product_id"]       = f"PROD_{random.randint(100, 800):04d}"
            det["shipping_address"] = (
                f"{random.randint(1, 999)} {random.choice(['Oak','Elm','Pine','Maple'])} Ave, "
                f"City {random.randint(1, 50)}"
            )
        elif et == "review_posted":
            det["product_id"]  = f"PROD_{random.randint(100, 800):04d}"
            det["review_text"] = random.choice([
                "Really happy with this purchase, fast delivery.",
                "Good quality, would buy again.",
                "Arrived on time and exactly as described.",
                "Decent product for the price point.",
                "Packaging was damaged but item is fine.",
            ])
            det["rating"] = random.randint(3, 5)
        events.append(PlatformEvent(
            timestamp  = _ts(base, random.randint(0, 3500)),
            ip_address = ip,
            event_type = et,
            user_id    = uid,
            order_id   = oid,
            details    = det,
        ))
    return events


def _generate_task1_logs(base: datetime) -> List[PlatformEvent]:
    """
    Task 1 — Promo Code Abuse.
    ABUSER_IP creates 15 accounts in ~2 minutes; each immediately checks out
    with NEWUSER50. One legitimate user also uses the promo once (decoy).
    """
    events = _make_legit_traffic(base, n=20)

    # 15 fake account creations + immediate promo checkouts — all from ABUSER_IP
    for idx, uid in enumerate(ABUSER_ACCOUNTS):
        t = idx * 8   # one account every 8 seconds → 15 accounts in 120 s
        events.append(PlatformEvent(
            timestamp  = _ts(base, t),
            ip_address = ABUSER_IP,
            event_type = "account_created",
            user_id    = uid,
            details    = {"email": f"{uid}@tempmail.xyz"},
        ))
        events.append(PlatformEvent(
            timestamp  = _ts(base, t + 3),
            ip_address = ABUSER_IP,
            event_type = "checkout_completed",
            user_id    = uid,
            order_id   = f"ORD_AB_{idx+1:04d}",
            details    = {
                "promo_code":       PROMO_CODE,
                "product_id":       "PROD_0055",
                "shipping_address": f"Fake Address {idx + 1}",
            },
        ))

    # Decoy: one legitimate user creates one account and uses the promo once
    events.append(PlatformEvent(
        timestamp  = _ts(base, 500),
        ip_address = LEGIT_PROMO_IP,
        event_type = "account_created",
        user_id    = LEGIT_PROMO_USER,
        details    = {"email": "real_customer@gmail.com"},
    ))
    events.append(PlatformEvent(
        timestamp  = _ts(base, 560),
        ip_address = LEGIT_PROMO_IP,
        event_type = "checkout_completed",
        user_id    = LEGIT_PROMO_USER,
        order_id   = "ORD_LEGIT_0001",
        details    = {
            "promo_code":       PROMO_CODE,
            "product_id":       "PROD_0055",
            "shipping_address": "99 Real Street, Springfield",
        },
    ))

    random.shuffle(events)
    return events


def _generate_task2_logs(base: datetime) -> List[PlatformEvent]:
    """
    Task 2 — Review Bombing.
    50 bot accounts post the exact same review_text within 3 minutes.
    5 legitimate reviewers post unique reviews on the same product (decoys).
    """
    events = _make_legit_traffic(base, n=25)

    # 50 bot reviews — identical text, one every 3 seconds
    for idx, uid in enumerate(BOT_REVIEWERS):
        events.append(PlatformEvent(
            timestamp  = _ts(base, idx * 3),
            ip_address = f"10.1.{idx // 10}.{idx % 10 + 1}",
            event_type = "review_posted",
            user_id    = uid,
            details    = {
                "product_id":  TARGET_PRODUCT,
                "review_text": BOT_REVIEW_TEXT,
                "rating":      5,
            },
        ))

    # 5 legitimate reviewers — each with a unique personal review
    unique_reviews = [
        "I've been using this for two weeks and it's held up really well.",
        "The colour looked different on screen but I'm still happy with it.",
        "Bit pricey but the build quality justifies the cost in my opinion.",
        "Ordered two, one was slightly off but customer service sorted it.",
        "My third purchase from this brand. Consistent quality as always.",
    ]
    for idx, uid in enumerate(LEGIT_REVIEWERS):
        events.append(PlatformEvent(
            timestamp  = _ts(base, 200 + idx * 45),
            ip_address = random.choice(LEGIT_IPS),
            event_type = "review_posted",
            user_id    = uid,
            details    = {
                "product_id":  TARGET_PRODUCT,
                "review_text": unique_reviews[idx],
                "rating":      random.choice([4, 5]),
            },
        ))

    random.shuffle(events)
    return events


def _generate_task3_logs(base: datetime) -> List[PlatformEvent]:
    """
    Task 3 — Sneaker Scalping with fuzzy address obfuscation.
    10 scalper accounts on 10 distinct IPs all checkout within 0.4 seconds.
    Each uses a slightly different string for the same physical address.
    60+ legitimate orders at genuinely different addresses provide noise.
    """
    events = _make_legit_traffic(base, n=60)

    # All scalper checkouts clustered within a 0.4-second burst
    BURST_OFFSET = 300.0   # 5 minutes into the log window
    for idx, (uid, ip, oid, addr) in enumerate(
        zip(SCALPER_ACCOUNTS, SCALPER_IPS, SCALPER_ORDERS, SCALPER_ADDRESSES)
    ):
        offset = BURST_OFFSET + (idx * 0.04)   # 40 ms apart → all within 0.36 s
        events.append(PlatformEvent(
            timestamp  = _ts(base, offset),
            ip_address = ip,
            event_type = "checkout_completed",
            user_id    = uid,
            order_id   = oid,
            details    = {
                "product_id":           HOT_PRODUCT,
                "shipping_address":     addr,
                "checkout_duration_ms": random.randint(80, 490),
            },
        ))

    random.shuffle(events)
    return events


# ---------------------------------------------------------------------------
# ─── TASK DESCRIPTIONS ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    1: (
        "EASY — Promo Code Abuse: "
        "One IP address created 15 brand-new accounts within 2 minutes and "
        "immediately applied promo code 'NEWUSER50' on every checkout. "
        "Issue a RevokeOrders action for that specific IP. "
        "Do NOT revoke orders from the legitimate user who used the promo once."
    ),
    2: (
        "MEDIUM — Review Bombing: "
        "A newly launched product received 50 five-star reviews in under 3 minutes "
        "where every single review_text is identical. "
        "Issue a DeleteBotReviews action listing all bot user_ids. "
        "Do NOT delete reviews from users who wrote unique, genuine review text."
    ),
    3: (
        "HARD — Sneaker Scalping (Fuzzy Address Matching): "
        "A high-demand product drop just occurred. 10 different accounts on 10 "
        "different IPs all completed checkout within 0.5 seconds of each other. "
        "Their shipping_address strings look different but all refer to the same "
        "physical location (e.g. '123 Main St Apt 4' vs '123 main street, apt #4'). "
        "Issue a CancelOrders action with all 10 scalper order_ids. "
        "Do NOT cancel orders shipping to genuinely different addresses."
    ),
}


# ---------------------------------------------------------------------------
# ─── ENVIRONMENT ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class SiferTrustEnv(_OpenEnvBase):
    """
    OpenEnv-compliant Trust & Safety fraud analyst simulation.

    Usage
    -----
        env = SiferTrustEnv()
        obs = env.reset(task_level=1)
        result = env.step(RevokeOrders(ip_address="10.0.0.77"))
        print(result.reward)
    """

    MAX_STEPS = 10

    def __init__(self, seed: int = 42) -> None:
        self._seed              = seed
        self._task_level        = 1
        self._events:           List[PlatformEvent] = []
        self._revoked_ips:      List[str] = []
        self._deleted_reviews:  List[str] = []
        self._cancelled_orders: List[str] = []
        self._step_count        = 0
        self._done              = False
        self._base_time         = datetime(2024, 6, 1, 0, 0, 0)

    # ------------------------------------------------------------------
    def reset(self, task_level: int = 1) -> Observation:
        """Reset environment for a given task level. Returns initial Observation."""
        if task_level not in (1, 2, 3):
            raise ValueError(f"task_level must be 1, 2 or 3 — got {task_level}")
        random.seed(self._seed)
        self._task_level        = task_level
        self._revoked_ips       = []
        self._deleted_reviews   = []
        self._cancelled_orders  = []
        self._step_count        = 0
        self._done              = False
        self._base_time         = datetime(2024, 6, 1, 0, 0, 0)

        generators = {
            1: _generate_task1_logs,
            2: _generate_task2_logs,
            3: _generate_task3_logs,
        }
        self._events = generators[task_level](self._base_time)
        return self._build_obs()

    # ------------------------------------------------------------------
    def step(self, action: Action) -> StepResult:
        """Apply an action and advance the environment by one timestep."""
        if self._done:
            raise RuntimeError("Episode finished — call reset() first.")
        self._step_count += 1
        reward, info = self._compute_reward(action)
        self._apply_action(action)
        decisive     = not isinstance(action, Pass)
        self._done   = decisive or (self._step_count >= self.MAX_STEPS)
        return StepResult(
            observation=self._build_obs(), reward=reward, done=self._done, info=info
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        """Return full internal state as a plain dict (for debugging / validator)."""
        return {
            "task_level":       self._task_level,
            "step_count":       self._step_count,
            "done":             self._done,
            "revoked_ips":      self._revoked_ips,
            "deleted_reviews":  self._deleted_reviews,
            "cancelled_orders": self._cancelled_orders,
            "num_events":       len(self._events),
        }

    # ------------------------------------------------------------------
    def _build_obs(self) -> Observation:
        return Observation(
            task_level       = self._task_level,
            task_desc        = TASK_DESCRIPTIONS[self._task_level],
            events           = self._events,
            revoked_ips      = list(self._revoked_ips),
            deleted_reviews  = list(self._deleted_reviews),
            cancelled_orders = list(self._cancelled_orders),
            step_count       = self._step_count,
        )

    def _apply_action(self, action: Action) -> None:
        if isinstance(action, RevokeOrders):
            if action.ip_address not in self._revoked_ips:
                self._revoked_ips.append(action.ip_address)
        elif isinstance(action, DeleteBotReviews):
            for uid in action.user_ids:
                if uid not in self._deleted_reviews:
                    self._deleted_reviews.append(uid)
        elif isinstance(action, CancelOrders):
            for oid in action.order_ids:
                if oid not in self._cancelled_orders:
                    self._cancelled_orders.append(oid)

    def _compute_reward(self, action: Action) -> tuple[Reward, Dict[str, Any]]:
        return {1: self._grade_task1, 2: self._grade_task2, 3: self._grade_task3}[
            self._task_level
        ](action)

    # ------------------------------------------------------------------
    # Task 1 Grader
    # ------------------------------------------------------------------
    def _grade_task1(self, action: Action) -> tuple[Reward, Dict[str, Any]]:
        if isinstance(action, RevokeOrders):
            if action.ip_address == ABUSER_IP:
                return Reward(value=1.0, breakdown={"correct": 1.0},
                              feedback=f"✅ Correct! Revoked all promo-abuse orders from {ABUSER_IP}."), {"correct": True}
            elif action.ip_address == LEGIT_PROMO_IP:
                return Reward(value=-1.0, breakdown={"false_positive": -1.0},
                              feedback=f"❌ Wrong! {LEGIT_PROMO_IP} is a real customer who used the promo once."), {"correct": False}
            else:
                return Reward(value=-0.2, breakdown={"unknown_ip": -0.2},
                              feedback=f"⚠️ {action.ip_address} is not the abuser IP."), {"correct": False}
        elif isinstance(action, Pass):
            return Reward(value=0.0, breakdown={"pass": 0.0},
                          feedback="ℹ️ Look for one IP with 15+ account_created events within 2 min, each followed by a promo checkout."), {"correct": False}
        else:
            return Reward(value=-0.5, breakdown={"wrong_type": -0.5},
                          feedback="⚠️ Wrong action type. Use RevokeOrders."), {"correct": False}

    # ------------------------------------------------------------------
    # Task 2 Grader
    # ------------------------------------------------------------------
    def _grade_task2(self, action: Action) -> tuple[Reward, Dict[str, Any]]:
        if isinstance(action, DeleteBotReviews):
            submitted = set(action.user_ids)
            bots      = set(BOT_REVIEWERS)
            legit_set = set(LEGIT_REVIEWERS)
            false_pos = submitted & legit_set
            true_pos  = submitted & bots

            if false_pos:
                return Reward(value=-1.0, breakdown={"false_positive": -1.0},
                              feedback=f"❌ Deleted legit review(s) from: {false_pos}. Only delete identical-text reviews."), {"correct": False}
            if true_pos == bots:
                return Reward(value=1.0, breakdown={"full_delete": 1.0},
                              feedback=f"✅ Perfect! All {len(bots)} bot reviews deleted."), {"correct": True}
            if len(true_pos) >= 25:
                return Reward(value=0.5, breakdown={"partial": 0.5},
                              feedback=f"⚠️ Partial — deleted {len(true_pos)}/{len(bots)} bot reviews."), {"correct": False}
            return Reward(value=-0.2, breakdown={"insufficient": -0.2},
                          feedback=f"❌ Only {len(true_pos)}/{len(bots)} identified."), {"correct": False}
        elif isinstance(action, Pass):
            return Reward(value=0.0, breakdown={"pass": 0.0},
                          feedback="ℹ️ Look for review_text that repeats identically across many users on the same product_id."), {"correct": False}
        else:
            return Reward(value=-0.5, breakdown={"wrong_type": -0.5},
                          feedback="⚠️ Wrong action type. Use DeleteBotReviews."), {"correct": False}

    # ------------------------------------------------------------------
    # Task 3 Grader
    # ------------------------------------------------------------------
    def _grade_task3(self, action: Action) -> tuple[Reward, Dict[str, Any]]:
        if isinstance(action, CancelOrders):
            submitted   = set(action.order_ids)
            scalper_set = set(SCALPER_ORDERS)
            legit_orders = {
                e.order_id for e in self._events
                if e.order_id and not e.order_id.startswith("ORD_SC_")
            }
            false_pos = submitted & legit_orders
            true_pos  = submitted & scalper_set

            if false_pos:
                return Reward(value=-1.0, breakdown={"false_positive": -1.0},
                              feedback=f"❌ Cancelled legit order(s): {false_pos}."), {"correct": False}
            if true_pos == scalper_set:
                return Reward(value=1.0, breakdown={"full_cancel": 1.0},
                              feedback=f"✅ Perfect! All {len(scalper_set)} scalper orders cancelled."), {"correct": True}
            if len(true_pos) >= 5:
                return Reward(value=0.5, breakdown={"partial": 0.5},
                              feedback=f"⚠️ Partial — cancelled {len(true_pos)}/{len(scalper_set)} scalper orders."), {"correct": False}
            return Reward(value=-0.2, breakdown={"insufficient": -0.2},
                          feedback=f"❌ Only {len(true_pos)}/{len(scalper_set)} scalper orders found."), {"correct": False}
        elif isinstance(action, Pass):
            return Reward(value=0.0, breakdown={"pass": 0.0},
                          feedback="ℹ️ Look for checkouts on the same product within 0.5 s with fuzzy-matching shipping addresses."), {"correct": False}
        else:
            return Reward(value=-0.5, breakdown={"wrong_type": -0.5},
                          feedback="⚠️ Wrong action type. Use CancelOrders."), {"correct": False}


# ---------------------------------------------------------------------------
# ─── SMOKE TEST ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = SiferTrustEnv(seed=42)

    oracle: Dict[int, Action] = {
        1: RevokeOrders(ip_address=ABUSER_IP),
        2: DeleteBotReviews(user_ids=BOT_REVIEWERS),
        3: CancelOrders(order_ids=SCALPER_ORDERS),
    }

    for level in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"  TASK {level}")
        print(f"{'='*60}")
        obs = env.reset(task_level=level)
        print(f"  Events    : {len(obs.events)}")
        print(f"  Task desc : {obs.task_desc[:80]}…")
        result = env.step(oracle[level])
        print(f"  Reward    : {result.reward.value}")
        print(f"  Feedback  : {result.reward.feedback}")
        print(f"  Done      : {result.done}")
        print(f"  State     : {env.state()}")