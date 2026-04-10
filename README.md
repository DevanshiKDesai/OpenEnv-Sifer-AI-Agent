---
title: SiferTrustEnv
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - trust-and-safety
  - fraud-detection
  - reinforcement-learning
---

# SiferTrustEnv

**OpenEnv-compliant environment — Scaler x OpenEnv Hackathon submission**

Simulates a Level-1 Trust & Safety (Fraud) Analyst role at an e-commerce platform. The agent reads synthetic platform logs and must detect and mitigate fraud, botting, and scalping without disrupting legitimate shoppers.

---

## Motivation

Trust & Safety teams at e-commerce companies review thousands of events per hour to catch promo abusers, fake reviewers, and scalper bots. This is an ideal RL/agent evaluation domain because:

- Attacks are hidden inside realistic legitimate traffic
- False positives have a real business cost (banning real customers)
- Each task requires a different type of pattern recognition
- It maps directly to work done by real fraud analysts every day

---

## Project Structure

```
SiferTrustEnv/
├── sifer_env.py       # Core environment — Pydantic models, log generators, graders
├── inference.py       # Baseline inference script (OpenAI-compatible client)
├── server.py          # FastAPI HTTP server for HF Spaces
├── server/
│   └── app.py         # FastAPI app entry point (openenv validate requirement)
├── openenv.yaml       # OpenEnv metadata configuration
├── pyproject.toml     # Package configuration and entry points
├── uv.lock            # Dependency lock file
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Observation Space

Each observation is a JSON object with the following top-level fields:

| Field | Type | Description |
|---|---|---|
| `task_level` | `int` | Current task: 1=Easy, 2=Medium, 3=Hard |
| `task_desc` | `str` | Plain-language description of the objective |
| `events` | `List[PlatformEvent]` | Recent platform log entries |
| `revoked_ips` | `List[str]` | IPs already revoked this episode |
| `deleted_reviews` | `List[str]` | user_ids whose reviews were deleted |
| `cancelled_orders` | `List[str]` | order_ids cancelled this episode |
| `step_count` | `int` | Steps taken so far |

Each `PlatformEvent` contains:

| Field | Type | Description |
|---|---|---|
| `timestamp` | `str` | ISO-8601 datetime |
| `ip_address` | `str` | Source IP of the request |
| `event_type` | `str` | One of: `account_created`, `review_posted`, `checkout_completed`, `page_view`, `login` |
| `user_id` | `str` (nullable) | Acting user |
| `order_id` | `str` (nullable) | Order ID for checkout events |
| `details` | `dict` | Extra payload: `promo_code`, `review_text`, `shipping_address`, `product_id`, etc. |

---

## Action Space

The agent must output exactly one JSON action per step:

```json
{"action_type": "RevokeOrders", "ip_address": "10.0.0.77"}
```

```json
{"action_type": "DeleteBotReviews", "user_ids": ["botreview_001", "botreview_002"]}
```

```json
{"action_type": "CancelOrders", "order_ids": ["ORD_SC_0001", "ORD_SC_0002"]}
```

```json
{"action_type": "Pass"}
```

---

## Tasks

### Task 1 — Easy: Promo Code Abuse

One IP address creates 15 brand-new accounts within 2 minutes and immediately checks out with promo code `NEWUSER50` on every order. A decoy legitimate user also uses the promo once. The agent must issue `RevokeOrders` for the abuser IP without touching the real customer.

**Reward:** +1.0 correct IP · -1.0 false positive (legit user's IP) · -0.2 unknown IP · -0.5 wrong action type · 0.0 Pass

### Task 2 — Medium: Review Bombing

A newly launched product receives 50 five-star reviews in under 3 minutes. Every bot review has identical text: `"Wow great item highly recommend!"`. Five legitimate reviewers wrote unique reviews on the same product. The agent must issue `DeleteBotReviews` for all 50 bot user_ids without deleting any genuine review.

**Reward:** +1.0 all 50 bots · +0.5 at least 25 bots with no false positives · -1.0 deleted a legit review · -0.2 too few identified

### Task 3 — Hard: Sneaker Scalping with Fuzzy Address Matching

A high-demand product drop occurs. 10 accounts on 10 different IPs all complete checkout within 0.5 seconds of each other. Each uses a slightly different string for the same physical address — for example `"123 Main St, Apt 4"` vs `"123 MAIN ST APT 4"` vs `"123 Main Street, #4"`. Over 60 legitimate orders with genuinely different addresses provide noise. The agent must issue `CancelOrders` for all 10 scalper order_ids.

**Reward:** +1.0 all 10 orders · +0.5 at least 5 with no false positives · -1.0 cancelled a legit order · -0.2 too few found

---

## Reward Function

| Value | Meaning |
|---|---|
| +1.0 | Fully correct decisive action |
| +0.5 | Partial success (Tasks 2 and 3 only) |
| 0.0 | Pass — no progress, no penalty |
| -0.2 | Incorrect but non-destructive action |
| -0.5 | Wrong action type for the task |
| -1.0 | Destructive false positive — hurt a legitimate user |

---

## Setup and Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Smoke test (no LLM needed)

```bash
python sifer_env.py
# Expected: Reward 1.0 on all 3 tasks
```

### Run inference

```bash
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
python inference.py
```

### Run HTTP server locally

```bash
python server.py
# Swagger UI available at: http://localhost:7860/docs
```

### Docker

```bash
docker build -t sifer-trust-env .

docker run --rm -p 7860:7860 \
  -e HF_TOKEN="hf_..." \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  sifer-trust-env
```

---

## HTTP API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — returns `{"status": "ok"}` |
| POST | `/reset` | Body: `{"task_level": 1}` — returns initial Observation |
| POST | `/step` | Body: `{"action": {...}}` — returns StepResult with reward |
| GET | `/state` | Returns current internal environment state |
| POST | `/run` | Executes inference.py and returns stdout |

Interactive Swagger docs available at `/docs`.

---

## Baseline Scores

All scores produced using the rule-based fallback analyser (no LLM quota required).

| Task | Difficulty | Score | Success |
|---|---|---|---|
| Promo Code Abuse | Easy | 1.0 | true |
| Review Bombing | Medium | 1.0 | true |
| Sneaker Scalping | Hard | 1.0 | true |

Scores are deterministic with `seed=42`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | Hugging Face API key — set as Space Secret, never hardcode |
| `API_BASE_URL` | Yes | OpenAI-compatible API endpoint |
| `MODEL_NAME` | Yes | Model identifier for inference |

---

## License

MIT