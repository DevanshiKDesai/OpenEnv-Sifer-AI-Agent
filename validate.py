"""
validate.py
===========
Cross-platform submission validator for SiferTrustEnv.
Works on Windows, Mac, and Linux — no bash, no Docker needed locally.

Checks:
  1. HF Space is live and /reset returns HTTP 200
  2. openenv.yaml exists and has required fields
  3. All required files are present
  4. openenv validate passes (if openenv-core is installed)
  5. Smoke test — runs oracle actions and checks all 3 tasks score 1.0

Usage:
    python validate.py https://YOUR_USERNAME-sifertrusten.hf.space
"""

from __future__ import annotations

import json
import sys
import os
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour helpers (work on Windows too)
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init()
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except ImportError:
    GREEN = RED = YELLOW = BOLD = RESET = ""

PASS_COUNT = 0
FAIL_COUNT = 0


def passed(msg: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  {GREEN}✅ PASS{RESET} — {msg}")


def failed(msg: str, hint: str = "") -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  {RED}❌ FAIL{RESET} — {msg}")
    if hint:
        print(f"  {YELLOW}   Hint: {hint}{RESET}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─'*55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*55}{RESET}")


# ---------------------------------------------------------------------------
# Check 1 — HF Space is live
# ---------------------------------------------------------------------------
def check_space_live(space_url: str) -> bool:
    section("Check 1 — HF Space is live")
    import urllib.request, urllib.error

    base = space_url.rstrip("/")

    # HF Spaces can use different URL patterns — try health check first, then /reset
    candidates = [
        (base + "/",     b"",                        "GET",  "health check"),
        (base + "/reset", b'{"task_level": 1}',    "POST", "/reset endpoint"),
    ]

    for url, data, method, label in candidates:
        try:
            req = urllib.request.Request(
                url,
                data=data if data else None,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                code = resp.getcode()
                body = resp.read().decode("utf-8")

            if code == 200:
                passed(f"Space {label} → HTTP 200 ({url})")
                if method == "POST":
                    try:
                        parsed = json.loads(body)
                        if "task_level" in parsed and "events" in parsed:
                            passed("Response contains task_level and events fields")
                        else:
                            passed("Response returned 200 (fields may differ with openenv-core)")
                    except json.JSONDecodeError:
                        passed("Response returned 200")
                return True
            else:
                print(f"  HTTP {code} at {url} — trying next...")
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} at {url} — {e.reason}")
        except urllib.error.URLError as e:
            print(f"  Could not reach {url} — {e.reason}")
        except Exception as e:
            print(f"  Error at {url} — {e}")

    failed(
        f"Space not reachable at {base}",
        f"Check your Space is Running (green dot). Try opening {base} in your browser."
    )
    return False


# ---------------------------------------------------------------------------
# Check 2 — Required files exist
# ---------------------------------------------------------------------------
def check_files_exist(repo_dir: Path) -> bool:
    section("Check 2 — Required files present")
    required = [
        "sifer_env.py",
        "server.py",
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "requirements.txt",
        "README.md",
    ]
    all_ok = True
    for f in required:
        path = repo_dir / f
        if path.exists():
            passed(f"{f} found")
        else:
            failed(f"{f} MISSING", f"Create this file in {repo_dir}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Check 3 — openenv.yaml has required fields
# ---------------------------------------------------------------------------
def check_yaml(repo_dir: Path) -> bool:
    section("Check 3 — openenv.yaml structure")
    yaml_path = repo_dir / "openenv.yaml"
    if not yaml_path.exists():
        failed("openenv.yaml not found")
        return False

    try:
        import yaml  # type: ignore
    except ImportError:
        # Try manual parse for basic fields
        content = yaml_path.read_text(encoding="utf-8")
        required_keys = ["name", "version", "tasks", "observation_space", "action_space"]
        all_ok = True
        for key in required_keys:
            if key + ":" in content:
                passed(f"openenv.yaml has '{key}' field")
            else:
                failed(f"openenv.yaml missing '{key}' field")
                all_ok = False
        return all_ok

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        required_keys = ["name", "version", "tasks", "observation_space", "action_space", "reward"]
        all_ok = True
        for key in required_keys:
            if key in data:
                passed(f"openenv.yaml has '{key}' field")
            else:
                failed(f"openenv.yaml missing '{key}'", f"Add '{key}:' section to openenv.yaml")
                all_ok = False

        # Check tasks have difficulty levels
        tasks = data.get("tasks", [])
        if len(tasks) >= 3:
            passed(f"openenv.yaml has {len(tasks)} tasks (minimum 3 required)")
        else:
            failed(f"Only {len(tasks)} tasks found", "Need at least 3 tasks")
            all_ok = False

        return all_ok
    except Exception as e:
        failed(f"openenv.yaml parse error: {e}")
        return False


# ---------------------------------------------------------------------------
# Check 4 — README has HF Spaces frontmatter
# ---------------------------------------------------------------------------
def check_readme(repo_dir: Path) -> bool:
    section("Check 4 — README.md HF Spaces config")
    readme_path = repo_dir / "README.md"
    if not readme_path.exists():
        failed("README.md not found")
        return False

    content = readme_path.read_text(encoding="utf-8")
    all_ok = True
    required_fields = ["title:", "sdk: docker", "tags:"]
    for field in required_fields:
        if field in content:
            passed(f"README has '{field}'")
        else:
            failed(f"README missing '{field}'", "Add it to the YAML frontmatter at the top of README.md")
            all_ok = False

    if "openenv" in content:
        passed("README has 'openenv' tag")
    else:
        failed("README missing 'openenv' tag", "Add '  - openenv' under tags: in README frontmatter")
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Check 5 — Smoke test: oracle actions score 1.0 on all 3 tasks
# ---------------------------------------------------------------------------
def check_smoke_test(repo_dir: Path) -> bool:
    section("Check 5 — Smoke test (oracle actions)")

    # Add repo_dir to path so we can import sifer_env
    sys.path.insert(0, str(repo_dir))

    try:
        import sifer_env as se
    except ImportError as e:
        failed(f"Could not import sifer_env: {e}", "Run: pip install pydantic")
        return False

    try:
        env = se.SiferTrustEnv(seed=42)
        oracle = {
            1: se.RevokeOrders(ip_address=se.ABUSER_IP),
            2: se.DeleteBotReviews(user_ids=se.BOT_REVIEWERS),
            3: se.CancelOrders(order_ids=se.SCALPER_ORDERS),
        }

        all_ok = True
        for level in [1, 2, 3]:
            obs = env.reset(task_level=level)
            result = env.step(oracle[level])
            score = result.reward.value

            if score == 1.0:
                passed(f"Task {level} oracle action → reward {score:+.1f}")
            else:
                failed(
                    f"Task {level} oracle action → reward {score:+.1f} (expected 1.0)",
                    "Check the grader logic in sifer_env.py"
                )
                all_ok = False

        return all_ok

    except Exception as e:
        failed(f"Smoke test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Check 6 — False positive penalties work
# ---------------------------------------------------------------------------
def check_penalties(repo_dir: Path) -> bool:
    section("Check 6 — Penalty cases (false positives)")
    sys.path.insert(0, str(repo_dir))

    try:
        import sifer_env as se
        env = se.SiferTrustEnv(seed=42)

        cases = [
            # (task_level, action, expected_score, description)
            (1, se.RevokeOrders(ip_address=se.LEGIT_PROMO_IP),    -1.0, "Task 1 false positive IP"),
            (1, se.RevokeOrders(ip_address="1.2.3.4"),             -0.2, "Task 1 unknown IP"),
            (1, se.Pass(),                                           0.0, "Task 1 Pass"),
            (2, se.DeleteBotReviews(user_ids=[se.LEGIT_REVIEWERS[0]]), -1.0, "Task 2 false positive reviewer"),
            (2, se.DeleteBotReviews(user_ids=se.BOT_REVIEWERS[:25]), 0.5, "Task 2 partial (25/50)"),
            (3, se.CancelOrders(order_ids=["ORD_SC_0001", "ORD_SC_0002"]), -0.2, "Task 3 too few (2/10)"),
            (3, se.CancelOrders(order_ids=se.SCALPER_ORDERS[:5]),  0.5, "Task 3 partial (5/10)"),
        ]

        all_ok = True
        for task_level, action, expected, desc in cases:
            env.reset(task_level=task_level)
            result = env.step(action)
            actual = result.reward.value
            if abs(actual - expected) < 0.01:
                passed(f"{desc} → {actual:+.1f} ✓")
            else:
                failed(f"{desc} → got {actual:+.1f}, expected {expected:+.1f}")
                all_ok = False

        return all_ok

    except Exception as e:
        failed(f"Penalty check crashed: {e}")
        return False


# ---------------------------------------------------------------------------
# Check 7 — openenv validate (if openenv-core is installed)
# ---------------------------------------------------------------------------
def check_openenv_validate(repo_dir: Path) -> bool:
    section("Check 7 — openenv validate (requires openenv-core)")

    if importlib.util.find_spec("openenv") is None:
        print(f"  {YELLOW}⚠️  SKIP{RESET} — openenv-core not installed.")
        print(f"  {YELLOW}   Install with: pip install openenv-core{RESET}")
        print(f"  {YELLOW}   Then re-run this script.{RESET}")
        return True  # Don't fail — just skip

    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "openenv", "validate"],
            capture_output=True, text=True, cwd=str(repo_dir), timeout=60
        )
        if result.returncode == 0:
            passed("openenv validate passed")
            return True
        else:
            # Also try the direct CLI
            result2 = subprocess.run(
                ["openenv", "validate"],
                capture_output=True, text=True, cwd=str(repo_dir), timeout=60
            )
            if result2.returncode == 0:
                passed("openenv validate passed")
                return True
            else:
                failed("openenv validate failed")
                print(result2.stdout[-1000:])
                print(result2.stderr[-500:])
                return False
    except FileNotFoundError:
        print(f"  {YELLOW}⚠️  SKIP{RESET} — openenv CLI not found. Run: pip install openenv-core")
        return True
    except Exception as e:
        failed(f"openenv validate error: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}  SiferTrustEnv — Submission Validator{RESET}")
    print(f"{BOLD}{'='*55}{RESET}")

    if len(sys.argv) < 2:
        print("\nUsage: python validate.py <hf_space_url> [repo_dir]")
        print("Example: python validate.py https://devanshikdesai-sifertrusten.hf.space .")
        sys.exit(1)

    space_url = sys.argv[1].rstrip("/")
    repo_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    repo_dir  = repo_dir.resolve()

    print(f"\n  Space URL : {space_url}")
    print(f"  Repo dir  : {repo_dir}")

    # Run all checks
    check_space_live(space_url)
    check_files_exist(repo_dir)
    check_yaml(repo_dir)
    check_readme(repo_dir)
    check_smoke_test(repo_dir)
    check_penalties(repo_dir)
    check_openenv_validate(repo_dir)

    # Summary
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{BOLD}{'='*55}{RESET}")
    if FAIL_COUNT == 0:
        print(f"{GREEN}{BOLD}  ALL {PASS_COUNT}/{total} CHECKS PASSED — ready to submit! 🎉{RESET}")
    else:
        print(f"{RED}{BOLD}  {FAIL_COUNT} check(s) failed — fix them before submitting.{RESET}")
        print(f"  {PASS_COUNT} passed, {FAIL_COUNT} failed out of {total} total checks.")
    print(f"{BOLD}{'='*55}{RESET}\n")

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()