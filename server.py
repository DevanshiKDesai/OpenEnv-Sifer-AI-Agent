"""
server.py — FastAPI HTTP server wrapping SiferTrustEnv.

The /reset endpoint accepts an empty body {}, null, or {"task_level": N}
so automated validators that POST with no body still get a 200 response.
"""
from __future__ import annotations
import os, subprocess, sys
from typing import Any, Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sifer_env import (
    CancelOrders, DeleteBotReviews, Observation,
    Pass, RevokeOrders, SiferTrustEnv, StepResult,
)

env = SiferTrustEnv(seed=42)

try:
    from openenv.core.env_server import create_fastapi_app
    from sifer_env import Action
    app = create_fastapi_app(env, Action, Observation)
    _using_core = True
except Exception:
    _using_core = False
    app = FastAPI(title="SiferTrustEnv", version="1.0.0",
                  description="OpenEnv-compliant Trust & Safety fraud analyst simulation.")


# ---------------------------------------------------------------------------
# Health check — always present
# ---------------------------------------------------------------------------
@app.get("/", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "SiferTrustEnv", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Manual routes — used when openenv-core is not installed
# ---------------------------------------------------------------------------
if not _using_core:

    class ActionRequest(BaseModel):
        action: Dict[str, Any]

    @app.post("/reset", summary="Reset the environment")
    async def reset_env(request: Request) -> Dict[str, Any]:
        """
        Accepts any of:
          - empty body
          - {}
          - {"task_level": 1}
        Defaults to task_level=1 when not provided.
        """
        task_level = 1
        try:
            body = await request.json()
            if isinstance(body, dict):
                task_level = int(body.get("task_level", 1))
        except Exception:
            pass  # empty or null body — use default task_level=1

        if task_level not in (1, 2, 3):
            task_level = 1  # silently clamp to valid range

        obs: Observation = env.reset(task_level=task_level)
        return obs.model_dump()

    @app.post("/step", summary="Step the environment")
    def step_env(req: ActionRequest) -> Dict[str, Any]:
        data  = req.action
        atype = data.get("action_type")
        try:
            if atype == "RevokeOrders":
                action = RevokeOrders(ip_address=data["ip_address"])
            elif atype == "DeleteBotReviews":
                action = DeleteBotReviews(user_ids=data["user_ids"])
            elif atype == "CancelOrders":
                action = CancelOrders(order_ids=data["order_ids"])
            elif atype == "Pass":
                action = Pass()
            else:
                raise HTTPException(400, detail=f"Unknown action_type '{atype}'")
        except KeyError as e:
            raise HTTPException(422, detail=f"Missing field: {e}")
        try:
            return env.step(action).model_dump()
        except RuntimeError as e:
            raise HTTPException(409, detail=str(e))

    @app.get("/state", summary="Get current internal state")
    def get_state() -> Dict[str, Any]:
        return env.state  # @property


# ---------------------------------------------------------------------------
# /run — always present
# ---------------------------------------------------------------------------
@app.post("/run", summary="Run inference.py")
def run_inference() -> Dict[str, Any]:
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
               if not os.environ.get(v)]
    if missing:
        raise HTTPException(400, detail=f"Missing env vars: {missing}")
    try:
        r = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True, text=True, timeout=1200,
            env=os.environ.copy(),
        )
        return {"returncode": r.returncode,
                "stdout": r.stdout[-8000:],
                "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail="Inference timed out")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def main():
    """Entry point required by openenv validate spec."""
    uvicorn.run("server:app", host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()