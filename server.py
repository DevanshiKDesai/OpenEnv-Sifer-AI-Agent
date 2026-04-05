"""
server.py — FastAPI HTTP server wrapping SiferTrustEnv.
Uses openenv-core's create_fastapi_app when available (passes openenv validate),
falls back to manual FastAPI routes otherwise.
"""
from __future__ import annotations
import os, subprocess, sys
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, HTTPException
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
    app = FastAPI(title="SiferTrustEnv", version="1.0.0")


@app.get("/", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "SiferTrustEnv", "version": "1.0.0"}


if not _using_core:
    class ResetRequest(BaseModel):
        task_level: int = 1

    class ActionRequest(BaseModel):
        action: Dict[str, Any]

    @app.post("/reset")
    def reset_env(req: ResetRequest) -> Dict[str, Any]:
        if req.task_level not in (1, 2, 3):
            raise HTTPException(status_code=400, detail="task_level must be 1, 2 or 3")
        return env.reset(task_level=req.task_level).model_dump()

    @app.post("/step")
    def step_env(req: ActionRequest) -> Dict[str, Any]:
        data, atype = req.action, req.action.get("action_type")
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

    @app.get("/state")
    def get_state() -> Dict[str, Any]:
        return env.state   # @property — no parentheses


@app.post("/run", summary="Run inference.py")
def run_inference() -> Dict[str, Any]:
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        raise HTTPException(400, detail=f"Missing env vars: {missing}")
    try:
        r = subprocess.run([sys.executable, "inference.py"],
                           capture_output=True, text=True, timeout=1200,
                           env=os.environ.copy())
        return {"returncode": r.returncode, "stdout": r.stdout[-8000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail="Inference timed out")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, log_level="info")