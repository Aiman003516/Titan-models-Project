"""
Titan Agent — FastAPI Application

Endpoints:
  GET  /                  Web UI
  GET  /health            Health check
  POST /api/build         Start ERP build (returns JSON)
  GET  /api/build/{id}    Poll build status
  WS   /ws/build          WebSocket with real-time events
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from titan.config import settings
from titan.agent import TitanAgent, ERPBuildResult
from titan.erp.prompts import MODULE_REGISTRY, MODULE_DESCRIPTIONS

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Build result store (in-memory) ────────────────────────────────────────────

builds: dict[str, dict] = {}

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Titan Agent starting...")
    logger.info(f"Backend endpoint: {settings.titan_backend_endpoint}")
    logger.info(f"UI endpoint: {settings.titan_ui_endpoint}")
    logger.info(f"Groq model: {settings.groq_model}")
    yield
    logger.info("Titan Agent shutting down...")


app = FastAPI(
    title="Titan ERP Generator",
    description="Standalone agent for generating Crystal Helix ERP modules",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend_endpoint": settings.titan_backend_endpoint,
        "ui_endpoint": settings.titan_ui_endpoint,
        "groq_model": settings.groq_model,
    }


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/modules")
async def list_modules():
    """List all available ERP modules."""
    modules = []
    for name, arch in MODULE_REGISTRY.items():
        desc = MODULE_DESCRIPTIONS.get(name, {})
        modules.append({
            "name": name,
            "architecture": arch,
            "display_name": desc.get("name", name),
            "entities": desc.get("entities", ""),
            "domain": desc.get("domain", ""),
        })
    return {"modules": modules, "total": len(modules)}


@app.post("/api/build")
async def start_build(modules: list[str]):
    """Start an ERP build. Returns a build ID for polling."""
    build_id = str(uuid.uuid4())[:8]
    builds[build_id] = {
        "id": build_id,
        "status": "starting",
        "modules": modules,
        "events": [],
        "result": None,
        "started_at": time.time(),
    }

    # Run in background
    asyncio.create_task(_run_build(build_id, modules))
    return {"build_id": build_id, "status": "starting"}


@app.get("/api/build/{build_id}")
async def get_build(build_id: str):
    """Get build status and results."""
    build = builds.get(build_id)
    if not build:
        return JSONResponse({"error": "Build not found"}, status_code=404)
    return build


@app.get("/api/build/{build_id}/download")
async def download_build(build_id: str):
    """Download build result as ZIP."""
    build = builds.get(build_id)
    if not build or not build.get("result"):
        return JSONResponse({"error": "Build not ready"}, status_code=404)

    agent = TitanAgent()
    result = build["result"]
    zip_bytes = agent.package_zip(result)
    await agent.close()

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=erp-{build_id}.zip"},
    )


async def _run_build(build_id: str, modules: list[str]):
    """Background task for ERP generation."""
    build = builds[build_id]

    def on_event(event: str, data):
        build["events"].append({"event": event, "data": data, "time": time.time()})
        build["status"] = event

    agent = TitanAgent(on_event=on_event)

    try:
        result = await agent.build_erp(modules)
        build["result"] = result
        build["status"] = "done"
        build["summary"] = result.summary()
    except Exception as e:
        build["status"] = "error"
        build["error"] = str(e)
        logger.exception(f"Build {build_id} failed: {e}")
    finally:
        await agent.close()


# ── WebSocket (real-time events) ──────────────────────────────────────────────

@app.websocket("/ws/build")
async def ws_build(websocket: WebSocket, modules: str = Query(...)):
    """
    WebSocket endpoint for real-time build monitoring.

    Connect: ws://host:port/ws/build?modules=crm,hr,sale
    Receives JSON events as the build progresses.
    """
    await websocket.accept()
    module_list = [m.strip() for m in modules.split(",") if m.strip()]

    if not module_list:
        await websocket.send_json({"error": "No modules specified"})
        await websocket.close()
        return

    async def on_event(event: str, data):
        try:
            await websocket.send_json({"event": event, "data": data})
        except Exception:
            pass

    agent = TitanAgent(on_event=on_event)
    build_id = str(uuid.uuid4())[:8]

    try:
        await websocket.send_json({
            "event": "build.start",
            "data": {"modules": module_list, "build_id": build_id},
        })

        result = await agent.build_erp(module_list)

        # Store result for ZIP download
        builds[build_id] = {
            "id": build_id,
            "status": "done",
            "modules": module_list,
            "result": result,
            "summary": result.summary(),
        }

        # Build file listing for the UI
        file_list = []
        for m in result.modules:
            for f in m.backend_files:
                file_list.append({"path": f.path, "language": f.language, "content": f.content})
            for f in m.frontend_files:
                file_list.append({"path": f.path, "language": f.language, "content": f.content})

        await websocket.send_json({
            "event": "build.complete",
            "data": {
                "build_id": build_id,
                "summary": result.summary(),
                "total_files": result.total_files,
                "pass_rate": f"{result.overall_pass_rate:.0%}",
                "time": f"{result.total_time_seconds:.1f}s",
                "files": file_list,
            },
        })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "data": {"error": str(e)}})
        except Exception:
            pass
        logger.exception(f"WebSocket build error: {e}")
    finally:
        await agent.close()


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Titan Agent</h1><p>UI template not found.</p>"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "titan.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
