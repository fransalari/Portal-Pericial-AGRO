from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router
from app.services.catalog import catalog

app = FastAPI(title="Portal Pericial AGRO · Sancor Seguros", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    catalog.ensure_ready(force=False)


@app.get("/")
def root():
    return {
        "app": "Portal Pericial AGRO",
        "org": "Sancor Seguros Agro",
        "ok": True,
        "index_ready": bool(catalog.meta.get("ready")),
    }
