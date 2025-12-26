import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router import router
from .settings import get_settings

cfg = get_settings()
app = FastAPI(title="Private LLM Assistant", version="0.1.0")

# Minimal CORS (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("app.main:app",
                host=cfg["server"]["host"],
                port=cfg["server"]["port"],
                reload=True)
