from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.core.ingestion.embedder import load_embedder
from app.core.retrieval.reranker import load_reranker
from app.api.routes import documents, query, study, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect once, load ML models once. Both are expensive to redo
    # per-request, which is exactly what would happen without this.
    await connect_to_mongo()
    load_embedder()
    load_reranker()
    yield
    # Shutdown
    await close_mongo_connection()


app = FastAPI(title="UniSage API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(study.router, prefix="/api/study", tags=["study"])