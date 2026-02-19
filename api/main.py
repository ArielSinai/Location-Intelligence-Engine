"""
FastAPI application entry point.
"""

from fastapi import FastAPI

from api.routers.recommendations import router as recommendations_router

app = FastAPI(
    title="Location Intelligence Engine",
    description="Recommends optimal locations to open a new business.",
    version="1.0.0",
)

app.include_router(recommendations_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
