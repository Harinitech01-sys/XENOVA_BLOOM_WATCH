from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from config import settings
from api.analysis_endpoints import analysis_router
from api.satellite_endpoints import satellite_router
from api.data_export_endpoints import export_router
from database.connection import init_db, close_db
from bloom_detection import BloomDetectionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting BloomWatch Analysis Backend...")
    await init_db()
    
    # Initialize bloom detection service
    app.state.bloom_service = BloomDetectionService()
    await app.state.bloom_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down BloomWatch Analysis Backend...")
    await close_db()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NASA Space Apps Challenge - Earth Observation for Global Flowering Phenology",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router, prefix=f"{settings.API_V1_STR}/analysis", tags=["analysis"])
app.include_router(satellite_router, prefix=f"{settings.API_V1_STR}/satellite", tags=["satellite"])
app.include_router(export_router, prefix=f"{settings.API_V1_STR}/export", tags=["export"])

@app.get("/")
async def root():
    return {
        "message": "BloomWatch Analysis Backend",
        "version": settings.VERSION,
        "status": "operational",
        "nasa_mission": "Global Flowering Phenology Monitoring"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": "2025-01-27T00:00:00Z",
        "services": {
            "database": "connected",
            "earth_engine": "initialized",
            "bloom_detection": "ready"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )