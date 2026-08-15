from fastapi import FastAPI
from backend.app.api.classify import router as classify_router

app = FastAPI(title="DocuMind API")
app.include_router(classify_router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
