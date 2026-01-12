"""
测试FastAPI lifespan
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("LIFESPAN STARTUP")
    print("=" * 60)
    yield
    print("=" * 60)
    print("LIFESPAN SHUTDOWN")
    print("=" * 60)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    print("Starting uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
