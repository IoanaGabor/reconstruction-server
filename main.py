from fastapi import FastAPI
from endpoints import reconstructions
from endpoints import metrics
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],           
)

app.include_router(metrics.router)
app.include_router(reconstructions.router)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

uvicorn.run(app, host="0.0.0.0", port=8000)
