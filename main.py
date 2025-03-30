from fastapi import FastAPI
from endpoints import reconstructions
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],           
)

app.include_router(reconstructions.router)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
