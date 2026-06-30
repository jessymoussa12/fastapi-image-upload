from fastapi import FastAPI

from app.routers import prediction

app = FastAPI()


@app.get("/")
def root():
    return {"message": "FastAPI vision API is running"}


app.include_router(prediction.router)
