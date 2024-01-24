import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.main_router import main_router
from api.model_router import model_router

app = FastAPI()
app.mount("/static", StaticFiles(directory="api/static"), name="static")
app.include_router(main_router)
app.include_router(model_router, prefix = "/model")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)