from fastapi import APIRouter
from fastapi.responses import HTMLResponse

main_router = APIRouter()

_main_page = ""
with open("api/static/index.html", "r") as f:
    _main_page = f.read()

@main_router.get("/")
def get_main_page():
    return HTMLResponse(content=_main_page, status_code=200)