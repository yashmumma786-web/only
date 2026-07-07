"""
Admin Authentication and Session Management Routes for StoneStocks

Provides password-protected admin login/logout routes and dependencies.
"""

import os
from typing import Optional
from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="templates")


def get_admin_password() -> str:
    """Get admin password from environment."""
    return os.environ.get("ADMIN_PASSWORD")


@router.get("/login", response_class=HTMLResponse)
async def admin_login_page(request: Request, next: Optional[str] = None):
    """Show admin login form."""
    return templates.TemplateResponse(request, "admin_login.html", {
        "request": request,
        "error": None,
        "next": next
    })


@router.post("/login")
async def admin_login(
    request: Request,
    password: str = Form(...),
    next: Optional[str] = Form(None)
):
    """Handle admin login."""
    admin_password = get_admin_password()
    
    if not admin_password:
        return templates.TemplateResponse(request, "admin_login.html", {
            "request": request,
            "error": "ADMIN_PASSWORD not configured. Please set it in environment secrets.",
            "next": next
        })
    
    if password == admin_password:
        request.session["is_admin"] = True
        redirect_url = next if next else "/find"
        return RedirectResponse(url=redirect_url, status_code=303)
    else:
        return templates.TemplateResponse(request, "admin_login.html", {
            "request": request,
            "error": "Invalid password",
            "next": next
        })


@router.get("/logout")
async def admin_logout(request: Request):
    """Log out admin."""
    request.session.pop("is_admin", None)
    return RedirectResponse(url="/admin/login", status_code=303)
