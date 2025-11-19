from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from template_manager import list_templates, generate_client_bot, TemplateNotFound

router = APIRouter(prefix="/api/templates", tags=["templates"])


@router.get("/", response_model=List[Dict[str, str]])
def api_list_templates():
    """
    Get a list of available templates, e.g. for a dropdown in admin portal.
    """
    return list_templates()


class GenerateTemplateRequest(BaseModel):
    bot_type: str          # e.g. "law_firm"
    filename: str          # e.g. "intake_v1.json"
    business_name: str
    contact_name: str | None = None
    phone: str | None = None
    email: str | None = None
    website: str | None = None
    city: str | None = None
    booking_link: str | None = None
    extra_context: Dict[str, Any] | None = None


@router.post("/generate")
def api_generate_template(req: GenerateTemplateRequest):
    """
    Generate a client-specific bot config from a master template.
    Returns JSON that you can save to DB or download.
    """
    ctx: Dict[str, Any] = {
        "business_name": req.business_name,
        "contact_name": req.contact_name or "",
        "phone": req.phone or "",
        "email": req.email or "",
        "website": req.website or "",
        "city": req.city or "",
        "booking_link": req.booking_link or "",
    }
    if req.extra_context:
        ctx.update(req.extra_context)

    try:
        rendered = generate_client_bot(req.bot_type, req.filename, ctx)
    except TemplateNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    # For now, just return the rendered JSON. You can later:
    # - Save it in your DB
    # - Attach it to a project
    # - Store as generated_bots/{client}.json
    return rendered
