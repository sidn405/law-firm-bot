import json
from pathlib import Path
from typing import Dict, Any, List

TEMPLATES_ROOT = Path("bot_templates")


class TemplateNotFound(Exception):
    pass


def list_templates() -> List[Dict[str, str]]:
    """
    Scan the bot_templates/ folder and return available templates.
    """
    results: List[Dict[str, str]] = []
    if not TEMPLATES_ROOT.exists():
        return results

    for bot_type_dir in TEMPLATES_ROOT.iterdir():
        if not bot_type_dir.is_dir():
            continue
        for f in bot_type_dir.glob("*.json"):
            results.append({
                "bot_type": bot_type_dir.name,
                "filename": f.name,
                "path": str(f),
                "template_id": f"{bot_type_dir.name}/{f.name}"
            })
    return results


def load_template(bot_type: str, filename: str) -> Dict[str, Any]:
    """
    Load a specific template JSON.
    Example: bot_type='law_firm', filename='intake_v1.json'
    """
    path = TEMPLATES_ROOT / bot_type / filename
    if not path.exists():
        raise TemplateNotFound(f"Template not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render_placeholders(data: Any, ctx: Dict[str, Any]) -> Any:
    """
    Recursively replace {{keys}} in strings using ctx.
    """
    if isinstance(data, dict):
        return {k: render_placeholders(v, ctx) for k, v in data.items()}
    if isinstance(data, list):
        return [render_placeholders(v, ctx) for v in data]
    if isinstance(data, str):
        out = data
        for key, val in ctx.items():
            out = out.replace(f"{{{{{key}}}}}", str(val))
        return out
    return data


def generate_client_bot(bot_type: str, filename: str, client_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load template + apply client_ctx (business_name, phone, etc).
    Returns final ready-to-use bot config (dict).
    """
    template = load_template(bot_type, filename)
    rendered = render_placeholders(template, client_ctx)
    # Optionally attach metadata
    rendered["_meta"] = {
        "bot_type": bot_type,
        "template_filename": filename,
        "client_business_name": client_ctx.get("business_name"),
    }
    return rendered
