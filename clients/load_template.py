import json
from pathlib import Path

def load_json(path):
    return json.loads(Path(path).read_text())

def render_placeholders(data, ctx):
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

def generate_client_bot(template_path, client_config_path, output_path):
    template = load_json(template_path)
    client_cfg = load_json(client_config_path)
    rendered = render_placeholders(template, client_cfg)
    Path(output_path).write_text(json.dumps(rendered, indent=2))

# example usage:
generate_client_bot(
    "bot_templates/law_firm/intake_v1.json",
    "clients/a1_law_group.json",
    "generated_bots/a1_law_group_law_firm_intake.json"
)
