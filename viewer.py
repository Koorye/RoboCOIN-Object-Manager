#!/usr/bin/env python3
"""
Object Library Viewer — browse the deduplicated object library.

Usage:
    pip install fastapi uvicorn jinja2
    python viewer.py
    # → open http://localhost:8888
"""

import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
LIBRARY_INDEX = BASE_DIR / "objects" / "library" / "index.json"
LIBRARY_DIR = BASE_DIR / "objects" / "library"
CROPS_DIR = BASE_DIR / "objects" / "crops"

app = FastAPI()

# Serve crop images and library (canonical) images
app.mount("/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/library", StaticFiles(directory=str(LIBRARY_DIR)), name="library")


def load_library():
    if not LIBRARY_INDEX.exists():
        return {}
    with open(LIBRARY_INDEX) as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def index(
    category: str = Query("", description="Filter by category"),
    color: str = Query("", description="Filter by color"),
    material: str = Query("", description="Filter by material"),
    sort: str = Query("count", description="Sort: count | category | color"),
):
    lib = load_library()
    objects = list(lib.values())

    # Extract for filters
    all_categories = sorted(set(
        o["attributes"].get("category", "?") for o in objects
    ))
    all_colors = sorted(set(
        o["attributes"].get("color", "?") for o in objects
    ))
    all_materials = sorted(set(
        o["attributes"].get("material", "?") for o in objects
    ))

    # Apply filters
    if category:
        objects = [o for o in objects if o["attributes"].get("category") == category]
    if color:
        objects = [o for o in objects if o["attributes"].get("color") == color]
    if material:
        objects = [o for o in objects if o["attributes"].get("material") == material]

    # Sort
    if sort == "count":
        objects.sort(key=lambda o: o["instance_count"], reverse=True)
    elif sort == "category":
        objects.sort(key=lambda o: o["attributes"].get("category", ""))
    elif sort == "color":
        objects.sort(key=lambda o: o["attributes"].get("color", ""))

    filters_html = _build_filters(all_categories, all_colors, all_materials,
                                  category, color, material, sort)

    cards = ""
    for obj in objects:
        canonical = obj["canonical_path"]
        canonical_url = "/library/" + canonical.replace("objects/library/", "", 1)
        attr = obj["attributes"]
        cards += f"""
        <a href="/object/{obj['id']}" class="card">
            <img src="{canonical_url}" loading="lazy">
            <div class="card-info">
                <div class="name">{attr.get('category', '?')}</div>
                <div class="sub">{attr.get('color', '?')} · {attr.get('material', '?')}</div>
                <div class="count">{obj['instance_count']} instances</div>
            </div>
        </a>"""

    return HTML_TEMPLATE.format(
        title=f"Object Library ({len(objects)} objects)",
        filters=filters_html,
        content=cards,
        count=len(objects),
    )


@app.get("/object/{obj_id}", response_class=HTMLResponse)
def object_detail(obj_id: str):
    lib = load_library()
    obj = lib.get(obj_id)
    if not obj:
        return HTMLResponse(f"<h1>Object {obj_id} not found</h1>", status_code=404)

    attr = obj["attributes"]
    canonical = obj["canonical_path"]
    canonical_url = "/library/" + canonical.replace("objects/library/", "", 1)

    instances_html = ""
    for inst in obj.get("instances", []):
        inst_path = inst
        if inst_path.startswith("objects/"):
            inst_path = inst_path.replace("objects/crops/", "")
        instances_html += f'<img src="/crops/{inst_path}" loading="lazy" title="{inst}">'

    html = f"""
    <div class="detail">
        <a href="/" class="back">← Back to library</a>
        <div class="detail-main">
            <img class="canonical" src="{canonical_url}">
            <div class="detail-attrs">
                <h2>{obj['id']}</h2>
                <table>
                    <tr><td>Category</td><td><b>{attr.get('category', '?')}</b></td></tr>
                    <tr><td>Color</td><td><b>{attr.get('color', '?')}</b></td></tr>
                    <tr><td>Material</td><td><b>{attr.get('material', '?')}</b></td></tr>
                    <tr><td>Shape</td><td><b>{attr.get('shape', '?')}</b></td></tr>
                    <tr><td>Texture</td><td><b>{attr.get('texture', '?')}</b></td></tr>
                    <tr><td>Instances</td><td><b>{obj['instance_count']}</b></td></tr>
                </table>
            </div>
        </div>
        <h3>All Instances</h3>
        <div class="instances">{instances_html}</div>
    </div>"""
    return HTML_TEMPLATE.format(title=f"{obj_id} - {attr.get('category', '?')}",
                                filters="", content=html, count=0)


def _build_filters(cats, colors, materials, sel_cat, sel_color, sel_mat, sort):
    def select_opts(options, selected, name):
        html = f'<select name="{name}" onchange="this.form.submit()">'
        html += f'<option value="">All {name}s</option>'
        for o in options:
            sel = "selected" if o == selected else ""
            html += f'<option value="{o}" {sel}>{o}</option>'
        html += '</select>'
        return html

    return f"""
    <form class="filters">
        {select_opts(cats, sel_cat, 'category')}
        {select_opts(colors, sel_color, 'color')}
        {select_opts(materials, sel_mat, 'material')}
        <select name="sort" onchange="this.form.submit()">
            <option value="count" {"selected" if sort=="count" else ""}>Sort: count ↓</option>
            <option value="category" {"selected" if sort=="category" else ""}>Sort: category</option>
            <option value="color" {"selected" if sort=="color" else ""}>Sort: color</option>
        </select>
    </form>"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #111; color: #eee; padding: 20px; }}
h1 {{ margin-bottom: 16px; font-size: 1.4em; }}
.filters {{ display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; }}
.filters select {{ padding:6px 10px; border-radius:6px; border:1px solid #444;
                    background:#222; color:#eee; cursor:pointer; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 12px; }}
.card {{ display:block; background:#1a1a1a; border-radius:10px; overflow:hidden;
        text-decoration:none; color:#ccc; transition:transform .15s; }}
.card:hover {{ transform:scale(1.03); }}
.card img {{ width:100%; aspect-ratio:1; object-fit:cover; }}
.card-info {{ padding:10px; }}
.card-info .name {{ color:#fff; font-weight:600; text-transform:capitalize; }}
.card-info .sub {{ font-size:.8em; color:#999; margin-top:2px; }}
.card-info .count {{ font-size:.75em; color:#666; margin-top:4px; }}
.detail {{ max-width:900px; margin:0 auto; }}
.back {{ color:#4af; text-decoration:none; display:inline-block; margin-bottom:16px; }}
.detail-main {{ display:flex; gap:24px; margin-bottom:30px; flex-wrap:wrap; }}
.canonical {{ width:300px; border-radius:10px; }}
.detail-attrs table {{ border-collapse:collapse; }}
.detail-attrs td {{ padding:6px 16px 6px 0; color:#999; }}
.detail-attrs td b {{ color:#eee; }}
.instances {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
             gap:8px; }}
.instances img {{ width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px; }}
.summary {{ color:#666; margin-bottom:12px; font-size:.9em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="summary">{count} objects</div>
{filters}
<div class="grid">{content}</div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
