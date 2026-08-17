#!/usr/bin/env python3
"""Export object library to a self-contained HTML file with inline images."""
import base64, json, os
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/home/baai/Projects/RoboCOIN-Object-Manager")
INDEX = BASE_DIR / "objects" / "library" / "index.json"


def img_b64(path):
    """Read image and return base64 data URI."""
    try:
        with open(BASE_DIR / path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        return ""


with open(INDEX) as f:
    lib = json.load(f)

objects = list(lib.values())
objects.sort(key=lambda o: o["instance_count"], reverse=True)

# Collect filter options
all_cats = sorted(set(o["attributes"].get("category", "?") for o in objects))
all_colors = sorted(set(o["attributes"].get("color", "?") for o in objects))
all_materials = sorted(set(o["attributes"].get("material", "?") for o in objects))

# Embed canonical images
for obj in objects:
    obj["_canonical_b64"] = img_b64(obj["canonical_path"])
    obj["_instances_b64"] = []
    for inst in obj.get("instances", []):
        obj["_instances_b64"].append(img_b64(inst))

# Build cards
card_parts = []
for obj in objects:
    a = obj["attributes"]
    cat = a.get("category", "?")
    col = a.get("color", "?")
    mat = a.get("material", "?")
    card_parts.append(
        '<div class="card" data-category="' + cat + '" data-color="' + col + '" data-material="' + mat + '">'
        '<img src="' + obj["_canonical_b64"] + '" loading="lazy" onclick="openDetail(\'' + obj["id"] + '\')">'
        '<div class="card-info">'
        '<div class="name">' + cat + "</div>"
        '<div class="sub">' + col + " · " + mat + "</div>"
        '<div class="count">' + str(obj["instance_count"]) + " instances</div>"
        "</div></div>"
    )

# Build detail panels
detail_parts = []
for obj in objects:
    a = obj["attributes"]
    inst_imgs = ""
    for b64 in obj["_instances_b64"]:
        inst_imgs += '<img src="' + b64 + '" loading="lazy">'
    detail_parts.append(
        '<div class="detail-panel" id="detail-' + obj["id"] + '">'
        '<a href="#" class="back" onclick="closeDetail()">← Back to library</a>'
        '<div class="detail-main">'
        '<img class="canonical" src="' + obj["_canonical_b64"] + '">'
        '<div class="detail-attrs">'
        "<h2>" + obj["id"] + "</h2>"
        "<table>"
        "<tr><td>Category</td><td><b>" + a.get("category", "?") + "</b></td></tr>"
        "<tr><td>Color</td><td><b>" + a.get("color", "?") + "</b></td></tr>"
        "<tr><td>Material</td><td><b>" + a.get("material", "?") + "</b></td></tr>"
        "<tr><td>Shape</td><td><b>" + a.get("shape", "?") + "</b></td></tr>"
        "<tr><td>Texture</td><td><b>" + a.get("texture", "?") + "</b></td></tr>"
        "<tr><td>Instances</td><td><b>" + str(obj["instance_count"]) + "</b></td></tr>"
        "</table>"
        "</div></div>"
        "<h3>All Instances</h3>"
        '<div class="instances">' + inst_imgs + "</div>"
        "</div>"
    )

# Category counts for sidebar
cat_counts = Counter(o["attributes"].get("category", "?") for o in objects)

sidebar_cat_links = ""
for c in all_cats:
    sidebar_cat_links += (
        '<label class="filter-btn" onclick="filterBy(\'category\',\'' + c + '\')">'
        + c + ' <span class="count">' + str(cat_counts[c]) + "</span></label>"
    )

sidebar_color_links = ""
for c in all_colors:
    sidebar_color_links += (
        '<label class="filter-btn" onclick="filterBy(\'color\',\'' + c + '\')">'
        + c + "</label>"
    )

sidebar_material_links = ""
for c in all_materials:
    sidebar_material_links += (
        '<label class="filter-btn" onclick="filterBy(\'material\',\'' + c + '\')">'
        + c + "</label>"
    )

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object Library (""" + str(len(objects)) + """ objects)</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #111; color: #eee; display:flex; min-height:100vh; }
.sidebar { width:240px; padding:20px; background:#161616; border-right:1px solid #222;
            position:sticky; top:0; height:100vh; overflow-y:auto; flex-shrink:0; }
.sidebar h2 { font-size:1em; margin-bottom:12px; color:#999; text-transform:uppercase;
              letter-spacing:.5px; }
.sidebar h2:not(:first-child) { margin-top:20px; }
.sidebar a, .sidebar label { display:block; padding:4px 8px; border-radius:4px; color:#bbb;
    text-decoration:none; font-size:.85em; cursor:pointer; }
.sidebar a:hover, .sidebar label:hover { background:#222; color:#fff; }
.sidebar .count { float:right; color:#555; font-size:.8em; }
.main { flex:1; padding:20px; min-width:0; }
h1 { margin-bottom:4px; font-size:1.4em; }
.summary { color:#666; margin-bottom:16px; font-size:.9em; }
.grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap:12px; }
.card { display:block; background:#1a1a1a; border-radius:10px; overflow:hidden;
        text-decoration:none; color:#ccc; transition:transform .15s; }
.card:hover { transform:scale(1.03); }
.card img { width:100%; aspect-ratio:1; object-fit:cover; cursor:pointer; }
.card-info { padding:10px; }
.card-info .name { color:#fff; font-weight:600; text-transform:capitalize; }
.card-info .sub { font-size:.8em; color:#999; margin-top:2px; }
.card-info .count { font-size:.75em; color:#666; margin-top:4px; }
.detail-panel { display:none; max-width:900px; }
.back { color:#4af; text-decoration:none; display:inline-block; margin-bottom:16px; }
.detail-main { display:flex; gap:24px; margin-bottom:30px; flex-wrap:wrap; }
.canonical { width:300px; border-radius:10px; }
.detail-attrs table { border-collapse:collapse; }
.detail-attrs td { padding:6px 16px 6px 0; color:#999; }
.detail-attrs td b { color:#eee; }
.instances { display:grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap:8px; }
.instances img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px; }
.filter-active { background:#2a2a2a !important; color:#fff !important; }
</style>
</head>
<body>

<div class="sidebar">
    <h2>Categories</h2>
    <label class="filter-btn filter-active" onclick="filterBy('category','')">All <span class="count">""" + str(len(objects)) + """</span></label>
    """ + sidebar_cat_links + """
    <h2>Colors</h2>
    <label class="filter-btn filter-active" onclick="filterBy('color','')">All</label>
    """ + sidebar_color_links + """
    <h2>Materials</h2>
    <label class="filter-btn filter-active" onclick="filterBy('material','')">All</label>
    """ + sidebar_material_links + """
</div>

<div class="main">
    <h1>Object Library</h1>
    <div class="summary" id="summary">""" + str(len(objects)) + """ objects</div>
    <div class="grid" id="grid">""" + "".join(card_parts) + """</div>
    """ + "".join(detail_parts) + """
</div>

<script>
let activeFilters = {category:'', color:'', material:''};
function filterBy(type, value) {
    activeFilters[type] = value;
    document.querySelectorAll('.sidebar label.filter-btn').forEach(function(btn) { btn.classList.remove('filter-active'); });
    event.target.classList.add('filter-active');
    var count = 0;
    document.querySelectorAll('.card').forEach(function(card) {
        var show = true;
        for (var k in activeFilters) {
            if (activeFilters[k] && card.dataset[k] !== activeFilters[k]) show = false;
        }
        card.style.display = show ? '' : 'none';
        if (show) count++;
    });
    document.getElementById('summary').textContent = count + ' objects';
}
function openDetail(id) {
    document.getElementById('grid').style.display = 'none';
    document.querySelector('.summary').style.display = 'none';
    document.getElementById('detail-' + id).style.display = 'block';
}
function closeDetail() {
    document.getElementById('grid').style.display = '';
    document.querySelector('.summary').style.display = '';
    document.querySelectorAll('.detail-panel').forEach(function(p) { p.style.display = 'none'; });
}
</script>

</body>
</html>"""

OUT = BASE_DIR / "object_library.html"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Exported: {OUT} ({size_mb:.1f}MB)")
