"""Export .drawio files to cropped vector PDF with headless Chrome and the official draw.io viewer.

    python report/figures/drawio/export_drawio.py [name ...]

The viewer script (viewer.diagrams.net) renders the diagram locally in the browser;
the diagram itself is not uploaded anywhere.
"""
import html, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PAPER = HERE.parents[2] / "paper" / "figures"  # paper_*.drawio go to the paper
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:#fff;overflow:hidden}</style></head><body>
<div id="d" class="mxgraph" data-mxgraph="%s"></div>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
<script>
function fit(){var s=document.querySelector('#d svg'); if(!s){setTimeout(fit,200);return;}
 var r=s.getBoundingClientRect(); var st=document.createElement('style');
 st.textContent='@page{size:'+Math.ceil(r.width+40)+'px '+Math.ceil(r.height+40)+'px;margin:0}';
 document.head.appendChild(st);}
setTimeout(fit,500);
</script></body></html>"""


def export(src: Path) -> Path:
    cfg = {"xml": src.read_text(encoding="utf-8"), "toolbar": "", "nav": False, "border": 2,
           "resize": False, "lightbox": False, "highlight": "none"}
    page = PAGE % html.escape(json.dumps(cfg), quote=True)
    with tempfile.TemporaryDirectory() as tmp:
        h = Path(tmp) / "page.html"
        h.write_text(page, encoding="utf-8")
        raw = Path(tmp) / "raw.pdf"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--print-to-pdf={raw}", h.as_uri()],
                       check=True, capture_output=True)
        dest = PAPER if src.stem.startswith("paper_") else OUT
        out = dest / (src.stem + ".pdf")
        subprocess.run(["pdfcrop", "--margins", "2", str(raw), str(out)], check=True, capture_output=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:]
    files = [HERE / f"{n}.drawio" for n in names] if names else sorted(HERE.glob("*.drawio"))
    for f in files:
        print(export(f))
