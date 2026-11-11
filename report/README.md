# B.Tech project report

University template (`coeptech_cse_report.tex`), filled in. Build with pdflatex:

```bash
cd report
latexmk -pdf coeptech_cse_report.tex
```

## Figures

All figures are generated, never drawn by hand:

| Source | Output |
|---|---|
| `figures/drawio/build_diagrams.py` | editable `figures/drawio/*.drawio` (open in draw.io / diagrams.net) |
| `figures/drawio/export_drawio.py` | vector PDFs in `figures/` (and `paper_*.pdf` in `paper/figures/`), via headless Chrome and the draw.io viewer script from viewer.diagrams.net; the diagram is rendered locally |
| `figures/make_report_charts.py` | `figures/ch_*.pdf`, only from saved results in `artifacts/` |
| `scripts/make_paper_figures.py` | `transfer.pdf`, `timeline.pdf`, `operating.pdf` (copied here from `paper/figures/`) |
| `figures/screens/*.png` | dashboard screenshots (Streamlit, light theme) |

If you edit a `.drawio` file in draw.io, re-export it with `python report/figures/drawio/export_drawio.py <name>`; running `build_diagrams.py` again overwrites the `.drawio` files.

## Before submission

- Insert the signed plagiarism page (placeholder page after the certificate).
- Check the month on the title page (currently November 2026).
- Each member reads and revises the text; keep the AI-use statement in the Acknowledgement accurate.
- Verify the bibliography entries (`coep_compit_report.bib`).
