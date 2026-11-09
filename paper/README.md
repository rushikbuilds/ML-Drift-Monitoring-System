# Paper: Early Warning of Model Degradation under Delayed Labels

One LaTeX source (`paper.tex`), two builds:

| File | For | Authors | Repository link |
|---|---|---|---|
| `conecct.tex` → `conecct.pdf` | IEEE CONECCT 2027 (double-blind) | "Anonymous Authors" | withheld |
| `arxiv.tex` → `arxiv.pdf` | arXiv preprint (cs.LG) | 4 students + guide | included |

## Build

```bash
python scripts/make_paper_figures.py      # from the repository root; regenerates paper/figures/*.pdf from saved results
cd paper
latexmk -pdf conecct.tex
latexmk -pdf arxiv.tex
```

Both builds use IEEEtran and are 6 pages, inside the 6–8 pages (including references) that the 2026 research track allowed.

## Before submitting (checklist)

**Everyone**
- [ ] Each author reads the whole paper and revises the text in your own words. You are responsible for every claim.
- [ ] Verify every reference in `refs.bib` (authors, title, venue, year, pages) on Google Scholar or the publisher's page; they were written from memory by an AI assistant.
- [ ] Re-check the numbers against `docs/verdict.md`, `docs/v7_verdict.md`, `docs/v8_verdict.md`, `docs/ablations.md`, `docs/v9_protocol.md` and `docs/v10_protocol.md`.
- [ ] Keep the AI-use statement in the Acknowledgment (required by IEEE and arXiv). Adjust it to describe what you actually changed.

**arXiv version**
- [ ] Fill the author block in `paper.tex` (names, department, institution, city, email) and the guide's name in the Acknowledgment.
- [ ] Push the repository to GitHub so the link in the paper works.
- [ ] Obtain an endorsement for cs.LG (from your guide or another established arXiv author), then upload the LaTeX source: `paper.tex`, `arxiv.tex`, `refs.bib`, the generated `arxiv.bbl` and `figures/`.

**CONECCT version**
- [ ] When the 2027 call for papers is published, check page limit, template, anonymization rules, preprint policy, track and submission system, and adjust. The current draft follows the 2026 rules: research track 6–8 pages, double-blind.
- [ ] Anonymization: no names, institution or repository link in the PDF (already handled); check the PDF metadata author field, and refer to your own earlier work in the third person.
- [ ] Choose the track (likely Computing & Software Systems).
