# Documentation fonts

The generated SVGs use two typefaces:

- **Bodoni Moda** for the display title.
- **IBM Plex Mono** for labels, captions, and the system map.

Both are distributed under the SIL Open Font License 1.1. The font files are
kept here so the artwork can be regenerated. The SVGs themselves contain text
outlines, so they render identically without loading a font.

Sources:

- Bodoni Moda: https://github.com/google/fonts/tree/main/ofl/bodonimoda
- IBM Plex Mono: https://github.com/google/fonts/tree/main/ofl/ibmplexmono

The license texts are `BodoniModa-OFL.txt` and `IBMPlexMono-OFL.txt`.

To regenerate the SVGs, install FontTools and run:

```bat
uv pip install --python .venv fonttools
.venv\Scripts\python.exe scripts\make_brand_assets.py
```
