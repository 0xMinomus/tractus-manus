"""Generate the monochrome system map.

The README banner is supplied artwork and is not generated here. Text in the
map is converted to outlines so the artwork renders the same on GitHub,
Windows, macOS, and Linux. Bodoni Moda and IBM Plex Mono are used under the
SIL Open Font License. See docs/assets/fonts/README.md.
"""

from __future__ import annotations

from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "docs" / "assets" / "fonts"
ASSET_DIR = ROOT / "docs" / "assets"

INK = "#050505"
PAPER = "#f5f5f5"
MUTED = "#a1a1aa"
FAINT = "#52525b"
RULE = "#3f3f46"


def _font(name: str, axes: dict[str, float] | None = None) -> TTFont:
    font = TTFont(FONT_DIR / name)
    if axes and "fvar" in font:
        font = instantiateVariableFont(font, axes, inplace=True)
    return font


DISPLAY = _font("BodoniModa-Variable.ttf", {"wght": 700, "opsz": 72})
MONO = _font("IBMPlexMono-Regular.ttf")
MONO_MEDIUM = _font("IBMPlexMono-Medium.ttf")


def text_width(text: str, font: TTFont, size: float, tracking: float = 0.0) -> float:
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    upem = font["head"].unitsPerEm
    width = 0.0
    for char in text:
        glyph = cmap.get(ord(char))
        if glyph is None:
            glyph = ".notdef"
        width += hmtx[glyph][0] * size / upem + tracking
    return width


def text_paths(
    text: str,
    font: TTFont,
    size: float,
    x: float,
    y: float,
    *,
    tracking: float = 0.0,
    fill: str = PAPER,
    opacity: float = 1.0,
    anchor: str = "start",
) -> str:
    width = text_width(text, font, size, tracking)
    if anchor == "end":
        x -= width
    elif anchor == "middle":
        x -= width / 2.0
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    upem = font["head"].unitsPerEm
    glyphset = font.getGlyphSet()
    scale = size / upem
    cursor = x
    paths: list[str] = []
    for char in text:
        glyph = cmap.get(ord(char))
        if glyph is None:
            glyph = ".notdef"
        pen = SVGPathPen(glyphset)
        glyphset[glyph].draw(pen)
        commands = pen.getCommands()
        if commands:
            paths.append(
                f'<path d="{commands}" transform="translate({cursor:.3f} {y:.3f}) '
                f'scale({scale:.6f} {-scale:.6f})"/>'
            )
        cursor += hmtx[glyph][0] * scale + tracking
    return f'<g fill="{fill}" opacity="{opacity}">' + "".join(paths) + "</g>"


def arrow(x1, y1, x2, y2, dashed=False) -> str:
    dash = ' stroke-dasharray="5 5"' if dashed else ""
    return (
        f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{MUTED}" '
        f'stroke-width="1.2"{dash}/>'
        f'<path d="M{x2 - 7} {y2 - 4} L{x2} {y2} L{x2 - 7} {y2 + 4} Z" '
        f'fill="{MUTED}"/>'
    )


def box(x, y, w, h, title, subtitle, dashed=False) -> str:
    dash = ' stroke-dasharray="5 5"' if dashed else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
        f'stroke="{RULE if not dashed else MUTED}" stroke-width="1"{dash}/>'
        + text_paths(title, MONO_MEDIUM, 11.5, x + 14, y + 25, tracking=0.9, fill=PAPER)
        + text_paths(subtitle, MONO, 9.5, x + 14, y + 45, tracking=0.4, fill=MUTED)
    )


def architecture_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 520" '
        'width="1200" height="520" role="img" aria-labelledby="title desc">',
        '<title id="title">Tractus Manus system map</title>',
        '<desc id="desc">Monochrome diagram. A webcam goes to MediaPipe, 47 features, '
        'the gesture engine, and Windows SendInput. A personal classifier is optional. '
        'Training uses landmark records only.</desc>',
        f'<rect width="1200" height="520" fill="{INK}"/>',
        text_paths("System map", DISPLAY, 48.0, 48, 72, tracking=0.5, fill=PAPER),
        text_paths(
            "RUNTIME / OPTIONAL CLASSIFIER / TRAINING",
            MONO,
            10.0,
            50,
            96,
            tracking=1.5,
            fill=MUTED,
        ),
        f'<path d="M48 112 H1152" stroke="{RULE}" stroke-width="1"/>',
        text_paths("01  RUNTIME", MONO, 10.0, 48, 140, tracking=1.2, fill=MUTED),
        text_paths("02  TRAINING", MONO, 10.0, 48, 354, tracking=1.2, fill=MUTED),
        box(48, 158, 168, 72, "WEBCAM", "LOCAL FRAME"),
        box(246, 158, 188, 72, "MEDIAPIPE", "21 LANDMARKS"),
        box(464, 158, 184, 72, "47 FEATURES", "SCALE / ROTATION"),
        box(678, 158, 196, 72, "GESTURE ENGINE", "GEOMETRY / TIMING"),
        box(904, 158, 248, 72, "WINDOWS SENDINPUT", "CURSOR / BUTTONS"),
        arrow(216, 194, 240, 194),
        arrow(434, 194, 458, 194),
        arrow(648, 194, 672, 194),
        arrow(874, 194, 898, 194),
        box(
            678,
            278,
            210,
            68,
            "PERSONAL CLASSIFIER",
            "OPTIONAL / NUMPY",
            dashed=True,
        ),
        f'<path d="M800 278 V236" fill="none" stroke="{MUTED}" stroke-width="1.2" '
        'stroke-dasharray="5 5"/>',
        '<path d="M793 242 L800 234 L807 242 Z" fill="' + MUTED + '"/>',
        box(48, 380, 208, 70, "COLLECTOR", "LANDMARKS ONLY"),
        box(286, 380, 208, 70, "TRAINER", "PYTORCH / CUDA"),
        box(524, 380, 208, 70, "MODEL NPZ", "5 MLP ENSEMBLE"),
        arrow(256, 414, 280, 414),
        arrow(494, 414, 518, 414),
        f'<path d="M732 414 C820 414 850 360 850 350" fill="none" stroke="{MUTED}" '
        'stroke-width="1.2" stroke-dasharray="5 5"/>',
        '<path d="M843 356 L850 348 L857 356 Z" fill="' + MUTED + '"/>',
        text_paths(
            "NO IMAGES / NO FACE DATA / LOCAL FILES",
            MONO,
            9.0,
            48,
            492,
            tracking=0.8,
            fill=FAINT,
        ),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "architecture.svg").write_text(
        architecture_svg() + "\n", encoding="utf-8"
    )
    print("Wrote docs/assets/architecture.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
