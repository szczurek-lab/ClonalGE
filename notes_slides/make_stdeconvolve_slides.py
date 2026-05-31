"""Generate PowerPoint slides explaining STdeconvolve and the ClonalGE vs STdeconvolve benchmark."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE   = RGBColor(0x1a, 0x3a, 0x6b)
DARK   = RGBColor(0x22, 0x22, 0x22)
GREY   = RGBColor(0x55, 0x55, 0x55)
GREEN  = RGBColor(0x1a, 0x6b, 0x3a)
MAROON = RGBColor(0x6b, 0x1a, 0x1a)
LIGHT  = RGBColor(0xee, 0xee, 0xee)


def add_text(slide, left, top, width, height, text, size=14, bold=False,
             color=DARK, align=PP_ALIGN.LEFT, italic=False):
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.alignment = align
    return tf


def add_multiline(slide, left, top, width, height, title, lines,
                  title_size=14, title_color=MAROON,
                  line_size=12, line_color=DARK, line_bold=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(title_size)
    p.font.bold = True
    p.font.color.rgb = title_color
    for line in lines:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(line_size)
        p.font.color.rgb = line_color
        p.font.bold = line_bold
        p.space_before = Pt(4)


def add_placeholder(slide, left, top, width, height, label):
    """Add a dashed-border rectangle as a figure placeholder."""
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = LIGHT
    rect.line.color.rgb = GREY
    rect.line.width = Pt(1)
    # Label inside
    tf = rect.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = label
    p.font.size = Pt(12)
    p.font.italic = True
    p.font.color.rgb = GREY
    p.alignment = PP_ALIGN.CENTER


# ══════════════════════════════════════════════════════════════════════════════
# Build presentation
# ══════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)


# ──────────────────────────────────────────────────────────────────────────────
# SLIDE 1: What is STdeconvolve?
# ──────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Title
add_text(slide, Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.55),
         "What is STdeconvolve?", size=28, bold=True, color=BLUE)
add_text(slide, Inches(0.4), Inches(0.7), Inches(12.5), Inches(0.4),
         "Reference-free deconvolution of spatial transcriptomics — our closest baseline",
         size=15, color=GREY, italic=True)

# Left column: definition
add_multiline(slide, Inches(0.4), Inches(1.3), Inches(6.2), Inches(2.5),
              "Method (Miller et al., 2022)", [
                  "",
                  "• Applies Latent Dirichlet Allocation (LDA) to spatial",
                  "   gene expression counts.",
                  "",
                  "• Decomposes each spot's expression into a mixture of",
                  "   K \"topics\" — analogous to clones in our setting.",
                  "",
                  "• Reference-free: needs no scRNA-seq atlas or external",
                  "   cell-type labels.",
              ], title_size=15, title_color=BLUE, line_size=12)

# Left column: outputs
add_multiline(slide, Inches(0.4), Inches(4.0), Inches(6.2), Inches(2.5),
              "Outputs", [
                  "",
                  "• θ̂ : per-spot topic proportions   (spots × K)",
                  "       → analogous to ClonalGE's H matrix",
                  "",
                  "• β̂ : per-topic gene expression profiles  (K × genes)",
                  "       → analogous to ClonalGE's B matrix",
                  "",
                  "Topics are unlabelled — must be matched to clones",
                  "post-hoc (we use the Hungarian algorithm).",
              ], title_size=15, title_color=BLUE, line_size=12)

# Right column: why this baseline
add_multiline(slide, Inches(7.0), Inches(1.3), Inches(6.0), Inches(3.0),
              "Why STdeconvolve (and not RCTD / Cell2location / etc.)?", [
                  "",
                  "Reference-based deconvolution methods (RCTD, SPOTlight,",
                  "Stereoscope, Cell2location, STRIDE) require a paired",
                  "single-cell RNA-seq atlas with cell-type labels.",
                  "",
                  "→  Such an atlas does not exist for cancer subclones.",
                  "→  Clones are defined by mutations, not transcriptomes.",
                  "",
                  "STdeconvolve is the only existing method applicable",
                  "under comparable data conditions to ClonalGE.",
              ], title_size=14, title_color=MAROON, line_size=12)

# Right column: key difference
add_multiline(slide, Inches(7.0), Inches(4.5), Inches(6.0), Inches(2.5),
              "Key methodological difference", [
                  "",
                  "STdeconvolve uses ONLY spatial expression (Y).",
                  "",
                  "ClonalGE additionally uses somatic SNVs (A, D, C)",
                  "from matched DNA-seq.",
                  "",
                  "→  Mutation signal disambiguates clones that have",
                  "    overlapping expression profiles.",
              ], title_size=14, title_color=GREEN, line_size=12)

# Reference at bottom
add_text(slide, Inches(0.4), Inches(7.15), Inches(12), Inches(0.3),
         "Miller et al. (2022), Reference-free cell type deconvolution of multi-cellular pixel-resolution spatially resolved transcriptomics data, Nature Communications",
         size=9, color=RGBColor(0x88, 0x88, 0x88))


# ──────────────────────────────────────────────────────────────────────────────
# SLIDE 2: Benchmarking design and results (boxplots)
# ──────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Title
add_text(slide, Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.55),
         "Benchmarking ClonalGE vs. STdeconvolve", size=28, bold=True, color=BLUE)
add_text(slide, Inches(0.4), Inches(0.7), Inches(12.5), Inches(0.4),
         "Mutation signal is essential for accurate clonal deconvolution",
         size=15, color=GREY, italic=True)

# Experimental design (top-left)
add_multiline(slide, Inches(0.4), Inches(1.25), Inches(6.2), Inches(2.5),
              "Experimental design", [
                  "",
                  "• 20 independent simulation replicates",
                  "• 3 configurations:  Normal, Low Variance, High Coverage",
                  "• Ground-truth H and B known per replicate",
                  "",
                  "• Hungarian algorithm used to match STdeconvolve's",
                  "   unlabelled topics to true clones",
                  "   (the most favourable mapping for STdeconvolve)",
                  "",
                  "• B matrices row-normalised → scale-free comparison",
              ], title_size=14, title_color=MAROON, line_size=12)

# Metrics (bottom-left)
add_multiline(slide, Inches(0.4), Inches(4.0), Inches(6.2), Inches(2.5),
              "Metrics", [
                  "",
                  "• H MAE — clone proportion recovery (lower = better)",
                  "• B MAE — clone-specific expression recovery (lower = better)",
                  "",
                  "Both metrics measure how close the inferred matrix is",
                  "to the ground truth, after Hungarian alignment.",
              ], title_size=14, title_color=MAROON, line_size=12)

# Figure placeholder (right side)
add_placeholder(slide, Inches(7.0), Inches(1.25), Inches(6.0), Inches(5.5),
                "[ FIGURE PLACEHOLDER ]\n\n"
                "Boxplots: H MAE and B MAE\nacross 3 simulation conditions\n"
                "(figures/stdeconvolve_comparison_figure.pdf)\n\n"
                "ClonalGE (blue) vs STdeconvolve (orange)")

# Bottom take-away strip
strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.4))
strip.fill.solid()
strip.fill.fore_color.rgb = GREEN
strip.line.fill.background()
tf = strip.text_frame
tf.margin_left = Inches(0.1)
p = tf.paragraphs[0]
p.text = "→ ClonalGE outperforms STdeconvolve in every condition, for both H and B"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


# ──────────────────────────────────────────────────────────────────────────────
# SLIDE 3: Results numbers + interpretation
# ──────────────────────────────────────────────────────────────────────────────
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# Title
add_text(slide, Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.55),
         "Results: Fold-improvement of ClonalGE over STdeconvolve",
         size=26, bold=True, color=BLUE)
add_text(slide, Inches(0.4), Inches(0.7), Inches(12.5), Inches(0.4),
         "Mean MAE across 20 simulation replicates per condition",
         size=14, color=GREY, italic=True)

# ── Table: H MAE ─────────────────────────────────────────────────────────────
add_text(slide, Inches(0.4), Inches(1.2), Inches(6), Inches(0.35),
         "Clone proportion recovery (H MAE)", size=14, bold=True, color=MAROON)

rows_h = [
    ["Condition",     "ClonalGE",          "STdeconvolve",       "Fold ↑"],
    ["Normal",        "0.0301 ± 0.0107",   "0.0682 ± 0.0187",    "2.3×"],
    ["Low Variance",  "0.0174 ± 0.0021",   "0.0463 ± 0.0165",    "2.7×"],
    ["High Coverage", "0.0191 ± 0.0018",   "0.0683 ± 0.0241",    "3.6×"],
]
table_h = slide.shapes.add_table(rows=4, cols=4,
                                  left=Inches(0.4), top=Inches(1.6),
                                  width=Inches(6.2), height=Inches(1.6)).table
for c in range(4):
    table_h.columns[c].width = Inches([1.5, 1.7, 1.7, 1.3][c])
for i, row in enumerate(rows_h):
    for j, cell_text in enumerate(row):
        cell = table_h.cell(i, j)
        cell.text = cell_text
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(11)
        para.font.bold = (i == 0) or (j == 0) or (j == 3)
        if i == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE
            para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        else:
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            if j == 3:
                para.font.color.rgb = GREEN

# ── Table: B MAE ─────────────────────────────────────────────────────────────
add_text(slide, Inches(0.4), Inches(3.5), Inches(6), Inches(0.35),
         "Clone-specific expression recovery (B MAE)",
         size=14, bold=True, color=MAROON)

rows_b = [
    ["Condition",     "ClonalGE",          "STdeconvolve",       "Fold ↑"],
    ["Normal",        "0.0015 ± 0.0002",   "0.0045 ± 0.0018",    "3.0×"],
    ["Low Variance",  "0.0006 ± 0.0001",   "0.0032 ± 0.0016",    "5.2×"],
    ["High Coverage", "0.0014 ± 0.0001",   "0.0045 ± 0.0026",    "3.2×"],
]
table_b = slide.shapes.add_table(rows=4, cols=4,
                                  left=Inches(0.4), top=Inches(3.9),
                                  width=Inches(6.2), height=Inches(1.6)).table
for c in range(4):
    table_b.columns[c].width = Inches([1.5, 1.7, 1.7, 1.3][c])
for i, row in enumerate(rows_b):
    for j, cell_text in enumerate(row):
        cell = table_b.cell(i, j)
        cell.text = cell_text
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(11)
        para.font.bold = (i == 0) or (j == 0) or (j == 3)
        if i == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE
            para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        else:
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            if j == 3:
                para.font.color.rgb = GREEN

# Interpretation panel (right)
add_multiline(slide, Inches(7.0), Inches(1.2), Inches(6.0), Inches(3.0),
              "Key observations", [
                  "",
                  "• ClonalGE has 2–4× lower error for clone proportions",
                  "   in every condition.",
                  "",
                  "• Even larger advantage for clone-specific expression",
                  "   (3–5× lower MAE).",
                  "",
                  "• Largest gain in High Coverage setting:",
                  "   deeper sequencing → stronger allele-frequency signal,",
                  "   which only ClonalGE can exploit.",
              ], title_size=14, title_color=MAROON, line_size=12)

# Conclusion box (right bottom)
concl = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(7.0), Inches(4.5), Inches(6.0), Inches(2.4))
concl.fill.solid()
concl.fill.fore_color.rgb = GREEN
concl.line.fill.background()
tf = concl.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.2)
tf.margin_right = Inches(0.2)
tf.margin_top = Inches(0.15)
p = tf.paragraphs[0]
p.text = "Conclusion"
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p = tf.add_paragraph()
p.text = (
    "Integrating somatic mutations is not merely complementary to "
    "expression-based deconvolution — it is essential for accurately "
    "recovering tumour clonal composition in spatial transcriptomics."
)
p.font.size = Pt(12)
p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
p.space_before = Pt(8)

# Reference
add_text(slide, Inches(0.4), Inches(7.15), Inches(12), Inches(0.3),
         "Source: Supplementary Tables S2/S3 and Supplementary Figure (STdeconvolve comparison) in the manuscript.",
         size=9, color=RGBColor(0x88, 0x88, 0x88))


# ── Save ──────────────────────────────────────────────────────────────────────
out = '/Users/darvis01/Documents/ClonalGE/STdeconvolve_benchmark.pptx'
prs.save(out)
print(f"Saved → {out}")
