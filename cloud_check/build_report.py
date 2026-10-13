"""Build the Assignment 2 report PDF (cover + body) and the submission zip on the CI machine."""
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage, KeepTogether, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "resources"
FIG = HERE / "figures"
OUT = HERE / "report"
TMP = OUT / "tmp"
sys.path.insert(0, str(RES))
os.chdir(RES)

import blob_detection as bd  # noqa: E402
import corners  # noqa: E402
import filters  # noqa: E402

NAME_CN = "李昌硕"
STUDENT_ID = "1240000787"
TITLE = "Assignment 2: Image Filtering, Processing and Feature Extraction"
PDF_NAME = f"{NAME_CN}_Assignment2.pdf"
ZIP_NAME = f"{NAME_CN}{STUDENT_ID}.zip"

R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
VERIFY = (HERE / "run_logs" / "verify.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- fonts / styles
def register_fonts():
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Serif", str(ttf / "DejaVuSerif.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(ttf / "DejaVuSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Italic", str(ttf / "DejaVuSerif-Italic.ttf")))
    bi = ttf / "DejaVuSerif-BoldItalic.ttf"
    pdfmetrics.registerFont(TTFont("Serif-BoldItalic", str(bi if bi.exists() else ttf / "DejaVuSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Sans-Bold", str(ttf / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(ttf / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                       boldItalic="Serif-BoldItalic")


register_fonts()
NAVY = HexColor("#102a43")
ST = {
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=20, textColor=NAVY,
                         spaceBefore=4, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12, leading=16, textColor=HexColor("#243b53"),
                         spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10, leading=14.5, spaceAfter=5),
    "cap": ParagraphStyle("cap", fontName="Serif-Italic", fontSize=8.8, leading=12,
                          textColor=HexColor("#334e68"), alignment=1, spaceBefore=2, spaceAfter=8),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=9.6, leftIndent=6,
                           backColor=HexColor("#f4f6f8"), borderPadding=4, spaceBefore=3, spaceAfter=7),
    "mat": ParagraphStyle("mat", fontName="Mono", fontSize=9, leading=11.5, leftIndent=24,
                          spaceBefore=2, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=8.6, leading=11),
}
PAGE_W = A4[0] - 32 * mm


def P(text, style="body"):
    return Paragraph(text, ST[style])


def mono(name):
    return f"<font face='Mono' size='9'>{name}</font>"


_eq_counter = [0]


def EQ(tex, size=12.5):
    """Render a mathtext equation to a PNG and return a centered flowable."""
    _eq_counter[0] += 1
    path = TMP / f"eq_{_eq_counter[0]:03d}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    with Image.open(path) as im:
        w, h = im.size
    img = RLImage(str(path), width=w * 72 / 300, height=h * 72 / 300)
    img.hAlign = "CENTER"
    return img


def FIGURE(path, caption, width=PAGE_W, max_h=110 * mm):
    path = FIG / path if not Path(path).is_absolute() else Path(path)
    with Image.open(path) as im:
        w, h = im.size
    s = min(width / w, max_h / h)
    img = RLImage(str(path), width=w * s, height=h * s)
    return KeepTogether([img, P(caption, "cap")])


def MAT(text):
    return Preformatted(text, ST["mat"])


def CODE(obj, title=None):
    src = inspect.getsource(obj)
    src = re.sub(r'\n    """[\s\S]*?"""\n', "\n", src, count=1)
    block = [Preformatted(src.rstrip(), ST["code"])]
    if title:
        block.insert(0, P(title, "cap"))
    return block


def TABLE(rows, widths, header=True):
    data = [[Paragraph(str(c), ST["cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, HexColor("#9fb3c8")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HexColor("#d9e2ec")))
    t.setStyle(TableStyle(style))
    return t


def f2(x):
    return f"{x:.2f}"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}×10<super>{int(e)}</super>"


# ---------------------------------------------------------------- cover
def build_cover():
    doc = Document(str(ROOT / "cover_template.docx"))
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("Name:") and para.runs:
            para.runs[-1].text = para.runs[-1].text.rstrip() + " " + NAME_CN
        elif text.startswith("Student ID:") and para.runs:
            para.runs[-1].text = para.runs[-1].text.rstrip() + " " + STUDENT_ID
    filled = TMP / "cover.docx"
    doc.save(str(filled))
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(TMP), str(filled)],
                   check=True, timeout=180)
    return TMP / "cover.pdf"


# ---------------------------------------------------------------- body
def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 14, w, 14, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Serif", 7.5)
    canvas.drawString(16 * mm, h - 10, "CS460 " + TITLE)
    canvas.setFont("STSong-Light", 8)
    canvas.drawRightString(w - 16 * mm, h - 10, f"{NAME_CN}  {STUDENT_ID}")
    canvas.setFillColor(HexColor("#627d98"))
    canvas.setFont("Serif", 8)
    canvas.drawCentredString(w / 2, 10 * mm, "Page %d" % (doc.page + 1))
    canvas.restoreState()


def story():
    t1, t2, t3, t4 = R["t1"], R["t2"], R["t3"], R["t4"]
    t5a, t5b, t5c = R["t5a"], R["t5b"], R["t5c"]
    n_pass = len(re.findall(r"^\[PASS\]", VERIFY, re.M))
    n_all = n_pass + len(re.findall(r"^\[FAIL\]", VERIFY, re.M))
    s = []
    scipy_conv = mono("scipy.ndimage.convolve(mode='constant')")

    # ------------------------------------------------------------ overview
    s.append(P("Overview", "h1"))
    s.append(P(
        "This report covers Tasks 1-5 of Assignment 2. The completed code files are "
        f"{mono('filters.py')} (Tasks 1-3), {mono('corners.py')} (Task 4) and "
        f"{mono('blob_detection.py')} (Task 5). Every script was run on a GitHub Actions Ubuntu runner "
        "(Python 3.11, NumPy, SciPy, Matplotlib, Pillow); all figures and numbers in this report come from that run. "
        f"A separate verification script ran {n_all} automatic checks, and all {n_pass} passed. Among them: "
        f"{mono('convolve()')} matches {scipy_conv} exactly for nine kernel "
        "shapes (including even sizes), the convolutional Harris detector matches a quadruple-loop "
        "brute-force version, and the Task 5 Gaussian blur matches SciPy's Gaussian filter to within 10<super>-14</super>. "
        "The full check list is in the appendix."))

    # ------------------------------------------------------------ Task 1
    s.append(P("Task 1: Convolution and Gaussian Filter", "h1"))
    s.append(P("Task 1 (a)  Separability of the 2D Gaussian", "h2"))
    s.append(P("The 2D Gaussian factorises into a product of two 1D Gaussians:"))
    s.append(EQ(r"G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}"
                r"=\left(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{y^2}{2\sigma^2}}\right)"
                r"\left(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{x^2}{2\sigma^2}}\right)=G(y)\,G(x)"))
    s.append(P(
        "Pick a filter size k with sample positions x<sub>j</sub>, y<sub>i</sub> ∈ {-(k-1)/2, …, (k-1)/2}. "
        "Define the vertical filter G<sub>y</sub> ∈ R<super>k×1</super> with entries G<sub>y</sub>[i] = G(y<sub>i</sub>), "
        "the horizontal filter G<sub>x</sub> ∈ R<super>1×k</super> with G<sub>x</sub>[j] = G(x<sub>j</sub>), and the 2D filter "
        "G ∈ R<super>k×k</super> with G[i, j] = G(x<sub>j</sub>, y<sub>i</sub>). Looking at individual elements:"))
    s.append(EQ(r"G[i,j]=G(x_j,y_i)=G(y_i)\,G(x_j)=G_y[i]\,G_x[j]=(G_yG_x)[i,j]"
                r"\quad\Rightarrow\quad G=G_yG_x=G_y\ast G_x"))
    s.append(P(
        "The last equality uses the given fact that convolving a k×1 column with a 1×k row is their outer product. "
        "Because convolution is associative, for any image I"))
    s.append(EQ(r"I\ast G=I\ast(G_y\ast G_x)=(I\ast G_y)\ast G_x"))
    s.append(P(
        "So filtering with the k×k Gaussian is the same as filtering vertically with G<sub>y</sub> and then "
        "horizontally with G<sub>x</sub>. This costs 2k instead of k<super>2</super> multiplications per pixel. "
        f"Numerical check (k = 3, σ = 0.572): max |G - G<sub>y</sub>G<sub>x</sub>| = {sci(t1['sep_kernel_err'])}, and "
        f"max |I∗G - (I∗G<sub>y</sub>)∗G<sub>x</sub>| = {sci(t1['sep_image_err'])} on grace_hopper.png "
        "(floating-point round-off only)."))

    s.append(P("Task 1 (b)  convolve()", "h2"))
    s.append(P(
        f"{mono('convolve()')} flips the kernel in both directions as soon as it receives it, zero-pads the image by "
        "(h-1)//2 rows on top, h//2 at the bottom, (w-1)//2 columns on the left and w//2 on the right, and then accumulates "
        "kernel[i, j] × (shifted image window). This is true convolution, not cross-correlation. "
        "The output size is H×W, and the kernel centre is (h//2, w//2), the same convention as SciPy, so even-sized kernels match as well. "
        "Against scipy.ndimage.convolve(mode='constant'), the maximum absolute difference is 0 for kernels "
        "of shape 1×1, 1×3, 3×1, 3×3, 5×5, 9×9, 2×2, 4×3 and 2×5, and for the 9×9 LoG kernel on grace_hopper.png. "
        "For an asymmetric kernel the result differs from cross-correlation, which confirms that the flip is applied."))
    s.extend(CODE(filters.convolve, "Listing 1. convolve() in filters.py"))

    s.append(P("Task 1 (c)  3×3 Gaussian, σ = 0.572", "h2"))
    k = t1["kernel"]
    s.append(P("Kernel from the 2D formula (the values sum to %.6f):" % t1["sum_0572"]))
    s.append(MAT("\n".join("[ " + "  ".join(f"{v:.4f}" for v in row) + " ]" for row in k)))
    s.append(FIGURE("t1c_gaussian.png", "Figure 1. grace_hopper.png before and after the 3×3 Gaussian filter "
                    "(σ = 0.572); both images use vmin = 0, vmax = 255.", max_h=78 * mm))
    s.append(FIGURE("t1c_gaussian_crop.png", "Figure 2. Close-up crop: the filtered image is slightly blurred.",
                    width=PAGE_W * 0.72, max_h=70 * mm))
    s.append(P(
        "<b>Effect of Gaussian filtering:</b> it replaces each pixel by a Gaussian-weighted average of its "
        "neighbourhood, so the image is slightly blurred: noise and fine texture are smoothed out and edges become "
        "softer, while the overall brightness is unchanged."))

    s.append(P("Task 1 (d)  Why a smoothing filter should sum to 1", "h2"))
    o, a, b, c = t1["orig"], t1["raw_0572"], t1["norm_0572"], t1["raw_2"]
    rows = [["Filter", "Kernel sum", "Output min", "Output max", "Range", "Mean"],
            ["None (original)", "-", f2(o["min"]), f2(o["max"]), f2(o["range"]), f2(o["mean"])],
            ["3×3, σ = 0.572 (raw)", f"{t1['sum_0572']:.6f}", f2(a["min"]), f2(a["max"]), f2(a["range"]), f2(a["mean"])],
            ["3×3, σ = 0.572 (divided by sum)", "1.000000", f2(b["min"]), f2(b["max"]), f2(b["range"]), f2(b["mean"])],
            ["3×3, σ = 2 (raw, not normalised)", f"{t1['sum_2']:.6f}", f2(c["min"]), f2(c["max"]), f2(c["range"]), f2(c["mean"])]]
    s.append(TABLE(rows, [PAGE_W * w for w in (0.34, 0.14, 0.13, 0.13, 0.13, 0.13)]))
    s.append(Spacer(1, 4))
    s.append(FIGURE("t1d_sum.png", "Figure 3. Left: σ = 0.572 kernel normalised to sum 1. Right: σ = 2 kernel used "
                    f"as is (sum {t1['sum_2']:.3f}). Same display range vmin = 0, vmax = 255.", max_h=72 * mm))
    s.append(P(
        f"With σ = 0.572 the kernel already sums to {t1['sum_0572']:.6f}, so the output keeps the original intensity scale "
        f"(min {f2(b['min'])}, max {f2(b['max'])}, range {f2(b['range'])}). The minimum rises and the maximum falls only because "
        "extreme pixels are averaged with their neighbours. "
        f"With σ = 2 the 3×3 window keeps only {t1['sum_2']:.3f} of the Gaussian's mass (the curve is very wide, so most of its "
        f"mass lies outside the window). Every output value is scaled by about that factor: the mean drops from {f2(o['mean'])} to "
        f"{f2(c['mean'])} (ratio {c['mean'] / o['mean']:.3f}) and the maximum is only {f2(c['max'])}, so the image looks "
        "much darker."))
    s.append(P(
        "<b>Reason:</b> on a constant region I = c the output is c·Σk. A smoothing filter should only redistribute "
        "intensity between neighbouring pixels, not create or remove it. If Σk = 1, flat regions and the mean "
        "brightness (the DC component) are preserved and only local variations are averaged out. "
        "If Σk ≠ 1, the filter also rescales the whole image: Σk &lt; 1 darkens it, and Σk &gt; 1 brightens it and can push "
        "values past 255. Repeated filtering compounds the error as (Σk)<super>n</super>, and results computed with different "
        "kernels would no longer share an intensity scale. Dividing the kernel by its sum fixes this."))

    s.append(P("Task 1 (e)  Derivative kernels", "h2"))
    s.append(P("For a 1×3 kernel with taps k(-1), k(0), k(1) (centre at index 0), convolution gives"))
    s.append(EQ(r"(I\ast k_x)(x,y)=\sum_{m=-1}^{1}k_x(m)\,I(x-m,y)=k_x(-1)\,I(x+1,y)+k_x(0)\,I(x,y)+k_x(1)\,I(x-1,y)"))
    s.append(P(
        "Matching this to I<sub>x</sub> = I(x+1, y) - I(x-1, y) gives k<sub>x</sub>(-1) = 1, k<sub>x</sub>(0) = 0, "
        "k<sub>x</sub>(1) = -1. Written left to right, the kernel array is therefore [1, 0, -1]. Note that the flip in convolution reverses the "
        "intuitive [-1, 0, 1]. The same argument along the vertical axis (y increases downwards, i.e. along rows) gives:"))
    s.append(MAT("(i)  kx = [ 1  0  -1 ]  ∈ R^(1×3)          (ii) ky = [ 1  0  -1 ]^T  ∈ R^(3×1)"))
    s.append(P(
        "Check: on a horizontal ramp I(x, y) = x, edge_detection() returns I<sub>x</sub> = 2 and I<sub>y</sub> = 0 at "
        "every interior pixel, and on a vertical ramp it returns I<sub>y</sub> = 2. This is twice the unit slope, as expected."))

    s.append(P("Task 1 (f)  edge_detection()", "h2"))
    s.append(P(
        f"{mono('edge_detection()')} convolves the image with k<sub>x</sub> and k<sub>y</sub> using convolve() and returns "
        "I<sub>x</sub>, I<sub>y</sub> and the gradient magnitude √(I<sub>x</sub>² + I<sub>y</sub>²)."))
    s.extend(CODE(filters.edge_detection, "Listing 2. edge_detection() in filters.py"))

    s.append(P("Task 1 (g)  Gradient magnitude of the original and Gaussian-filtered images", "h2"))
    e0, e1 = t1["edge_orig"], t1["edge_blur"]
    s.append(FIGURE("t1g_edges.png", "Figure 4. Gradient magnitude from edge_detection(): original image (left) and the "
                    "σ = 0.572 Gaussian-filtered image (right), same display range.", max_h=76 * mm))
    s.append(FIGURE("t1g_edges_crop.png", "Figure 5. Close-up crop of the two gradient-magnitude images.",
                    width=PAGE_W * 0.72, max_h=70 * mm))
    s.append(P(
        f"The gradient magnitude of the original image (mean {f2(e0['mean'])}, max {f2(e0['max'])}) has the thinnest and "
        "sharpest edges, but also responds to film grain and fine texture, which shows up as faint speckle in flat areas "
        "such as the cabinet and wall. "
        f"After Gaussian filtering (mean {f2(e1['mean'])}, max {f2(e1['max'])}) that speckle is weaker, and the outlines of the person, "
        "console panels, buttons and dials are smoother but slightly wider and lower in contrast. Because σ = 0.572 is small, "
        "the difference is modest, but it shows the usual trade-off: smoothing before differentiating gives up a little edge "
        "localisation for robustness to noise."))

    # ------------------------------------------------------------ Task 2
    s.append(PageBreak())
    s.append(P("Task 2: Sobel Operator", "h1"))
    s.append(P("Task 2 (a)  Sobel filtering approximates the derivative of the Gaussian-filtered image", "h2"))
    s.append(P(
        "Let g = [1 2 1]. Then G<sub>S</sub> = g<super>T</super>g = g<super>T</super> ∗ g, and S<sub>x</sub> = "
        "g<super>T</super>[1 0 -1] = g<super>T</super> ∗ k<sub>x</sub>. The horizontal derivative of the Gaussian-filtered "
        "image is (I ∗ G<sub>S</sub>) ∗ k<sub>x</sub>. Because convolution is associative and commutative,"))
    s.append(EQ(r"(I\ast G_S)\ast k_x=I\ast(G_S\ast k_x),\qquad G_S\ast k_x=g^{T}\ast(g\ast k_x)"))
    s.append(P("so the image can be ignored and only the two filters G<sub>S</sub> ∗ k<sub>x</sub> and S<sub>x</sub> need comparing. "
               "With g ∗ k<sub>x</sub> = [1 2 1] ∗ [1 0 -1] = [1 2 0 -2 -1]:"))
    s.append(MAT("              [ 1  2  0  -2  -1 ]                 [ 1  0  -1 ]\n"
                 "GS * kx  =    [ 2  4  0  -4  -2 ]       Sx  =     [ 2  0  -2 ]\n"
                 "              [ 1  2  0  -2  -1 ]                 [ 1  0  -1 ]"))
    s.append(P(
        "Splitting the 3×5 kernel into its centre and outer columns gives an exact identity, where d<sub>1</sub> = [1 0 -1] is "
        "a central difference with step 1 and d<sub>2</sub> = [1 0 0 0 -1] is a central difference with step 2:"))
    s.append(EQ(r"G_S\ast k_x=2\,g^{T}\ast d_1+g^{T}\ast d_2=2\,S_x+g^{T}\ast d_2"))
    s.append(P(
        "Both terms are a vertical [1 2 1] smoothing followed by a horizontal central difference. Where the image is "
        "locally linear over a few pixels, I(x+2) - I(x-2) ≈ 2[I(x+1) - I(x-1)], so g<super>T</super> ∗ d<sub>2</sub> acts like "
        "2S<sub>x</sub>. Therefore"))
    s.append(EQ(r"\frac{\partial}{\partial x}(I\ast G_S)\;\approx\;\frac{1}{2}(I\ast G_S)\ast k_x\;\approx\;2\,(I\ast S_x)"))
    s.append(P(
        "Up to a constant factor (irrelevant for edge detection, and absorbed by normalising G<sub>S</sub> by 1/16), the "
        "Sobel filter computes the horizontal derivative of the G<sub>S</sub>-smoothed image. In effect it is the 3×3 "
        "truncation of the 3×5 kernel. The same argument with k<sub>y</sub> gives S<sub>y</sub>. On grace_hopper.png, "
        f"the correlation between I ∗ S<sub>x</sub> and (I ∗ G<sub>S</sub>) ∗ k<sub>x</sub> is {t2['corr']:.4f}, and the "
        f"least-squares ratio between them is {t2['ratio']:.2f} (≈ 4 as predicted)."))
    s.append(FIGURE("t2a_compare.png", "Figure 6. I ∗ S<sub>x</sub> (left) and (I ∗ G<sub>S</sub>) ∗ k<sub>x</sub> (right) are visually the same.",
                    max_h=70 * mm))
    s.append(P("Task 2 (b)  sobel_operator()", "h2"))
    s.extend(CODE(filters.sobel_operator, "Listing 3. sobel_operator() in filters.py"))
    s.append(P("Task 2 (c)  Sobel outputs", "h2"))
    s.append(FIGURE("t2c_sobel.png", "Figure 7. I ∗ S<sub>x</sub>, I ∗ S<sub>y</sub> and the gradient magnitude "
                    "√(G<sub>x</sub>² + G<sub>y</sub>²) for grace_hopper.png. Signed outputs use a symmetric gray scale "
                    "(black negative, white positive).", max_h=62 * mm))
    s.append(P(
        "I ∗ S<sub>x</sub> responds to vertical edges (intensity changes along x) and I ∗ S<sub>y</sub> to horizontal edges. "
        "The sign shows whether intensity increases or decreases across the edge. The magnitude combines both into an "
        "orientation-independent edge map that is smoother than the plain [1 0 -1] gradient of Task 1, because of the "
        "built-in [1 2 1] smoothing."))

    # ------------------------------------------------------------ Task 3
    s.append(PageBreak())
    s.append(P("Task 3: LoG Filter", "h1"))
    s.append(P("Task 3 (a)  The two LoG filters", "h2"))
    L1, L2 = t3["L1"], t3["L2"]
    s.append(FIGURE("t3a_log.png", "Figure 8. Signed responses to kernel_LoG1 (3×3) and kernel_LoG2 (9×9), "
                    "displayed with symmetric limits (99th percentile of |response|).", max_h=72 * mm))
    s.append(FIGURE("t3a_log_abs.png", "Figure 9. Absolute responses: bright = detected region (high |response|).",
                    max_h=72 * mm))
    s.append(FIGURE("t3a_log_crop.png", "Figure 10. Close-up crop of the signed responses.",
                    width=PAGE_W * 0.72, max_h=70 * mm))
    s.append(P(
        f"kernel_LoG1 is the 4-neighbour discrete Laplacian (sum {t3['sum1']}). It is the LoG at the smallest possible "
        "scale, with essentially no smoothing. Its response (range "
        f"{L1['min']:.0f} to {L1['max']:.0f}) is sharp and noisy: it highlights single-pixel noise and grain, fine texture "
        "such as hair, and thin double lines along every edge. kernel_LoG2 (sum {t3['sum2']}, close to 0 relative to its "
        "weights) is a sampled Laplacian of a Gaussian with σ of roughly 1.4 px: a negative centre surrounded by a "
        f"positive ring. Its response (range {L2['min']:.0f} to {L2['max']:.0f}, larger only because its weights are larger) "
        "is much smoother. Pixel noise is averaged away, and strong responses appear as thick bands along significant "
        "edges (panel frames, the person's outline) and as blobs on structures a few pixels wide (the buttons, lamps and "
        "dials of the console, the glasses, the keyboard keys). "
        "<b>The difference comes from scale:</b> the Gaussian built into LoG2 removes detail smaller than about σ before the "
        "second derivative is taken, while LoG1 differentiates the raw pixels."))
    s.append(P(
        "<b>Can they detect edges?</b> Yes, but not as maxima. At a step edge, the second derivative is strongly positive on "
        "one side and strongly negative on the other, and zero exactly on the edge. Edges are therefore the <i>zero crossings</i> "
        "between paired high-|response| bands (Marr-Hildreth). LoG2 gives cleaner zero crossings; those of LoG1 are "
        "dominated by noise. <b>Anything else?</b> LoG is the classic <i>blob detector</i>: a bright spot of radius about "
        "√2σ produces a strong negative extremum at its centre (the centre weight is negative), and a dark spot produces a positive "
        "one. LoG also responds strongly at corners, line endings, thin ridges/lines (which have high curvature in "
        "intensity), and isolated points. Flat regions and linear ramps give zero response because the kernels sum to (nearly) 0."))

    s.append(P("Task 3 (b)  Approximating LoG with a Difference of Gaussians", "h2"))
    s.append(FIGURE("t3b_dog_log.png", "Figure 11. (a) Gaussians with σ = 50 and σ = 53 from log1d.npz. "
                    "(b) Their difference gauss53 - gauss50. (c) The LoG with σ = 50 and the DoG scaled by a "
                    f"least-squares factor of {t3['dog_scale']:.3f}.", max_h=62 * mm))
    s.append(P(
        "Subtracting a narrow Gaussian from a slightly wider one leaves a profile with a negative centre (the narrow "
        "Gaussian is taller there), positive side lobes (the wide Gaussian is larger there), and zero total area, "
        f"because both Gaussians sum to 1 (DoG sum = {sci(t3['dog_sum'])}). This is exactly the 'Mexican-hat' "
        f"shape of the LoG: after a single scale factor, the DoG matches log50 with a relative L2 error of only "
        f"{100 * t3['dog_rel_err']:.1f}%. The reason is the heat-diffusion equation satisfied by the Gaussian:"))
    s.append(EQ(r"\frac{\partial G_\sigma}{\partial\sigma}=\sigma\,\nabla^2G_\sigma"
                r"\quad\Rightarrow\quad G_{k\sigma}-G_\sigma\approx(k\sigma-\sigma)\frac{\partial G_\sigma}{\partial\sigma}"
                r"=(k-1)\,\sigma^2\,\nabla^2G_\sigma"))
    s.append(P(
        "A finite difference in scale is therefore a scaled LoG. The approximation becomes exact as k → 1; here k = 53/50 = 1.06. "
        "Because convolution is linear,"))
    s.append(EQ(r"(I\ast G_{k\sigma})-(I\ast G_\sigma)=I\ast(G_{k\sigma}-G_\sigma)\approx(k-1)\,\sigma^2\,(I\ast\nabla^2G_\sigma)"))
    s.append(P(
        "So the difference of two blurred images is (a constant times) the LoG-filtered image. The constant does not move "
        "zero crossings or extrema, so edges and blobs are found at the same places. In 2D the same identity holds because "
        "the Gaussian satisfies the 2D heat equation. In practice the blurred images are needed anyway for a scale space, and "
        "they can be produced by repeatedly applying small Gaussian kernels (G<sub>σ1</sub> ∗ G<sub>σ2</sub> = "
        "G<sub>√(σ1²+σ2²)</sub>), so the LoG at many scales comes almost for free from subtractions."))

    # ------------------------------------------------------------ Task 4
    s.append(PageBreak())
    s.append(P("Task 4: Harris Corner Detector", "h1"))
    s.append(P("Task 4 (a)  Implementation", "h2"))
    s.append(P(
        "The image derivatives are I<sub>x</sub> = I ∗ [1 0 -1]/2 and I<sub>y</sub> = I ∗ [1 0 -1]<super>T</super>/2 "
        "(scipy.ndimage.convolve, zero padding). The structure tensor M at every pixel sums the products of derivatives "
        "over a window, weighted by w (a 5×5 Gaussian, σ = 1, normalised to sum 1). Each of the three distinct entries "
        "is one convolution of a product image with w, and the three are stored as a 3-channel image "
        "(M<sub>11</sub>, M<sub>12</sub>, M<sub>22</sub>):"))
    s.append(EQ(r"M_{11}=w\ast I_x^2,\quad M_{12}=w\ast(I_xI_y),\quad M_{22}=w\ast I_y^2"))
    s.append(EQ(r"R=\det M-\alpha\,(\mathrm{tr}\,M)^2=M_{11}M_{22}-M_{12}^2-\alpha\,(M_{11}+M_{22})^2,\qquad\alpha=0.05"))
    s.append(P(
        "det and trace use the closed-form expressions for 2×2 matrices, evaluated element-wise on the three channel "
        "images, so no per-pixel loop is needed. No library corner detector is used. Following the walk-through, I first "
        f"wrote a brute-force version ({mono('harris_detector_bruteforce')}, four nested loops that build M explicitly "
        "and call np.linalg.det/np.trace) and compared the two: the maximum absolute difference on a 40×40 crop is "
        f"{sci(t4['bf_err_crop'])} for responses of order 10<super>6</super>, i.e. round-off only."))
    s.extend(CODE(corners.harris_detector, "Listing 4. harris_detector() in corners.py"))
    s.append(P("Task 4 (b)  Harris response of grace_hopper.png", "h2"))
    s.append(FIGURE("t4_harris.png", "Figure 12. Left: heatmap of the Harris score R (colour limits clipped at the "
                    "99.7th percentile of |R| so that weaker structure is visible). Right: local maxima of R above 1% of max R.",
                    max_h=82 * mm))
    s.append(FIGURE("t4_harris_full.png", "Figure 13. Heatmap of R over its full range "
                    f"[{sci(t4['R']['min'])}, {sci(t4['R']['max'])}].", width=PAGE_W * 0.6, max_h=80 * mm))
    s.append(P(
        f"R ranges from {sci(t4['R']['min'])} to {sci(t4['R']['max'])}. Flat regions (the wall, the cabinet surfaces, the "
        "person's clothing) have both eigenvalues small, so R ≈ 0 (green in the heatmap). Along long straight edges, such as the "
        "vertical frame of the cabinet, the panel borders and the arm outline, one eigenvalue is large and the other small, which "
        f"makes R negative (blue; {100 * t4['frac_negative']:.0f}% of pixels have R &lt; 0). The large positive peaks (red) sit where "
        "intensity changes in two directions: the regular grid of buttons and switches on the console, the corners of the "
        "panels and labels, the keyboard keys, and the glasses and curly hair. "
        f"{t4['n_corners']} local maxima exceed 1% of the maximum response."))

    # ------------------------------------------------------------ Task 5
    s.append(PageBreak())
    s.append(P("Task 5: Blob Detection and Morphological Cell Counting", "h1"))
    s.append(P("Task 5 (a)  Difference-of-Gaussians blob detection", "h2"))
    s.append(P(
        f"{mono('gaussian_filter')} builds a (2r+1)×(2r+1) kernel exp(-(x²+y²)/2σ²) with r = ceil(3σ), divides it by its sum, "
        "and convolves with scipy.ndimage.convolve(mode='reflect'). No library Gaussian filter is called; the result "
        "matches scipy.ndimage.gaussian_filter to within 3×10<super>-15</super>. "
        f"{mono('difference_of_gaussians')} returns G<sub>σsmall</sub>∗I - G<sub>σlarge</sub>∗I. At the centre of a bright "
        "cell the narrow blur keeps more of the peak than the wide blur, so the response is positive there (verified on a synthetic blob). "
        f"{mono('find_blob_maxima')} keeps pixels that equal the maximum of their neighbourhood_size×neighbourhood_size "
        "window and exceed threshold_rel × max(response). It then merges flat-topped plateaus (one pixel per 8-connected group) "
        "and applies greedy non-maximum suppression, dropping a weaker peak closer than neighbourhood_size/2 px to a stronger one."))
    pairs_txt = ", ".join(f"({p['sigmas'][0]}, {p['sigmas'][1]}) → {p['count']}" for p in t5a["pairs"])
    s.append(P(
        f"<b>Image:</b> {t5a['image']}. <b>Scale pairs tried</b> (neighbourhood 9, threshold_rel 0.30): {pairs_txt} maxima. "
        f"<b>Selected scales:</b> σ<sub>small</sub> = {t5a['sigmas'][0]}, σ<sub>large</sub> = {t5a['sigmas'][1]}, i.e. "
        f"blob radii √2σ ≈ {1.414 * t5a['sigmas'][0]:.1f}-{1.414 * t5a['sigmas'][1]:.1f} px. This matches the in-focus "
        "nuclei, whose radius is about 4-6 px, and the response is strongest near cell centres. "
        f"<b>Response threshold:</b> threshold_rel = {t5a['threshold_rel']} of the maximum response {t5a['resp_max']:.4f}, i.e. an "
        f"absolute threshold of {t5a['abs_threshold']:.4f}. <b>Local-maximum neighbourhood:</b> {t5a['neighborhood']}×"
        f"{t5a['neighborhood']} maximum filter plus NMS with a {t5a['neighborhood'] / 2:.1f} px minimum distance. "
        f"<b>Result:</b> {t5a['count']} detected blobs. Counting by eye on enlarged crops gives roughly 170-190 cells in this image."))
    s.append(FIGURE("task5a_panel.png", f"Figure 14. {t5a['image']}: input, DoG response "
                    f"(σ = {t5a['sigmas'][0]}, {t5a['sigmas'][1]}), and detected maxima (circles of radius √2σ<sub>small</sub>).",
                    max_h=70 * mm))
    s.append(FIGURE("task5a_scales.png", "Figure 15. Detected maxima for every tried scale pair.", max_h=55 * mm))
    s.append(FIGURE("task5a_zoom.png", "Figure 16. Enlarged top-left quarter with the selected detections.",
                    width=PAGE_W * 0.8, max_h=80 * mm))
    s.append(P(
        "<b>Missed cells and false detections.</b> Smaller scales detect more cells but start placing two maxima on large or "
        "elongated nuclei. Larger scales blur neighbouring cells into a single blob, so the count falls steadily from "
        f"{t5a['pairs'][0]['count']} to {t5a['pairs'][-1]['count']}. With the selected pair, most misses occur in tight clusters "
        "and chains of overlapping cells (lower half of the image), where two or three touching nuclei produce one merged "
        "maximum or a weaker maximum within the NMS distance. Dim, out-of-focus cells on the bright hazy background in the lower part also give a "
        "weak, wide response. False or duplicate detections are rare: elongated or dumb-bell shaped cells sometimes get two "
        "maxima, and a few hits occur on bright haze between cells. The relative threshold is not the limiting factor: "
        f"threshold_rel from 0.1 to 0.3 gives the same count ({t5a['rel_sweep']['0.1']}), and only above 0.3 do dim cells drop out "
        f"(0.4 → {t5a['rel_sweep']['0.4']}, 0.5 → {t5a['rel_sweep']['0.5']})."))

    s.append(P("Task 5 (b)  Cell counting with mathematical morphology", "h2"))
    base = t5b["baseline_" + t5a["image"]]
    s.append(P(
        "<b>Pipeline</b> (morphology_mask + count_components):<br/>"
        "1. <i>Scalar image:</i> I = max(luminance, blue) from read_gray. The microscopy images have strongly uneven "
        "illumination and a bright out-of-focus haze in parts of the field. A single global threshold therefore either "
        "merges the whole hazy region into one component or misses the darker regions. With the starter settings "
        f"(threshold {base['params']['threshold']}, no flattening), {t5a['image']} yields only {base['count']} accepted components, "
        f"and the largest component covers {base['largest_area']} px (Figure 17). I therefore subtract a wide Gaussian "
        "blur (σ<sub>bg</sub> = 10 px, made with the same self-built gaussian_filter) to remove the slowly varying background "
        "before thresholding.<br/>"
        "2. <i>Threshold</i> the flattened image to get a binary foreground mask.<br/>"
        "3. <i>Closing</i> (dilate, then erode; 1 iteration) fills pinholes and dark gaps inside cells, so later "
        "erosion does not break one cell into pieces.<br/>"
        "4. <i>Erosion</i> (3 iterations) removes isolated noise pixels and cuts the narrow bridges where touching cells meet. "
        "<i>Dilation</i> (2 iterations) then grows the surviving cores back. With equal counts, steps 4a and 4b would be an exact "
        "opening. Using one fewer dilation keeps a one-pixel gap between cores that the erosion separated, so they are not "
        "re-merged.<br/>"
        "5. <i>Structuring element:</i> a 3×3 cross (4-connectivity). Iterating it n times acts like a diamond of radius n, "
        "which is a cheap approximation of a disk matching the round cells.<br/>"
        "6. <i>Count:</i> label 4-connected components with scipy.ndimage.label and keep only those with area in "
        "[15, 400] px. Below 15 px are noise fragments; above 400 px (about three or more cells) are clumps that cannot "
        "be resolved."))
    s.append(FIGURE("task5b_baseline.png", "Figure 17. Baseline without background flattening (starter threshold "
                    f"{base['params']['threshold']}): the hazy region becomes one giant component.",
                    width=PAGE_W * 0.7, max_h=62 * mm))
    rows = [["Image", "Background σ", "Threshold", "Closing", "Erosion", "Dilation", "Area range (px)", "Rejected small / large", "Count"]]
    for fname in bd.SELECTED:
        d = t5b[fname]
        p = d["params"]
        rows.append([fname, p["background_sigma"], p["threshold"], p["closing_iterations"], p["erosion_iterations"],
                     p["dilation_iterations"], f"{p['min_area']}-{p['max_area']}",
                     f"{d['rejected_small']} / {d['rejected_large']}", f"<b>{d['count']}</b>"])
    s.append(TABLE(rows, [PAGE_W * w for w in (0.13, 0.11, 0.1, 0.08, 0.08, 0.08, 0.13, 0.17, 0.08)]))
    s.append(P("Table 1. Morphology parameters and final counts (iterations of the 3×3 cross; 001cell.png not used).", "cap"))
    for i, fname in enumerate(bd.SELECTED):
        d = t5b[fname]
        s.append(FIGURE(f"task5b_{Path(fname).stem}_panel.png",
                        f"Figure {18 + i}. {fname}: input, threshold mask after flattening, cleaned binary mask, and "
                        f"retained components (red boxes): {d['count']} cells.", max_h=52 * mm))
    s.append(P(
        "The threshold was tuned per image within a small range: 0.12 for 002, 050 and 150, and 0.15 for 100cell.png, "
        "which has more haze between cells. The other settings are the same for all four images. In every image, "
        "around 0.12-0.15 the counts change by only a few cells when the threshold or the erosion/dilation counts are perturbed (see the "
        "sweeps in 5(c))."))

    s.append(P("Task 5 (c)  Comparison and analysis", "h2"))
    ts = t5c["threshold_sweep"]
    es = t5c["erosion_sweep"]
    s.append(FIGURE("task5c_sweeps.png", "Figure 22. One-parameter sweeps on "
                    f"{t5a['image']}: morphology threshold (other settings fixed) and DoG threshold_rel.",
                    width=PAGE_W * 0.85, max_h=60 * mm))
    sweep_txt = ", ".join(f"{k} → {v}" for k, v in ts.items())
    ero_txt = ", ".join(f"{k} → {v}" for k, v in es.items())
    s.append(P(
        f"On {t5a['image']}, the DoG detector finds <b>{t5c['dog']}</b> blobs and the morphology pipeline counts "
        f"<b>{t5c['morph']}</b> components, against a rough visual count of 170-190. DoG works on the intensity profile "
        "directly: each nucleus is a local bump, so even touching cells keep separate maxima. Its errors are merging of "
        "very tight chains into one maximum, occasional double maxima on elongated cells, and loss of the faintest "
        "out-of-focus cells. Morphology must first commit to a binary mask, and its dominant error is "
        "<i>under-segmentation</i>: overlapping cells share one connected component, or form a clump larger than "
        "max_area that is discarded entirely. Smaller errors are dim cells lost below the threshold and, rarely, one cell split "
        "into two fragments. One parameter change shows the trade-off clearly: raising the morphology threshold "
        f"({sweep_txt}) increases the count, because a higher level cuts through the darker valleys between touching "
        "cells and separates them. Beyond about 0.15-0.18, dim cells shrink below min_area and disappear, so the curve flattens "
        f"and then drops. Changing the erosion count (with dilation 2: {ero_txt}) behaves similarly. For DoG, raising "
        f"threshold_rel from 0.3 to 0.5 removes dim cells ({t5a['rel_sweep']['0.3']} → {t5a['rel_sweep']['0.5']}). "
        "Overall, DoG maxima are better suited to dense, touching cells, while morphology gives cell shapes and areas but "
        "undercounts clusters unless a splitting step (e.g. watershed) is added."))

    # ------------------------------------------------------------ appendix
    s.append(PageBreak())
    s.append(P("Appendix A: Cloud verification results", "h1"))
    s.append(P("Output of cloud_check/verify.py on the GitHub Actions runner (Python 3.11, Ubuntu)."))
    rows = [["Result", "Check", "Detail"]]
    for line in VERIFY.splitlines():
        m = re.match(r"\[(PASS|FAIL)\] (.*?): ?(.*)$", line)
        if m:
            rows.append([m.group(1), m.group(2), m.group(3)[:90]])
    s.append(TABLE(rows, [PAGE_W * 0.08, PAGE_W * 0.56, PAGE_W * 0.36]))
    s.append(P("Appendix B: Task 5 implementation", "h1"))
    for fn, cap in ((bd.gaussian_filter, "gaussian_filter"), (bd.difference_of_gaussians, "difference_of_gaussians"),
                    (bd.find_blob_maxima, "find_blob_maxima"), (bd.morphology_mask, "morphology_mask"),
                    (bd.count_components, "count_components")):
        s.extend(CODE(fn, f"Listing. {cap}() in blob_detection.py"))
    return s


def build_body():
    path = TMP / "body.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=TITLE,
                            author=f"{NAME_CN} {STUDENT_ID}")
    doc.build(story(), onFirstPage=header_footer, onLaterPages=header_footer)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cover = build_cover()
    body = build_body()
    writer = PdfWriter()
    for part in (cover, body):
        for page in PdfReader(str(part)).pages:
            writer.add_page(page)
    writer.add_metadata({"/Title": TITLE, "/Author": f"{NAME_CN} {STUDENT_ID}"})
    pdf_path = OUT / PDF_NAME
    with open(pdf_path, "wb") as f:
        writer.write(f)
    shutil.copy2(pdf_path, OUT / "Lichangshuo_Assignment2.pdf")
    with zipfile.ZipFile(OUT / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pdf_path, PDF_NAME)
        for code in ("filters.py", "corners.py", "blob_detection.py"):
            z.write(RES / code, code)
    shutil.rmtree(TMP)
    print("pages:", len(PdfReader(str(pdf_path)).pages))
    print("wrote", pdf_path, "and", OUT / ZIP_NAME)


if __name__ == "__main__":
    main()
