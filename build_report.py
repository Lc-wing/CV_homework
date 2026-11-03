"""Build the Assignment 3 report from the completed code and saved figures."""
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

import task1
import task23

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "submission"
TMP = OUT / "tmp"
NAME = "李昌硕"
SID = "1240000787"
TITLE = "Assignment 3: Stereo Vision"

pdfmetrics.registerFont(TTFont("Serif", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Bold", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Italic", "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("CJK", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"))

NAVY = HexColor("#102a43")
ST = {
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=20, textColor=NAVY,
                         spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12, leading=16, textColor=HexColor("#243b53"),
                         spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10, leading=14.5, spaceAfter=6),
    "cap": ParagraphStyle("cap", fontName="Serif-Italic", fontSize=8.6, leading=11.5,
                          textColor=HexColor("#334e68"), alignment=1, spaceBefore=2, spaceAfter=8),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.2, leading=9.2, leftIndent=4,
                           backColor=HexColor("#f4f6f8"), borderPadding=4, spaceBefore=2, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=8.4, leading=11),
    "mat": ParagraphStyle("mat", fontName="Mono", fontSize=9, leading=12, leftIndent=18, spaceAfter=6),
}
PAGE_W = A4[0] - 32 * mm


def P(text, style="body"):
    return Paragraph(text, ST[style])


def mono(name):
    return f"<font face='Mono' size='9'>{name}</font>"


def figure(path, caption, max_h=115 * mm):
    with PILImage.open(path) as im:
        w, h = im.size
        rgb = PILImage.new("RGB", im.size, (255, 255, 255))
        rgba = im.convert("RGBA")
        rgb.paste(rgba, mask=rgba.split()[-1])
        jpg = TMP / (Path(path).stem + ".jpg")
        rgb.save(jpg, quality=85, optimize=True)
    scale = min(PAGE_W / w, max_h / h)
    img = RLImage(str(jpg), width=w * scale, height=h * scale)
    return KeepTogether([img, P(caption, "cap")])


def code_block(text):
    return Preformatted(text.rstrip(), ST["code"])


def table(rows, widths):
    data = [[Paragraph(str(c), ST["cell"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#d9e2ec")),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#9fb3c8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def sym_error(F, p1, p2):
    P1 = np.c_[p1, np.ones(len(p1))]
    P2 = np.c_[p2, np.ones(len(p2))]
    residual = np.einsum("ij,jk,ik->i", P2, F, P1)
    line2 = (F @ P1.T).T
    line1 = (F.T @ P2.T).T
    d2 = np.abs(residual) / np.sqrt(line2[:, 0] ** 2 + line2[:, 1] ** 2)
    d1 = np.abs(residual) / np.sqrt(line1[:, 0] ** 2 + line1[:, 1] ** 2)
    return float(0.5 * (d1.mean() + d2.mean()))


def fmt_vec(v):
    return "[" + ", ".join(f"{x:.6g}" for x in v) + "]"


def cartesian(v):
    if abs(v[2]) < 1e-12:
        return "at infinity"
    xy = v[:2] / v[2]
    return f"({xy[0]:.2f}, {xy[1]:.2f})"


def build_cover():
    doc = Document(str(ROOT / "cover_template.docx"))
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("Name:"):
            run = para.add_run(" " + NAME)
        elif text.startswith("Student ID:"):
            run = para.add_run(" " + SID)
        else:
            continue
        run.font.name = "WenQuanYi Micro Hei"
        run.font.size = Pt(14)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "WenQuanYi Micro Hei")
    filled = TMP / "cover.docx"
    doc.save(str(filled))
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(TMP), str(filled)],
        check=True, timeout=180,
    )
    return TMP / "cover.pdf"


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 14, w, 14, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Serif", 7.5)
    canvas.drawString(16 * mm, h - 10, "CS460 " + TITLE)
    canvas.setFont("CJK", 8)
    canvas.drawRightString(w - 16 * mm, h - 10, f"{NAME}  {SID}")
    canvas.setFillColor(HexColor("#627d98"))
    canvas.setFont("Serif", 8)
    canvas.drawCentredString(w / 2, 10 * mm, "Page %d" % (doc.page + 1))
    canvas.restoreState()


def build_body():
    pts2d = np.loadtxt(ROOT / "task1" / "pts2d.txt")
    pts3d = np.loadtxt(ROOT / "task1" / "pts3d.txt")
    M = task1.find_projection(pts2d, pts3d)
    distance = task1.compute_distance(pts2d, pts3d)
    matrix = "\n".join(
        "  [" + "  ".join(f"{v: .8f}" for v in row) + "]" for row in M
    )

    rows = [["Dataset", "Mean symmetric epipolar distance (px)", "Matches OpenCV 8-point"]]
    epipoles = {}
    import cv2
    for name in ["temple", "reallyInwards", "xtrans"]:
        img = cv2.imread(str(ROOT / "task23" / name / "im1.png"), cv2.IMREAD_GRAYSCALE)
        data = np.load(ROOT / "task23" / name / "data.npz")
        p1 = data["pts1"].astype(float)
        p2 = data["pts2"].astype(float)
        F = task23.find_fundamental_matrix(img.shape, p1, p2)
        Fcv, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_8POINT)
        e1, e2 = task23.compute_epipoles(F)
        epipoles[name] = (e1, e2, sym_error(F, p1, p2))
        ours = F / np.linalg.norm(F)
        ref = Fcv / np.linalg.norm(Fcv)
        if np.sum(ours * ref) < 0:
            ours = -ours
        rows.append([name, f"{epipoles[name][2]:.4g}", f"{np.linalg.norm(ours - ref):.2e}"])

    e1_ri, e2_ri, _ = epipoles["reallyInwards"]
    e1_x, e2_x, _ = epipoles["xtrans"]

    story = []
    story.append(P("Task 1: Estimating the projection matrix M", "h1"))
    story.append(P("Task 1 (a)  find_projection", "h2"))
    story.append(P(
        "Each correspondence satisfies p ≡ MX, with p = [u, v, 1]<super>T</super> and "
        "X = [x, y, z, 1]<super>T</super>. The cross product p × MX = 0 removes the unknown scale. "
        "Only two of the three resulting equations are independent. Writing m for the 12 entries of M "
        "stacked row by row, correspondence i contributes the two rows"))
    story.append(Preformatted(
        "[  0          X_i    -v_i X_i ]\n"
        "[ X_i         0      -u_i X_i ]",
        ST["mat"]))
    story.append(P(
        "Stacking every correspondence produces a matrix A with 2N rows. The fit is the homogeneous "
        "least-squares problem Am = 0. The solution is the right singular vector of A belonging to the "
        "smallest singular value, reshaped into a 3×4 matrix. That vector is returned at unit length, so "
        "the reported M has Frobenius norm 1. Any other scale represents the same camera."))
    story.append(code_block(
        "X = [pts3d | 1]\n"
        "A[0::2] = [0,  X, -v X]\n"
        "A[1::2] = [X,  0, -u X]\n"
        "M = svd(A).Vt[-1].reshape(3, 4)"))
    story.append(P("Task 1 (b)  M for task1/pts2d.txt and pts3d.txt", "h2"))
    story.append(P(
        f"There are {len(pts2d)} correspondences. The unit-norm projection matrix is"))
    story.append(Preformatted(matrix, ST["mat"]))
    story.append(P(
        "The scale is arbitrary: if M is a solution then so is λM, because the homogeneous image "
        "coordinate is divided by its third entry. The matrix above is the one with ||M||<sub>F</sub> = 1 "
        "returned by the SVD."))
    story.append(P("Task 1 (c)  Average reprojection distance", "h2"))
    story.append(P(
        f"{mono('compute_distance')} projects every 3D point with this M, converts the result from "
        "homogeneous to pixel coordinates by proj([x, y, w]) = [x/w, y/w], and returns the mean Euclidean "
        f"distance to the given 2D points. On the task1 sample that distance is <b>{distance:.6f} pixels</b>, "
        "below the 0.01 threshold in the assignment. The points are therefore consistent with a single linear camera."))
    story.append(P("Task 1 (d)  Equation 2 and Equation 6", "h2"))
    story.append(P(
        "The two expressions are related, but they are not the same objective."))
    story.append(P(
        "Equation 6 is the matrix form of p × MX = 0. It is linear in the entries of M, which is why it can "
        "be solved by homogeneous least squares. Its residual on one correspondence is made of terms "
        "M<sub>1</sub>·X − u M<sub>3</sub>·X and M<sub>2</sub>·X − v M<sub>3</sub>·X. If w = M<sub>3</sub>·X "
        "is the homogeneous scale of the projected point, those terms equal w times the pixel error in x and in y. "
        "The quantity that the SVD actually minimizes is the sum of squares of these algebraic residuals."))
    story.append(P(
        "Equation 2 is the mean Euclidean distance, in pixels, between the observed point and the "
        "dehomogenized projection. It is the geometric reprojection error. It equals the algebraic residual "
        "only after dividing by w. Because w depends on depth and on the arbitrary scale of M, the algebraic "
        "objective weights points differently from the geometric one."))
    story.append(P(
        "On this sample there is no measurement noise, so both residuals can be driven essentially to zero "
        f"(the geometric mean is {distance:.6f} px). The two objectives then pick out the same camera. "
        "With noisy measurements they do not. Equation 2 is the one that should be minimized: it is an actual "
        "distance in the image, in pixel units, and it does not depend on how M is scaled. Equation 6 does not "
        "measure a geometric distance. It is still the right tool for a first estimate, because it is linear and "
        "has a closed-form solution. A noisy problem would use that estimate to initialize a nonlinear "
        "minimization of Equation 2."))

    story.append(P("Task 2: Estimating the fundamental matrix", "h1"))
    story.append(P("Task 2 (a)  Eight-point algorithm", "h2"))
    story.append(P(
        f"{mono('find_fundamental_matrix')} follows the normalized eight-point algorithm."))
    story.append(P(
        "1. Each image gets its own similarity T. The point set is translated so that its centroid is at the "
        "origin, then scaled so that the mean distance from the origin is √2. This is the normalization from "
        "Hartley's eight-point algorithm. It is the stable form of the instruction to center the data at 0 and "
        "to scale coordinates that otherwise range over hundreds of pixels. The image diagonal, "
        "√(h<sup>2</sup>+w<sup>2</sup>) from the given image shape, is the fallback scale if the points are degenerate. "
        "Raw pixel products u′u would otherwise dominate the constant term 1 and make U<super>T</super>U ill-conditioned."))
    story.append(P(
        "2. For the normalized points, each correspondence supplies one row "
        "[u′u, u′v, u′, v′u, v′v, v′, u, v, 1], the Kronecker product of [u′, v′, 1] and [u, v, 1]. "
        "The vectorized F is the smallest right singular vector of this design matrix."))
    story.append(P(
        "3. A fundamental matrix must have rank 2, otherwise the epipoles do not exist and the epipolar lines "
        "do not meet. The SVD of the fitted matrix is recomputed, the smallest singular value is set to 0, and "
        "the matrix is reconstructed. That is the closest rank-2 matrix in the Frobenius norm."))
    story.append(P(
        "4. The normalization is undone by F = T<sub>2</sub><super>T</super> F<sub>rank2</sub> T<sub>1</sub>. "
        "Substituting the normalized points p̄ = T p into p̄′<super>T</super> F<sub>rank2</sub> p̄ = 0 gives exactly "
        "p′<super>T</super> (T<sub>2</sub><super>T</super> F<sub>rank2</sub> T<sub>1</sub>) p = 0."))
    story.append(P(
        "F is defined only up to scale. After dividing by the Frobenius norm and fixing the sign, the result "
        "agrees with cv2.findFundamentalMat(..., FM_8POINT), which uses the same normalization:"))
    story.append(table(rows, [PAGE_W * 0.28, PAGE_W * 0.42, PAGE_W * 0.30]))
    story.append(Spacer(1, 4))
    story.append(P(
        "The residual of about 0.36 px on temple and reallyInwards is the same residual OpenCV reports. "
        "The correspondences are stored as integers, so a perfect real-valued F cannot put every point exactly "
        "on its line. On xtrans the geometry is a pure horizontal translation and the residual is about 10<sup>−13</sup> px."))
    story.append(P("Task 2 (b)  Epipoles", "h2"))
    story.append(P(
        "The epipole in image 1 lies on every epipolar line of image 2, so F e<sub>1</sub> = 0. "
        "The epipole in image 2 satisfies e<sub>2</sub><super>T</super> F = 0. "
        f"{mono('compute_epipoles')} reads these from the SVD F = UΣV<sup>T</sup>: "
        "e<sub>1</sub> is the last column of V, and e<sub>2</sub> is the last column of U. "
        "Both are returned as homogeneous coordinates, without forcing the third entry to 1. "
        "A third entry of 0 is a real answer: the epipole is then a point at infinity, and dividing by it is not defined. "
        "For every dataset, ||F e<sub>1</sub>|| and ||e<sub>2</sub><super>T</super> F|| are on the order of 10<sup>−16</sup> to 10<sup>−19</sup>."))
    story.append(P("Task 2 (c)  Epipolar lines", "h2"))
    story.append(P(
        "The figures below are drawn by the supplied draw_epipolar routine, using every tenth correspondence "
        "so that the lines stay readable. A line and its matching points share a color. The epipole is drawn "
        "when it falls inside the image."))
    story.append(figure(
        RES / "temple_us.png",
        "Figure 1. Temple. The epipoles are far outside the frame, at about (15206, 279) and (−9279, −2), "
        "so inside this narrow image the epipolar lines are nearly parallel. Each line passes through the point of the same color."))
    story.append(figure(
        RES / "reallyInwards_us.png",
        "Figure 2. reallyInwards. The cameras are toed in, so both epipoles fall inside the images "
        "(the black-and-white dots). Every epipolar line passes through the epipole and through the matching point."))
    story.append(figure(
        RES / "xtrans_us.png",
        "Figure 3. xtrans, the extra dataset. The motion is a horizontal translation, so the epipolar lines are "
        "horizontal and the epipoles are at infinity. No epipole is drawn, and points of one color stay on a common horizontal line."))
    story.append(P("Task 2 (d)  Epipoles for reallyInwards and xtrans", "h2"))
    epi_rows = [["Dataset", "Epipole", "Homogeneous (unit vector)", "Image coordinates"]]
    epi_rows.append(["reallyInwards", "e1, image 1", fmt_vec(e1_ri), cartesian(e1_ri)])
    epi_rows.append(["reallyInwards", "e2, image 2", fmt_vec(e2_ri), cartesian(e2_ri)])
    epi_rows.append(["xtrans", "e1, image 1", fmt_vec(e1_x), cartesian(e1_x)])
    epi_rows.append(["xtrans", "e2, image 2", fmt_vec(e2_x), cartesian(e2_x)])
    story.append(table(epi_rows, [PAGE_W * 0.18, PAGE_W * 0.16, PAGE_W * 0.42, PAGE_W * 0.24]))
    story.append(Spacer(1, 6))
    story.append(P(
        "For reallyInwards the image is 1280×960 and the principal point is (640, 480). "
        f"e<sub>1</sub> dehomogenizes to {cartesian(e1_ri)} and e<sub>2</sub> to {cartesian(e2_ri)}. "
        "Both are inside the image, at nearly the same height as the principal point, one on the right and one on the left. "
        "That is the pattern in Figure 2."))
    story.append(P(
        "For xtrans, e<sub>1</sub> is [1, 0, 0] up to 10<sup>−15</sup>, and the third entry of e<sub>2</sub> is exactly 0, "
        "so e<sub>2</sub> = [−1, 0, 0]. Both are points at infinity in the horizontal direction. "
        "Dividing by the third entry is undefined for e<sub>2</sub> and gives a meaningless coordinate near 10<sup>17</sup> "
        "for e<sub>1</sub>. The homogeneous vectors are the right quantities to report. They are why the lines in Figure 3 "
        "are parallel and horizontal: a pure sideways translation sends the baseline parallel to the image plane."))
    story.append(P(
        f"The same null-vector procedure gives the temple epipoles used in the caption of Figure 1, "
        "and it places the ztrans epipoles on the principal point, as a forward translation should."))
    return story


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    cover = build_cover()
    body = TMP / "body.pdf"
    doc = SimpleDocTemplate(
        str(body), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm, title=TITLE, author=f"{NAME} {SID}",
    )
    doc.build(build_body(), onFirstPage=header_footer, onLaterPages=header_footer)
    writer = PdfWriter()
    for part in (cover, body):
        for page in PdfReader(str(part)).pages:
            writer.add_page(page)
    writer.add_metadata({"/Title": TITLE, "/Author": f"{NAME} {SID}"})
    pdf_path = OUT / f"{NAME}_Assignment3.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    zip_path = OUT / f"{NAME}{SID}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pdf_path, pdf_path.name)
        for code in ("task1.py", "task23.py", "utils.py"):
            z.write(ROOT / code, code)
    print("pages", len(PdfReader(str(pdf_path)).pages))
    print(pdf_path)
    print(zip_path)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
