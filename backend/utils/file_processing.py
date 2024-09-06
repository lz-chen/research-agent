from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
import io
import subprocess


def pdf2images(pdf_path: Path, output_folder: Path, dpi: int = 200) -> List[str]:
    """
    Convert each page of a PDF to an image and save them to the output folder.

    :param pdf_path: Path to the PDF file.
    :param output_folder: Folder to save the output images.
    :param dpi: Desired DPI for the output images.
    :return: List of paths to the saved images.
    """
    doc = fitz.open(pdf_path)
    image_paths = []
    zoom = dpi / 72  # 72 is the default DPI for PDF

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        img = Image.open(io.BytesIO(img_data))
        img_path = f"{output_folder}/page_{page_num + 1}.png"
        img.save(img_path)
        image_paths.append(img_path)

    return image_paths


def pptx2pdf(pptx_path: Path, pdf_path: Path):
    # Call LibreOffice in headless mode to convert pptx to pdf
    # subprocess.run(
    #     [
    #         "unoconv",
    #         "-f",
    #         "pdf",
    #         "-e",
    #         "ExportHiddenSlides=true",
    #         "-o",
    #         pdf_path.as_posix(),
    #         pptx_path.as_posix(),
    #     ], check=True
    # )
    subprocess.run(
        [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            pdf_path.parent.as_posix(),
            pptx_path.as_posix(),
        ], check=True
    )
    # return pdf_path


def pptx2images(pptx_path: Path, output_folder: Optional[Path] = None, dpi: int = 200) -> str:
    """
    Convert each slide of a PPTX to an image and save them to the output folder.
    :param pptx_path: file path to the PPTX file
    :param output_folder: folder to save the output images
    :param dpi: desired DPI for the output images, default is 200
    :return: output folder path for the images
    """
    pdf_path = pptx_path.parent.joinpath(f"{pptx_path.stem}.pdf")
    pptx2pdf(pptx_path, pdf_path)
    img_output_folder = output_folder or pptx_path.parent.joinpath(f"{pptx_path.stem}_images")
    img_output_folder.mkdir(exist_ok=True, parents=True)
    pdf2images(Path(pdf_path), img_output_folder, dpi)
    return img_output_folder.as_posix()
    # # Convert PDF to images
    # subprocess.run(
    #     [
    #         "convert",
    #         "-density",
    #         str(dpi),
    #         pdf_path,
    #         f"{output_folder}/slide_%03d.png",
    #     ], check=True
    # )


if __name__ == '__main__':
    # for pdf_name in Path("data/papers").glob("*.pdf"):
    #     paper_img_dir = Path("data/papers_image")
    #     # paper_img_dir.mkdir(exist_ok=True, parents=True)
    #     output_dir = paper_img_dir / pdf_name.stem
    #     output_dir.mkdir(exist_ok=True, parents=True)
    #     image_paths = pdf2images(pdf_name, output_dir)
    #     print(image_paths)

    # pptx2pdf(Path("workflow_artifacts/generated_presentation.pptx"),
    #          Path("workflow_artifacts/generated_presentation.pdf"))

    pptx2images(Path("workflow_artifacts/generated_presentation.pptx"), None)
