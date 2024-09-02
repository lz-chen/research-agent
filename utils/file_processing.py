from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import io


def pdf_to_images(pdf_path: Path, output_folder: Path, dpi: int = 200) -> List[str]:
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


if __name__ == '__main__':
    for pdf_name in Path("data/papers").glob("*.pdf"):
        paper_img_dir = Path("data/papers_image")
        # paper_img_dir.mkdir(exist_ok=True, parents=True)
        output_dir = paper_img_dir / pdf_name.stem
        output_dir.mkdir(exist_ok=True, parents=True)
        image_paths = pdf_to_images(pdf_name, output_dir)
        print(image_paths)
