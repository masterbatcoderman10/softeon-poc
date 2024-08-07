import pymupdf
import os
import shutil
import pathlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(file_path, dst_path):
    try:
        os.makedirs(dst_path, exist_ok=True)
        if file_path.endswith(".DS_Store"):
            return
        pdf_file = pymupdf.open(file_path)
        # get the file name without the extension
        output_path = os.path.join(
            dst_path, pathlib.Path(file_path).stem + ".txt")
        with open(output_path, "wb") as text_file:
            for page in pdf_file:
                text_file.write(page.get_text().encode("utf-8"))
                # write page delimiter (form feed 0x0C)
                text_file.write(bytes((12,)))
    except Exception as e:
        print(e)
        return None


def extract_images(file_path, dst_path):
    try:
        os.makedirs(dst_path, exist_ok=True)
        if file_path.endswith(".DS_Store"):
            return
        doc = pymupdf.open(file_path)  # open a document
        # get the file name without the extension
        file_stem = pathlib.Path(file_path).stem

        for page_index in range(len(doc)):  # iterate over pdf pages
            page = doc[page_index]  # get the page
            image_list = page.get_images()

            # enumerate the image list
            for image_index, img in enumerate(image_list, start=1):
                xref = img[0]  # get the XREF of the image
                pix = pymupdf.Pixmap(doc, xref)  # create a Pixmap

                if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                width, height = pix.width, pix.height
                product = width * height

                if product >= 100000:
                    # save the image as png with the original file name and image ids
                    image_filename = f"{file_stem}_page_{page_index + 1}_image_{image_index}.png"
                    image_path = os.path.join(dst_path, image_filename)
                    pix.save(image_path)
                    pix = None

    except Exception as e:
        print(e)
        return None


def process_files(dir_path, dst_path):

    # check if the directory exists
    if not os.path.exists(dir_path):
        print("Directory does not exist")
        return

    # for each file in the directory extract text
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        extract_text(file_path, os.path.join(dst_path, 'text'))
        extract_images(file_path, os.path.join(dst_path, 'images'))

        # at the end move the file to the processed directory
        shutil.move(file_path, os.path.join('data/processed_files', file))


if __name__ == "__main__":
    process_files("data/new_files", "data/processed_data")
