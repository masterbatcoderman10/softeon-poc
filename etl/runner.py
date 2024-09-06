from .extraction import process_files
from .summarization import summarize_images
from .loading import load_retriever, tag_documents, create_documents_and_load
from utils.setup_retriever import retriever
import os


def run_etl_pipeline():

    new_files_dir = "data/new_files"
    processed_data_dir = "data/temporary_data"
    images_dir = f"{processed_data_dir}/images"
    image_summaries_dir = f"{processed_data_dir}/image_summaries"
    # for self-query retriever
    metadata_dir = f"{processed_data_dir}/file_metadata"

    os.makedirs(new_files_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(image_summaries_dir, exist_ok=True)

    # # process the files
    process_files(new_files_dir, processed_data_dir)

    # summarize the images
    summarize_images(images_dir, image_summaries_dir)

    # define paths
    text_dir = f"{processed_data_dir}/text"

    # load the retriever
    # load_retriever(retriever, text_dir, images_dir, image_summaries_dir)

    # alternate
    tag_documents(text_dir, metadata_dir)
    create_documents_and_load(text_dir, f"{metadata_dir}/metadata_v1.json")

    print("ETL pipeline completed successfully")


if __name__ == "__main__":
    run_etl_pipeline()
