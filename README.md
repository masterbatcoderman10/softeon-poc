# Softeon Support Bot
This project contains the code for the Softeon Support Bot. The bot uses GenAI to answer queries about the WMS.

## File Structure

The project is distributed in the following directories:
- `app`: Contains the main application code. This is the chainlit application
- `data`: Contains the data used by the application. Includes the local chroma database as well as the text and image data.
- `etl`: Contains the code for the ETL process. This includes the code for processing the PDF files and loading the data into the database.
- `utils`: Contains utility functions used by the application.

## Installation

Once the repository is cloned, go ahead and create a virtual environment and install the dependencies.

This can be done in 2 ways:

1. Running the following commands in the terminal/command prompt (Both Windows and Mac):

```bash
$ python3 -m venv venv
$ source venv/bin/activate # On Mac
$ venv\Scripts\activate # On Windows
$ pip install -r requirements.txt # On Mac/Linux
$ pip install -r requirements_w.txt # On Windows
```


## Running the Application

Prior to running the application, it will be important to setup the vector database. This can be done by running the following command in the terminal:

```bash
python3 -m etl.loading
```

To run the application after setup, run the following command in the terminal:

```bash
python3 -m chainlit run app/app.py
```