import openai
import streamlit as st
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
import os
from PyPDF2 import PdfFileWriter, PdfFileReader


# Constants
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'
openai.api_key = os.getenv("OPENAI_API_KEY","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")

# Retry decorator to handle exceptions
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]


# Function to truncate text
def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


# Function to split a long text into smaller chunks and return the embeddings of the chunks
def get_chunk_embeddings(long_text):
    # Split the long text into smaller chunks
    chunks = [long_text[i:i+EMBEDDING_CTX_LENGTH] for i in range(0, len(long_text), EMBEDDING_CTX_LENGTH)]
    embeddings = []
    # Get the embeddings of each chunk
    for chunk in chunks:
        truncated_chunk = truncate_text_tokens(chunk)
        embeddings.append(get_embedding(truncated_chunk))
    return embeddings


# Streamlit app
def app():
    # Title and description
    st.title("PDF Truncation Tool")
    st.write("This app can be used to truncate a PDF file.")

    # Upload file
    file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if file is uploaded
    if file is not None:
        # Read the PDF file
        pdf_reader = PdfFileReader(file)
        # Get the total number of pages in the PDF file
        num_pages = pdf_reader.getNumPages()

        # Truncate each page of the PDF file and save it to a new PDF file
        pdf_writer = PdfFileWriter()
        for page in range(num_pages):
            # Get the text of the page
            page_text = pdf_reader.getPage(page).extractText()
            # Get the embeddings of the chunks of the page
            embeddings = get_chunk_embeddings(page_text)
            # Add the embeddings to a new PDF file
            pdf_writer.addPage(pdf_reader.getPage(page))

        # Save the new PDF file
        new_file = f"{file.name[:-4]}_truncated.pdf"
        with open(new_file, "wb") as f:
            pdf_writer.write(f)

        # Show the link to the new PDF file
        st.success(f"The truncated PDF file is available [here](/{new_file}).")
