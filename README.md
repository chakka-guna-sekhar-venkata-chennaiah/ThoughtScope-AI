# ThoughtScope AI

ThoughtScope AI is a Python-based application that analyzes PDF documents and annotates the relevant informatoin required for getting the final answer.
![image](https://github.com/user-attachments/assets/90fdbe71-69cb-4919-a963-57c29e7b8676)

## Description

This project uses Unstructured.io for PDF processing, sentence transformers for text embedding, and Google's Gemini model for question answering. It provides a Streamlit interface for easy interaction.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.7 or higher (exact version requirements unclear)
* A Google Cloud API key with access to the Gemini model
* An Unstructured.io API key

To obtain a Google Cloud API key:
1. Go to the [Google Cloud Console](https://aistudio.google.com/prompts/new_chat). Create a folder `.streamlit` and store the api key as `api_key="xxxxx"` in `secrets.toml` file.
2. Create a new project or select an existing one
3. Enable the Gemini API for your project
4. Create credentials (API key) for Gemini

For an Unstructured.io API key, sign up at [unstructured.io](https://unstructured.io/).

## Installation

To install ThoughtScope AI, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ThoughtScope-AI.git
   ```
2. Navigate to the project directory:
   ```
   cd ThoughtScope-AI
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
## Usage

To use ThoughtScope AI:

1. Place your PDF file in the project directory.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open the provided URL in your web browser.
4. Upload your PDF through the Streamlit interface.
5. Ask questions about the PDF content.

## Limitations and Notes

- The exact functionality and user experience may vary. This README is based on code analysis and may not reflect the full implementation details.
- Large PDFs may take significant time to process.
- The accuracy of answers depends on the quality of the PDF and the capabilities of the AI models used.

For more detailed information, please refer to the code comments and docstrings within the project files.
