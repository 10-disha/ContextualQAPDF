# ContextualQAPDF

ContextualQAPDF is an intelligent application designed to help you interact with and extract meaningful insights from your PDF documents. Leveraging the power of AWS Bedrock and various AI models, ContextualQAPDF allows users to upload PDFs, ask questions, and receive detailed, contextually accurate answers.

## Features

- **Upload PDF Documents**: Easily upload PDF files to the application.
- **Intelligent Question Answering**: Ask questions related to the content of the PDFs and receive detailed answers.
- **Vector Embedding**: Utilize Titan Embeddings to generate embeddings for document content.
- **Vector Store Management**: Create and manage a FAISS-based vector store for efficient retrieval.
- **Multiple AI Models**: Switch between different AI models for question answering.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/contextualqapdf.git
   cd contextualqapdf
   ```

2. **Set up a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   Create a `.env` file in the project root and add your AWS credentials:

   ```plaintext
   AWS_ACCESS_KEY_ID=your_access_key_id
   AWS_SECRET_ACCESS_KEY=your_secret_access_key
   AWS_SESSION_TOKEN=your_session_token  # Only if using temporary credentials
   ```

## Usage

1. **Run the application**:

   ```bash
   streamlit run app.py
   ```

2. **Upload PDF**: Use the sidebar to upload your PDF documents.

3. **Ask Questions**: Enter your questions in the text input field and get answers based on the content of your PDFs.

## Configuration

The application uses the following AI models:

- **Claude**: AI21 model for general question answering.
- **Llama 2**: Meta model for advanced context handling and detailed answers.

You can switch between models and manage vector stores using the sidebar options.

## Code Overview

- **Data Ingestion**: Load and process PDF documents using `PyPDFDirectoryLoader` and `RecursiveCharacterTextSplitter`.
- **Vector Embedding and Store**: Generate embeddings with `BedrockEmbeddings` and manage them with `FAISS`.
- **Question Answering**: Use `RetrievalQA` to handle user queries and provide detailed answers.

## Troubleshooting

- Ensure your AWS credentials are correctly set in the `.env` file.
- Make sure the FAISS index file exists before attempting to query the vector store.
- Check the console for detailed error messages and tracebacks.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

---

Happy Document Searching with ContextualQAPDF!
