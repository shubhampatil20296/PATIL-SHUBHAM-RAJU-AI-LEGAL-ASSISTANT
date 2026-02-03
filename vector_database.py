from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple text splitter to avoid heavy dependencies
class SimpleCharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200, add_start_index=True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

    def split_documents(self, documents):
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for i, chunk_text in enumerate(text_chunks):
                chunk = type(doc)(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                if self.add_start_index:
                    chunk.metadata['start_index'] = i * (self.chunk_size - self.chunk_overlap)
                chunks.append(chunk)
        return chunks

    def split_text(self, text):
        """Split text into chunks of specified size with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for natural break points
                break_point = end
                for sep in ["\n\n", "\n", ". ", " ", ""]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep != -1 and last_sep > start + self.chunk_size // 2:
                        break_point = last_sep + len(sep)
                        break
                end = break_point

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#Step 1: Upload & Load raw PDF(s)
pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    logger.info(f"Loading PDF from: {file_path}")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    logger.info(f"✓ PDF loaded successfully with {len(documents)} pages")
    return documents


# #Step 2: Create Chunks
def create_chunks(documents): 
    text_splitter = SimpleCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


# #Step 3: Setup Embeddings Model (Use HuggingFace Embeddings)
def get_embedding_model():
    logger.info("Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("✓ Embeddings model ready")
    return embeddings


# #Step 4: Index Documents **Store embeddings in FAISS (vector store)
FAISS_DB_PATH="vectorstore/db_faiss"

# Load existing FAISS database if available
try:
    faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
    logger.info("✓ Loaded existing FAISS vector store")
except Exception as e:
    logger.warning(f"Could not load existing FAISS database: {e}")
    faiss_db = None


if __name__ == "__main__":
    logger.info("Initializing vector database setup...")

    file_path = 'pdfs/universal_declaration_of_human_rights.pdf'
    logger.info(f"Starting PDF processing for: {file_path}")
    documents = load_pdf(file_path)
    logger.info(f"Total PDF pages: {len(documents)}")

    logger.info("Creating text chunks from documents...")
    text_chunks = create_chunks(documents)
    logger.info(f"✓ Created {len(text_chunks)} text chunks")

    logger.info("Setting up HuggingFace embeddings model...")
    logger.info("Building embeddings for all chunks (this may take a moment)...")
    faiss_db=FAISS.from_documents(text_chunks, get_embedding_model())
    logger.info(f"✓ FAISS vector store created successfully")

    logger.info(f"Saving FAISS database to: {FAISS_DB_PATH}")
    faiss_db.save_local(FAISS_DB_PATH)
    logger.info("✓ Vector database setup complete!")
