from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import (
    FAISS,
    lancedb,
    Chroma,
)



from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    AzureAIDocumentIntelligenceLoader,
    UnstructuredODTLoader,
    WikipediaLoader,  
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

import pandas as pd
from tqdm.notebook import tqdm
from typing import Optional, List, Tuple

DATA_PATH = "Classical_ML/"
DB_FAISS_PATH = "vectorstore\db_faiss_ML"

def preprocess_document(document):
    text = []
    if "page_content" in document:
        text = document["page_content"]
    else:
        # Handle documents without the key (e.g., log a warning or skip processing)
        pass

    text = text.lower()

    punctuation = string.punctuation
    text = "".join([char for char in text if char not in punctuation])

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    preprocessed_document = {"preprocessed_text": " ".join(tokens)}
    return preprocessed_document


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    print(knowledge_base[0].page_content[:400])

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def create_vector_db():
    docs_processed = []
    loaders = [
        DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=UnstructuredMarkdownLoader),
        DirectoryLoader(DATA_PATH, glob="*.json", loader_cls=JSONLoader),
        DirectoryLoader(DATA_PATH, glob="*.html", loader_cls=UnstructuredHTMLLoader),
        DirectoryLoader(DATA_PATH, glob="*.csv", loader_cls=CSVLoader),
        DirectoryLoader(DATA_PATH, glob="*.odt", loader_cls=UnstructuredODTLoader),
    ]

    # Specify your Azure AI Document Intelligence parameters
    # endpoint = "<your_endpoint>"
    # key = "<your_key>"
    # file_path = "<path_to_your_document>"
    # loader = AzureAIDocumentIntelligenceLoader(
    #   api_endpoint=endpoint,
    # api_key=key,
    # file_path=file_path,
    # api_model="prebuilt-layout"
    # )

    all_documents = []
    for loader in loaders:
        documents = loader.load()
        all_documents += documents
        # for doc in documents:
        #     preprocessed_doc = preprocess_document(doc.copy())
        #     all_documents.append(preprocessed_doc)

    docs = WikipediaLoader(query="Material Science", load_max_docs=10).load()
    all_documents += docs
    # df = pd.DataFrame(all_documents)
    # df.to_csv("all_documents.csv", index=False)

    # print(all_documents[0])

    docs_processed += split_documents(
        512,
        all_documents,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = FAISS.from_documents(
        docs_processed, embeddings, distance_strategy=DistanceStrategy.COSINE
    )
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()

