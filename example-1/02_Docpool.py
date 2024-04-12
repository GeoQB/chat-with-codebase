####
#
# Demonstriert wie Dokumente anhand der "Ähnlichkeit" mit der Anfrage ermittelt werden ...
#

from joblib import dump, load
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import fnmatch
from sentence_transformers import SentenceTransformer, util
import torch
import os

import faiss




class EmbeddingBasedRetriever:
    def __init__(self, files_path='./temp/file_paths.joblib', embeddings_path='./temp/document_embeddings.joblib'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.files_path = files_path
        self.embeddings_path = embeddings_path

        # Load or build the document corpus
        if os.path.exists(files_path):
            self.file_paths = load(files_path)
        else:
            print("File paths not found, please check the path or initialize DocumentRetriever first.")
            self.file_paths = []

        # Check if embeddings exist
        if os.path.exists(self.embeddings_path):
            print("Loading existing document embeddings...")
            self.document_embeddings = load(self.embeddings_path)
        else:
            print("Embeddings not found, generating embeddings...")
            self.documents = [self.extract_text_from_file(file_path) for file_path in self.file_paths]
            self.document_embeddings = self.model.encode(self.documents, convert_to_tensor=True)
            self.save_embeddings(self.document_embeddings)

    def save_embeddings(self, embeddings):
        """Save the document embeddings to a file."""
        dump(embeddings, self.embeddings_path)
        print("Document embeddings saved.")

    def extract_text_from_file(self, file_path):
        # Use your existing functions to extract text based on file extension
        if file_path.endswith('.pdf'):
            return read_pdf(file_path)
        elif file_path.endswith('.docx'):
            return read_docx(file_path)
        else:
            return ""

    def retrieve_documents(self, query, top_k=2):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        # Use self.file_paths to retrieve the paths of the top documents
        return [(self.file_paths[idx], top_results.values[i].item()) for i, idx in enumerate(top_results.indices)]


class DocumentRetriever:
    def __init__(self, index_path='./temp/tfidf_index', files_path='./temp/file_paths.joblib'):
        self.index_path = index_path
        self.files_path = files_path
        self.vectorizer, self.tfidf_matrix, self.file_paths = self.initialize_retriever()

    def initialize_retriever(self):
        if not os.path.exists(f'{self.index_path}_vectorizer.joblib'):
            print(f"Index not found, building index... in {self.index_path}")
            folder_path = './docs'
            corpus, file_paths = build_document_corpus(folder_path)
            tfidf_matrix, vectorizer = create_tfidf_index(corpus)
            save_index(vectorizer, tfidf_matrix, file_paths, index_path=self.index_path, files_path=self.files_path)
        else:
            print(f"Loading existing index... from {self.index_path}")
            vectorizer, tfidf_matrix, file_paths = load_index(index_path=self.index_path, files_path=self.files_path)
        return vectorizer, tfidf_matrix, file_paths

    def retrieve_documents(self, query, top_n=5):
        top_documents = document_retriever(query, self.tfidf_matrix, self.vectorizer, self.file_paths, top_n=top_n)
        return top_documents

def read_pdf(file_path):
    """Extract text from a PDF file."""
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + ' '
    return text

def read_docx(file_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def read_adoc(file_path):
    """Extract text from an AsciiDoc file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def build_document_corpus(folder_path):
    """Build a corpus from PDF, DOCX, AsciiDoc documents, and README files in the given folder."""
    print(f"Processing folder: {folder_path}")

    # Initialize counters
    count_pdf = 0
    count_docx = 0
    count_adoc = 0
    count_readme = 0

    corpus = []
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf') or file.endswith('.docx') or file.endswith('.adoc') or fnmatch.fnmatch(file, 'README.*'):
                file_path = os.path.join(root, file)
                print(f"* {file_path}")

                if file.endswith('.pdf'):
                    text = read_pdf(file_path)
                    count_pdf += 1
                elif file.endswith('.docx'):
                    text = read_docx(file_path)
                    count_docx += 1
                elif file.endswith('.adoc'):
                    text = read_adoc(file_path)
                    count_adoc += 1
                elif fnmatch.fnmatch(file, 'README.*'):  # Assuming README.* files are plaintext
                    text = read_adoc(file_path)  # Reusing read_adoc for plaintext files
                    count_readme += 1

                if len(text) > 0:
                    corpus.append(text)
                    file_paths.append(file_path)

    # Print the counters before returning
    print(f"Total PDF files processed: {count_pdf}")
    print(f"Total DOCX files processed: {count_docx}")
    print(f"Total AsciiDoc files processed: {count_adoc}")
    print(f"Total README files processed: {count_readme}")
    print(f"Total documents processed: {len(corpus)}")
    return corpus, file_paths

def create_tfidf_index(corpus):
    """Create a TF-IDF index from the corpus."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def document_retriever(query, tfidf_matrix, vectorizer, file_paths, top_n=5):
    """Retrieve the top_n most relevant documents for a given query."""
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    relevant_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    return [(file_paths[i], cosine_similarities[i]) for i in relevant_indices]

def save_index(vectorizer, tfidf_matrix, file_paths, index_path='tfidf_index', files_path='file_paths.joblib'):
    """Persist the TF-IDF vectorizer, matrix, and file paths."""
    dump(vectorizer, f'{index_path}_vectorizer.joblib')
    dump(tfidf_matrix, f'{index_path}_matrix.joblib')
    dump(file_paths, files_path)

def load_index(index_path='tfidf_index', files_path='file_paths.joblib'):
    """Load the TF-IDF vectorizer, matrix, and file paths."""
    vectorizer = load(f'{index_path}_vectorizer.joblib')
    tfidf_matrix = load(f'{index_path}_matrix.joblib')
    file_paths = load(files_path)
    return vectorizer, tfidf_matrix, file_paths

def main():

    global folder_path

    index_path = './temp/tfidf_index'
    files_path = './temp/file_paths.joblib'

    # Check if the index exists
    if not os.path.exists(f'{index_path}_vectorizer.joblib'):
        print("Index not found, building index...")

        corpus, file_paths = build_document_corpus(folder_path)

        tfidf_matrix, vectorizer = create_tfidf_index(corpus)
        save_index(vectorizer, tfidf_matrix, file_paths, index_path=index_path, files_path=files_path)
    else:
        print("Loading existing index...")
        vectorizer, tfidf_matrix, file_paths = load_index(index_path=index_path, files_path=files_path)

    # Example query
    query = "What do we know about Semantic Chunking?"
    print(query)
    top_documents = document_retriever(query, tfidf_matrix, vectorizer, file_paths, top_n=5)

    for doc_path, score in top_documents:
        print(f"Document: {doc_path}, Relevance Score: {score}")

def precompute_embeddings():
    print("Pre-computing document embeddings...")
    # Initialize the retriever which will compute and save embeddings if not already done
    _ = EmbeddingBasedRetriever()

from transformers import BertTokenizer
from transformers import BertModel, BertTokenizer


def generate_embeddings(model_name, text):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.pooler_output.cpu().numpy()  # Use the pooled output for document-level embeddings
    return embeddings

def create_faiss_index(dimension):
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    #index = faiss.IndexIDMap(index)
    return index


def run_indexer(model_name='allenai/llama-small'):

    global folder_path

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    embeddings = []
    document_ids = []  # Keep track of document IDs

    index = None  # We'll initialize the FAISS index after getting the first embedding's dimension

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if file_path.endswith('.pdf') or file_path.endswith('.docx'):
                text = read_pdf(file_path) if file_path.endswith('.pdf') else read_docx(file_path)
                if text:  # Ensure there's text extracted
                    print(f"Processing {file_path}")
                    doc_embedding = generate_embeddings(model_name, text)
                    if index is None:  # Initialize FAISS index with the correct dimension
                        dimension = doc_embedding.shape[1]
                        index = create_faiss_index(dimension)
                    index.add(doc_embedding)  # Add the document embedding to the index
                    document_ids.append(os.path.basename(file_path))

    # At this point, embeddings are indexed in FAISS, and document_ids correlate to the indexed embeddings
    # You might want to save 'index' and 'document_ids' for later use in retrieval
    return index, document_ids

folder_path = './docs'

if __name__ == "__main__":

    precompute_embeddings()

    # Example usage
    index, document_ids = run_indexer()

    # To save the index and document IDs, you could use FAISS's `write_index` and numpy's save function
    faiss.write_index(index, "temp/llama_index.faiss")
    np.save("temp/document_ids.npy", document_ids)

    main()
