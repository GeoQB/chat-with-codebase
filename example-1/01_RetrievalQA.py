###
#
# Beispiel für eine "RetrievalQA Chain"
#
# https://medium.com/@ahmed.mohiuddin.architecture/using-ai-to-chat-with-your-documents-leveraging-langchain-faiss-and-openai-3281acfcc4e9

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

def query_pdf(query):
    # Load document using PyPDFLoader document loader
    loader = PyPDFLoader("docs/LLM Powered Autonomous Agents.pdf")
    documents = loader.load()
    # Split document in chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()

    # Create vectors
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Persist the vectors locally on disk
    vectorstore.save_local("faiss_index_PDF")

    # Load from local storage
    persisted_vectorstore = FAISS.load_local("faiss_index_PDF", embeddings, allow_dangerous_deserialization=True)

    # Use RetrievalQA chain for orchestration
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=persisted_vectorstore.as_retriever())
    result = qa.run(query)
    print(result)


def main():

    query = input("Type in your query: \n")
    while query != "exit":
        query_pdf(query)
        query = input("Type in your query: \n")


if __name__ == "__main__":
    main()