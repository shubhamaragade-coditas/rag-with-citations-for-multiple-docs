from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ai21 import AI21Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever

try:
  deep_learning_file_loader: PyPDFLoader = PyPDFLoader("Deep Learning.pdf")
  deep_learning_file_pages: list[Document] = deep_learning_file_loader.load()
except FileNotFoundError:
  print("Error: Could not find the PDF file 'Deep Learning.pdf'")
  exit()


try:
  working_file_loader: PyPDFLoader = PyPDFLoader("working.pdf")
  working_file_pages: list[Document] = working_file_loader.load()
except FileNotFoundError:
  print("Error: Could not find the PDF file 'working.pdf'")
  exit()

deep_learning_file_text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
deep_learning_file_documents: list[Document] = deep_learning_file_text_splitter.split_documents(deep_learning_file_pages)

working_file_text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
working_file_documents: list[Document] = working_file_text_splitter.split_documents(working_file_pages)


try:
  embedding_model: AI21Embeddings = AI21Embeddings()
  deep_learning_file_db: FAISS = FAISS.from_documents(documents=deep_learning_file_documents, embedding=embedding_model)
except Exception:  
  exit()

try:
  working_file_db: FAISS = FAISS.from_documents(documents=working_file_documents, embedding=embedding_model)
except Exception:  
  exit()

combined_db = deep_learning_file_db

combined_db.merge_from(working_file_db)

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-pro")

retriever: VectorStoreRetriever = combined_db.as_retriever()
retrival_source_chain: RetrievalQAWithSourcesChain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever,
                                                                          return_source_documents=True)

retrival_source_chain.combine_documents_chain.llm_chain.prompt.template = 'QUESTION: \n=========Content: \n  ...\n=========\n{summaries}\n=========\nFINAL ANSWER:'

continue_asking = True
while continue_asking:
  question: str = input("Ask a question: ")
  try:
    answer = retrival_source_chain.invoke(question)
    print(answer["answer"])
    print(answer["source_documents"])
  except Exception as e:  
    print("Error:", e)

  continue_asking = int(input("Do you want to continue:\n 0. No\n 1. Yes\n"))
