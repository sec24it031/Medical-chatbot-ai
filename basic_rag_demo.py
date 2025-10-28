from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_Openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

from langchain.chains import RetrievalQA

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


loaders = TextLoader("data/Anatomy&Physiology.pdf/Cardiology.pdf/Dentistry.pdf/Emergency.pdf/Gastrology.pdf/General.pdf/InfectiousDisease.pdf/InternalMedicine.pdf/Nephrology.pdf")
document = loaders.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(document)

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever())

#query
query = "What are the common symptoms of kidney disease?"
result = qa.invoke(query)
print(result)

