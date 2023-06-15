from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import VectorDBQA
from pathlib import Path
import pandas as pd
import os
import openai
import chromadb

pip install tiktoken
pip install chromadb
pip install pypdf
key = os.environ['OPENAI_API_KEY']

llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")
text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
embeddings = OpenAIEmbeddings(openai_api_key=key)

def pdf_summarizer(pdf_paths):
    names = []
    summaries = []
    q1s = []
    q2s = []
    for path in pdf_paths:
        file_name = Path(path).stem
        loader = PyPDFLoader(path)
        data = loader.load()
        data = text_splitter.split_documents(data)
        summary = chain.run(data)
        names.append(file_name)
        summaries.append(summary)
        pagesearch = Chroma.from_documents(data, embeddings)
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=pagesearch)
        query1 = "Summarize what section one thousand one hundred fifteen covers?"
        query2 = "Summarize section three hundred thirty-two?"
        q1 = qa.run(query1)
        q2 = qa.run(query2)
        print(q1)
        print(q2)
        q1s.append(q1)
        q2s.append(q2)
    df = pd.DataFrame({'Doc#':list(range(1,len(pdf_paths)+1)), 'Title/Name':names, 'Summary': summaries, 'Query1': q1s, 'Query2': q2s})
    return df

pdf_paths = ["C:/Users/atul60176/Downloads/Legal_Docs/LegalContract_1.pdf", "C:/Users/atul60176/Downloads/Legal_Docs/LegalContract_2.pdf"]

df = pdf_summarizer(pdf_paths)
df.to_excel('C:/Users/atul60176/Downloads/Legal_Docs//output.xlsx',index=False)
