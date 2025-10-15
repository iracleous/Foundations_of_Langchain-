"""
Example 11: Retrieval-Augmented Generation (RAG) Pipeline
Goal: Use a retriever (Chroma) + HuggingFace embeddings + LLM to answer questions from documents.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI  # or AzureChatOpenAI if using Azure
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.docstore.document import Document

# --- 1️⃣ Prepare documents ---
docs = [
    Document(page_content="AI can help teachers personalize lessons."),
    Document(page_content="RAG pipelines combine retrieval from vector stores with LLMs for accurate answers.")
]

# --- 2️⃣ Set up embeddings and vector store ---
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, hf_embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 3️⃣ LLM and prompt ---

import os
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()

llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),  # default to gpt-4o
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=200
    ) 

prompt = PromptTemplate.from_template(
    "Answer the question based on the context below:\n{context}\n\nQuestion: {question}"
)

qa_chain = create_stuff_documents_chain(llm, prompt)

# --- 4️⃣ Combine retriever and QA in a pipeline ---
rag_pipeline = RunnableMap({
    "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
    "question": RunnableLambda(lambda x: x["question"])
}) | qa_chain

# --- 5️⃣ Run the pipeline ---
query = {"question": "How can AI support teachers?"}
result = rag_pipeline.invoke(query)

print("Answer:", result)
