"""
Example 3:   RunnableSequence (pipeline). Unified way to connect components — such as prompts, models, and output parsers — in sequence
Goal: Implement a multi-step Runnable Sequential Pipeline using modern LangChain (Runnable interface), 
    where the output of one step feeds into the next.
"""

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

import os
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()

# --- 1️⃣ Initialize the LLM ---
llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    model_name = os.getenv("AZURE_OPENAI_MODEL"),
    max_tokens=150, 
    temperature=0.7
)

# --- 2️⃣ Define prompts ---
title_prompt = PromptTemplate.from_template(
    "Create an engaging blog post title about {topic}."
)
content_prompt = PromptTemplate.from_template(
    "Write a detailed blog post based on the title: {title}."
)

# --- 3️⃣ Build each runnable component ---
title_chain = title_prompt | llm | StrOutputParser()
content_chain = content_prompt | llm | StrOutputParser()

# --- 4️⃣ Sequential logic (manual chaining) ---
def full_chain(input_data):
    title = title_chain.invoke({"topic": input_data["topic"]})
    content = content_chain.invoke({"title": title})
    return {"title": title, "content": content}

# --- 5️⃣ Run it ---
result = full_chain({"topic": "Artificial Intelligence in Education"})

# --- 6️⃣ Print the model output ---
print("📝 Title:", result["title"])
print("\n📘 Content:\n", result["content"])
