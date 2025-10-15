"""
Example 4:   Complete, realistic, “extensive” Runnable Sequential Pipeline using the modern LangChain interface.
Goal:  Demonstrate how to build a complete, realistic, multi-step AI workflow using LangChain’s modern 
Runnable Sequential Pipeline architecture — replacing deprecated LLMChain / SequentialChain with 
the unified Runnable interface.
"""

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import  RunnableLambda

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

# --- 2️⃣ Define PromptTemplates ---
title_prompt = PromptTemplate.from_template(
    "Create an engaging blog title about {topic}."
)

summary_prompt = PromptTemplate.from_template(
    "Write a 3-sentence summary for a blog post titled '{title}'."
)

article_prompt = PromptTemplate.from_template(
    "Write a detailed, well-structured article for a blog titled '{title}' about {topic}. "
    "Include an introduction, 3 body sections, and a conclusion."
)

keywords_prompt = PromptTemplate.from_template(
    "Extract 5 SEO-friendly keywords from this text:\n\n{article}"
)

# --- 3️⃣ Define runnable steps ---
title_chain = title_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()
article_chain = article_prompt | llm | StrOutputParser()
keywords_chain = keywords_prompt | llm | StrOutputParser()

# --- 4️⃣ Combine steps using Runnable logic ---

# Step A → generate title
# Step B → use title to generate summary and article
# Step C → extract keywords from article
# Step D → return structured output

pipeline = (
    # Step A
    title_chain
    # Step B
    | RunnableLambda(
        lambda title: {
            "title": title,
            "summary": summary_chain.invoke({"title": title}),
            "article": article_chain.invoke({"title": title, "topic": current_topic}),
        }
    )
    # Step C
    | RunnableLambda(
        lambda data: {
            **data,
            "keywords": keywords_chain.invoke({"article": data["article"]}),
        }
    )
)

# --- 5️⃣ Run the full pipeline ---
current_topic = "The Role of Artificial Intelligence in Modern Education"

result = pipeline.invoke(current_topic)

# --- 6️⃣ Display results ---
print("\n📝 TITLE:\n", result["title"])
print("\n📋 SUMMARY:\n", result["summary"])
print("\n📘 ARTICLE:\n", result["article"])
print("\n🏷️ KEYWORDS:\n", result["keywords"])
