"""
Example 5:   Mix different runnable types — RunnableMap, RunnableLambda, RunnableSequence 
    to create a complex pipeline  
Goal: Run multiple chains in parallel (e.g., summary + sentiment + hashtags), then merge the results.
"""

from langchain.schema.runnable import RunnableMap, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
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

# --- 2️⃣ Define PromptTemplates and chains ---
summary_prompt = PromptTemplate.from_template("Summarize: {text}")
sentiment_prompt = PromptTemplate.from_template("Classify sentiment (positive, neutral, negative) for: {text}")
hashtags_prompt = PromptTemplate.from_template("Generate 5 relevant hashtags for: {text}")

summary_chain = summary_prompt | llm | StrOutputParser()
sentiment_chain = sentiment_prompt | llm | StrOutputParser()
hashtags_chain = hashtags_prompt | llm | StrOutputParser()

# --- 3️⃣ Combine using RunnableMap and RunnableLambda ---
# Run 3 tasks in parallel, then merge results
parallel_chain = RunnableMap({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "hashtags": hashtags_chain
}) | RunnableLambda(lambda outputs: f"{outputs['summary']}\nSentiment: {outputs['sentiment']}\n{outputs['hashtags']}")

# --- 4️⃣ Run it ---
result = parallel_chain.invoke({"text": "AI in education is transforming how teachers personalize learning."})

# --- 5️⃣ Print the model output ---
print(result)
