"""
Example 6:  Data Enrichment Pipeline (Batch Processing)
Goal: Take tabular or API data, enrich each record via LLM, and return structured output.
"""

from langchain.schema.runnable import RunnableLambda, RunnableMap
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

#--- 2️⃣ Sample tabular data (list of dicts) ---
# In real scenarios, this could come from a CSV, database, or API
data = [
    {"product": "smartwatch", "review": "Battery life is excellent but the strap broke fast."},
    {"product": "headphones", "review": "Sound is great, noise canceling is weak."},
]


#-- 3️⃣ Define a chain to analyze each review ---

prompt = PromptTemplate.from_template(
    "Summarize sentiment and key features mentioned in this review: {review}"
)
review_chain = prompt | llm | StrOutputParser()

#--- 4️⃣ Define the full pipeline ---
pipeline = RunnableLambda(lambda dataset: [
    {"product": item["product"], "analysis": review_chain.invoke({"review": item["review"]})}
    for item in dataset
])

#--- 5️⃣ Run the pipeline ---
print(pipeline.invoke(data))
