"""
Example 8: Chat-style Multi-role Pipeline
Goal: Use ChatPromptTemplate for system + user roles (ideal for GPT-4o chat models).
"""

from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model_name=os.getenv("AZURE_OPENAI_MODEL"),
    max_tokens=150,
    temperature=0.7
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional dietitian."),
    ("human", "Create a 3-day gluten-free meal plan for a 51-year-old male with prediabetes.")
])

chain = chat_prompt | llm | StrOutputParser()
print(chain.invoke({}))