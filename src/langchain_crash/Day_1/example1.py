"""
Example 1: Call Azure OpenAI with Python
Goal: low-level / direct API example (not LangChain this time) using the official openai Python SDK
"""

from openai import AzureOpenAI, OpenAI
import os
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()


# client = OpenAI(
#         api_key=os.getenv("OPENAI_API_KEY")
#     ) 

# --- 1️⃣ Initialize the client ---
client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
    )

# --- 2️⃣ Define the prompt ---
prompt = "Hello, Azure OpenAI!"


# --- 3️⃣ Call the model ---
response = client.chat.completions.create(
    model = os.getenv("AZURE_OPENAI_MODEL"),
    messages=[{"role": "user", "content":  prompt}]
)

# --- 4️⃣ Print the model output ---
print(response.choices[0].message.content)