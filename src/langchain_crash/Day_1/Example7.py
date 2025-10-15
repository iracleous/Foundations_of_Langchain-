"""
Example 7: Evaluation / Comparison Pipeline
Goal: Have one LLM generate output and another LLM evaluate or score it.
"""

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
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

# Define prompts
generator_prompt = PromptTemplate.from_template("Write a paragraph about {topic}.")
evaluator_prompt = PromptTemplate.from_template("Rate the following text for clarity (1-10):\n\n{text}")

# Create runnables
generator = generator_prompt | llm | StrOutputParser()
evaluator = evaluator_prompt | llm | StrOutputParser()

# Build pipeline with pipes
evaluation_pipeline = RunnableLambda(
    lambda input_data: generator.invoke({"topic": input_data["topic"]})
) | RunnableLambda(
    lambda text: print (text) or evaluator.invoke({"text": text})
)

# Run pipeline
result = evaluation_pipeline.invoke({"topic": "The ethics of AI"})
print("Evaluation Score:", result)
