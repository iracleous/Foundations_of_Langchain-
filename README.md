Instructions to use the examples

Setup environment
https://github.com/codehub-learn/development-environment-setup/blob/main/Generative_and_Agentic_AI.md

GitHub repository



pyproject.toml -> requires-python = ">=3.12, <4.0"


poetry add openai langchain langchain_openai dotenv langchain_community text_generation langchain_huggingface

poetry add sentence-transformers langchain_chroma





============================================================

Day 1 Examples

Example 1: Call Azure OpenAI with Python
Goal: low-level / direct API example (not LangChain this time) using the official openai Python SDK

Example 2:   RunnableSequence (pipeline) 
Goal: Demonstrate a simple Runnable Sequential Pipeline using the modern LangChain interface

Example 3:   RunnableSequence (pipeline). Unified way to connect components — such as prompts, models, and output parsers — in sequence
Goal: Implement a multi-step Runnable Sequential Pipeline using modern LangChain (Runnable interface), 
    where the output of one step feeds into the next.

Example 4:   Complete, realistic, “extensive” Runnable Sequential Pipeline using the modern LangChain interface.
Goal:  Demonstrate how to build a complete, realistic, multi-step AI workflow using LangChain’s modern 
Runnable Sequential Pipeline architecture — replacing deprecated LLMChain / SequentialChain with 
the unified Runnable interface.

Example 5:   Mix different runnable types — RunnableMap, RunnableLambda, RunnableSequence 
    to create a complex pipeline  
Goal: Run multiple chains in parallel (e.g., summary + sentiment + hashtags), then merge the results.

Example 6:  Data Enrichment Pipeline (Batch Processing)
Goal: Take tabular or API data, enrich each record via LLM, and return structured output.

Example 7: Evaluation / Comparison Pipeline
Goal: Have one LLM generate output and another LLM evaluate or score it.

Example 8: Chat-style Multi-role Pipeline
Goal: Use ChatPromptTemplate for system + user roles (ideal for GPT-4o chat models).

Example 9: Complete Nutrition Planning Pipeline. Full-featured nutrition AI assistant pipeline.
Goal: Create a multi-step pipeline to generate a personalized 7-day meal plan, 
summarize nutrition, and provide dietary tips.
