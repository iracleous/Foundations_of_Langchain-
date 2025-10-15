"""
Example 9: Complete Nutrition Planning Pipeline. Full-featured nutrition AI assistant pipeline.
Goal: Create a multi-step pipeline to generate a personalized 7-day meal plan, 
summarize nutrition, and provide dietary tips.


User Info (age, weight, height, allergies, goal)
       │
       ▼
Meal Plan (7 days) ----------------------> JSON
       │
       ▼
Nutrition Summary (calories, macros) ---> JSON
       │
       ▼
Tips & Warnings -------------------------> JSON
       │
       ▼
Final Structured JSON Output


# Note: In a real application, add error handling, input validation, and possibly
# more sophisticated parsing/formatting of the LLM outputs. 

"""



from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1️⃣ Initialize Azure OpenAI ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model_name=os.getenv("AZURE_OPENAI_MODEL"),  # e.g., "gpt-4o"
    max_tokens=800,
    temperature=0.7
)

# --- 2️⃣ Define prompts ---
meal_plan_prompt = PromptTemplate.from_template(
    """
    Create a 7-day meal plan for a {age}-year-old, {weight}kg, {height}cm, {gender}.
    Allergies: {allergies}.
    Dietary goal: {goal}.
    Include breakfast, lunch, dinner, and snacks for each day.
    Return in JSON format with keys: day1, day2, ..., day7; each day contains meals.
    """
)

nutrition_summary_prompt = PromptTemplate.from_template(
    """
    Summarize the nutrition of the following meal plan in calories and macronutrients:
    {meal_plan}
    Return JSON with total_calories, protein_g, carbs_g, fat_g.
    """
)

tips_prompt = PromptTemplate.from_template(
    """
    Based on the meal plan and user's allergies ({allergies}), provide dietary tips, warnings, and substitutions.
    Return a JSON object with keys: tips, warnings, substitutions.
    """
)

# --- 3️⃣ Build Runnables ---
meal_plan_chain = meal_plan_prompt | llm | StrOutputParser()
nutrition_summary_chain = nutrition_summary_prompt | llm | StrOutputParser()
tips_chain = tips_prompt | llm | StrOutputParser()

# --- 4️⃣ Combine steps into a single pipeline ---
nutrition_pipeline = RunnableLambda(
    lambda user_info: {
        "meal_plan": meal_plan_chain.invoke(user_info)
    }
) | RunnableLambda(
    lambda data: {
        **data,
        "nutrition_summary": nutrition_summary_chain.invoke({"meal_plan": data["meal_plan"]})
    }
) | RunnableLambda(
    lambda data: {
        **data,
        "tips": tips_chain.invoke({"meal_plan": data["meal_plan"], "allergies": user_info["allergies"]})
    }
)

# --- 5️⃣ Example user input ---
user_info = {
    "age": 51,
    "weight": 112,
    "height": 176,
    "gender": "male",
    "allergies": ["milk", "pork", "gluten", "sesame"],
    "goal": "weight loss"
}

# --- 6️⃣ Run pipeline ---
result = nutrition_pipeline.invoke(user_info)

# --- 7️⃣ Display structured output ---
import json
print(json.dumps(result, indent=4))
