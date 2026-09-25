from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import GOOGLE_API_KEY, GEMINI_MODEL,TEMPERATURE

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
)

question = "What is LangChain and why is it used. Give response only one line?"
print(f"Question: {question}\n")

response = llm.invoke([HumanMessage(content=question)])
print(f"Answer: {response.content}")
