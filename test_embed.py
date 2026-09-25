from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, EMBED_MODEL

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

text = "LangChain is a framework for building LLM applications."
vector = embeddings.embed_query(text)

print(f"Model used: {EMBED_MODEL}")
print(f"Vector length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
