from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)
response = model.invoke("What is an API? Answer in 2 lines.")

print("TYPE:", type(response))
print("TEXT:", response.text)
print("TOKENS:", response.usage_metadata)