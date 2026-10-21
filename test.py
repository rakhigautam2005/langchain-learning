from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

response = model.invoke("Explain LangChain in one sentence")
print(response.text)