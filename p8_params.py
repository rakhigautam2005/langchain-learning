from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
question = "Write a one-line slogan for a cab booking app. Reply with just the slogan."

for temp in [0.0, 2.0]:
    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=temp,
                                   timeout=60, max_retries=1)
    print(f"--- temperature = {temp} ---")
    for i in range(3):
        print(model.invoke(question).text)