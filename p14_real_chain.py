import asyncio
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=1)
chain = PromptTemplate.from_template("Explain {topic} in 1 line.") | model | StrOutputParser()

# 1. batch: two inputs, run at the same time (2 API calls)
print("1) batch:")
start = time.time()
for r in chain.batch([{"topic": "API"}, {"topic": "token"}], config={"max_concurrency": 2}):
    print("-", r)
print("   time:", round(time.time() - start, 1), "seconds")

# 2. stream: the answer arrives in pieces (1 API call)
print("2) stream:")
for piece in chain.stream({"topic": "Runnable"}):
    print(piece, end="", flush=True)
print()

# 3. async (1 API call)
async def main():
    return await chain.ainvoke({"topic": "LCEL"})

print("3) async:", asyncio.run(main()))

# 4. fallback, no API call: a deliberately wrong model name falls back to the real one
broken = ChatGoogleGenerativeAI(model="this-model-does-not-exist", timeout=60, max_retries=0)
safe_chain = (
    PromptTemplate.from_template("Explain {topic} in 1 line.")
    | broken.with_fallbacks([model])
    | StrOutputParser()
)
print("4) fallback: (the first model fails, so the backup answers)")
print(safe_chain.invoke({"topic": "fallback"}))