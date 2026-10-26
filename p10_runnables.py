from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

def word_count(text: str) -> int:
    return len(text.split())

# Part 1: your own Python function becomes a Runnable (no API call)
counter = RunnableLambda(word_count)
print("1) invoke:", counter.invoke("LangChain makes chains easy"))
print("1) batch:", counter.batch(["one two", "one two three", "a b c d e"]))
print()

# Part 2: the | operator builds a RunnableSequence (no API call)
shout = RunnableLambda(lambda s: s.upper())
exclaim = RunnableLambda(lambda s: s + "!!!")
seq = shout | exclaim
print("2) type:", type(seq).__name__)
print("2) result:", seq.invoke("hello"))
print()

# Part 3: RunnableParallel and RunnablePassthrough (no API call)
parallel = RunnableParallel({
    "original": RunnablePassthrough(),
    "upper": RunnableLambda(lambda s: s.upper()),
    "words": RunnableLambda(word_count),
})
print("3) parallel:", parallel.invoke("langchain is fun"))
print()

# Part 4: .assign adds a key to a dict (no API call)
with_count = RunnablePassthrough.assign(count=lambda d: word_count(d["text"]))
print("4) assign:", with_count.invoke({"text": "langchain is fun"}))
print()

# Part 5: a real model with streaming (1 API call)
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)
prompt = PromptTemplate.from_template("Explain {topic} in 2 lines.")
chain = prompt | model | StrOutputParser()
print("5) streaming:")
for piece in chain.stream({"topic": "Runnables in LangChain"}):
    print(piece, end="", flush=True)
print()