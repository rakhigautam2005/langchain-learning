from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)
parser = StrOutputParser()

PART = "C"   # change to "B", then "C", and run again each time

if PART == "A":
    # Simple chain: prompt -> model -> parser
    prompt = PromptTemplate.from_template("Give me 3 interesting facts about {topic}")
    chain = prompt | model | parser
    print(chain.invoke({"topic": "cricket"}))

elif PART == "B":
    # Sequential chain: report first, then a summary of that report
    prompt1 = PromptTemplate.from_template("Write a short report (5 lines) on {topic}")
    prompt2 = PromptTemplate.from_template("Summarize this in 2 lines:\n{text}")
    chain = prompt1 | model | parser | prompt2 | model | parser
    print(chain.invoke({"topic": "black holes"}))

elif PART == "C":
    # Parallel chain: notes and quiz at the same time, then merge
    notes_prompt = PromptTemplate.from_template("Make 3 short notes from:\n{text}")
    quiz_prompt = PromptTemplate.from_template("Make 3 quiz questions from:\n{text}")
    merge_prompt = PromptTemplate.from_template(
        "Merge into one study document.\nNOTES:\n{notes}\nQUIZ:\n{quiz}"
    )
    parallel = RunnableParallel({
        "notes": notes_prompt | model | parser,
        "quiz": quiz_prompt | model | parser,
    })
    chain = parallel | merge_prompt | model | parser
    text = "An API is a set of rules that lets two apps talk to each other. A client sends a request and the server sends back a response."
    print(chain.invoke({"text": text}))