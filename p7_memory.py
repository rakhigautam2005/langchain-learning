from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers to 2 lines."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])
chain = chat_prompt | model | StrOutputParser()

chat_history = []
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    reply = chain.invoke({"chat_history": chat_history, "query": q})
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=reply))
    print("AI:", reply)