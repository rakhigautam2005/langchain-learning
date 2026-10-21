from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)

# Part A: template with blanks
template = PromptTemplate.from_template("Explain {topic} to a {audience} in 2 lines.")
filled = template.invoke({"topic": "APIs", "audience": "10-year-old"})
print("A) FILLED PROMPT:", filled.to_string())
print("A) ANSWER:", model.invoke(filled).text)
print()

# Part B: role-based messages
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Answer in 2 lines."),
    ("human", "Explain {topic}"),
])
messages = chat_prompt.invoke({"domain": "cricket", "topic": "LBW"})
print("B) MESSAGES:", messages.to_messages())
print("B) ANSWER:", model.invoke(messages).text)
print()

# Part C: few-shot
few_shot = PromptTemplate.from_template(
    "Classify the message as billing, technical, or other.\n"
    "Message: Refund not received -> billing\n"
    "Message: App keeps crashing -> technical\n"
    "Message: {msg} ->"
)
print("C) ANSWER:", model.invoke(few_shot.invoke({"msg": "I was charged twice"})).text)