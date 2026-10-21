from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

classify_prompt = PromptTemplate(
    template="Classify the sentiment of this feedback as positive or negative.\n{feedback}\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)
classifier = classify_prompt | model | parser2

pos_prompt = PromptTemplate.from_template(
    "Write ONE short thank-you reply (2 sentences max). No options, no extra tips.\n\nFeedback: {feedback}"
)
neg_prompt = PromptTemplate.from_template(
    "Write ONE short, apologetic and helpful reply (2 sentences max). No options, no extra tips.\n\nFeedback: {feedback}"
)
branch = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive", pos_prompt | model | parser),
    (lambda x: x["sentiment"].sentiment == "negative", neg_prompt | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment"),
)

# carry the original feedback along, add the classifier's result, then route
chain = RunnablePassthrough.assign(sentiment=classifier) | branch

feedback = "Great service, my order arrived early"
print(chain.invoke({"feedback": feedback}))