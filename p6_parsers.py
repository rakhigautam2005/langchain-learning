from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)

parser = JsonOutputParser()
prompt = PromptTemplate(
    template="Give the name, age and city of a fictional {place} person.\n{format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# See the instruction the parser adds to your prompt (no API call)
print(prompt.invoke({"place": "Indian"}).to_string())
print("-----")

chain = prompt | model | parser
result = chain.invoke({"place": "Indian"})
print(type(result), result)
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City the person lives in")

strict_parser = PydanticOutputParser(pydantic_object=Person)
print("-----")
print(strict_parser.get_format_instructions())