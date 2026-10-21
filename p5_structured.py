from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", timeout=60, max_retries=2)

TEXT = "App crashes whenever I try to pay, please fix ASAP!! Order A123."
PART = "B"   # change to "A", "B" or "C", save, and run again

if PART == "A":   # TypedDict: returns a plain dict, no validation
    class Ticket(TypedDict):
        """Details extracted from a customer support message."""
        category: Annotated[str, ..., "One of: billing, technical, other"]
        urgency: Annotated[int, ..., "1 (low) to 5 (critical)"]
        order_id: Annotated[Optional[str], None, "Order ID if mentioned, else null"]

    result = model.with_structured_output(Ticket).invoke(TEXT)
    print(type(result), result)

elif PART == "B":   # Pydantic: returns an object, validates types and ranges
    class Ticket(BaseModel):
        """Details extracted from a customer support message."""
        category: Literal["billing", "technical", "other"]
        urgency: int = Field(ge=1, le=5, description="1 (low) to 5 (critical)")
        order_id: Optional[str] = Field(default=None, description="Order ID if mentioned, else null")

    # Validation test (no API call): Pydantic should reject urgency=9
    try:
        Ticket(category="billing", urgency=9)
    except ValidationError as e:
        print("Validation error caught:")
        print(e)
    print("-----")

    result = model.with_structured_output(Ticket).invoke(TEXT)
    print(type(result), result)
    print("Urgency:", result.urgency)

elif PART == "C":   # JSON schema: returns a dict
    schema = {
        "title": "Ticket",
        "description": "Details extracted from a customer support message.",
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["billing", "technical", "other"]},
            "urgency": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["category", "urgency"],
    }
    result = model.with_structured_output(schema).invoke(TEXT)
    print(type(result), result)

else:
    print("PART must be exactly A, B or C")