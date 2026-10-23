from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)

docs = [
    "I love playing cricket",
    "Cricket is my favourite sport",
    "The stock market fell today",
]
query = "What happened in the share market?"
doc_vectors = emb.embed_documents(docs)
query_vector = emb.embed_query(query)

print("Vector length:", len(query_vector))
print("First 5 numbers:", [round(x, 4) for x in query_vector[:5]])
print("-----")
for d, v in zip(docs, doc_vectors):
    print(round(cosine(query_vector, v), 3), "|", d)