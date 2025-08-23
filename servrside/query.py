import faiss, numpy as np, pickle
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# Load saved index + metadata
index = faiss.read_index("./databse/rigveda.index")
with open("./databse/rigveda_meta.pkl", "rb") as f:
    verses = pickle.load(f)

def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

def search(query, topk=5):
    q = embed([query])
    D, I = index.search(q, topk)
    return [(verses[i], float(D[0][j])) for j, i in enumerate(I[0])]

def ask(query, topk=5, model="gpt-4o-mini"):
    # 1. retrieve
    results = search(query, topk=topk)
    
    # 2. make context
    context = "\n".join([
        f"RV {r['mandala']}.{r['sukta']}.{r['verse']}: {r['text_sa']}"
        for r, _ in results
    ])
    
    # 3. ask LLM
    prompt = f"""
You are an expert on the Rigveda. Use ONLY the following verses to answer the question. 
If the answer is not present in the verses, give the most relevant answer based on the the query.

Question: {query}

Relevant Verses:
{context}

Answer:
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 🔹 Example usage:
print(ask("What is rigveda all about"))

