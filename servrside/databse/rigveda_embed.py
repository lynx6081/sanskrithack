import faiss, numpy as np, json, pickle
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-large",   # or "text-embedding-3-small"
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

# Load Rigveda JSON
verses = json.load(open("../parsing/rigveda_merged.json", "r", encoding="utf-8"))
texts = [v["text_sa"] for v in verses]

# Build embeddings with OpenAI
X = embed(texts)

# Build FAISS index
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# Save index + metadata
faiss.write_index(index, "rigveda.index")
with open("rigveda_meta.pkl", "wb") as f:
    pickle.dump(verses, f)

print("✅ Index built and saved")
