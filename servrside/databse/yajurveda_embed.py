import faiss, numpy as np, json, pickle
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# -------- Embed Function --------
def embed(texts, model="text-embedding-3-large"):
    # Handles batching if needed
    embeddings = []
    batch_size = 100  # adjust if needed
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([d.embedding for d in response.data])
    return np.array(embeddings, dtype="float32")

# -------- Load JSON --------
verses = json.load(open("../parsing/yajurveda_merged.json", "r", encoding="utf-8"))
texts = [v["text"] for v in verses]   # analytic text field

# -------- Build Embeddings --------
X = embed(texts)

# -------- Build FAISS Index --------
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# -------- Save Index + Metadata --------
faiss.write_index(index, "yajurveda.index")
with open("yajurveda_meta.pkl", "wb") as f:
    pickle.dump(verses, f)

print("✅ Yajurveda index built and saved")
