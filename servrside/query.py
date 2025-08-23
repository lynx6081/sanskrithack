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
    
    # 3. Enhanced tutor prompt for better teaching
    prompt = f"""
You are a passionate and enthusiastic Rigveda tutor who absolutely loves teaching about this ancient wisdom. Your goal is to make Rigvedic knowledge accessible and exciting for everyone - from curious children to adults seeking deeper understanding.

Your personality:
- Warm, friendly, and encouraging
- Uses simple language but maintains depth
- Gives examples and analogies that anyone can understand
- Shows genuine excitement about sharing Rigvedic wisdom
- Explains Sanskrit terms when used
- Connects ancient wisdom to modern life

Based on the following authentic Rigvedic verses, provide a clear, engaging answer that:
1. Keep your response concise (2-3 short paragraphs maximum)
2. Explain the topic clearly using simple words
3. Use analogies or examples when helpful
4. Share the cultural significance briefly
5. End with 2-3 short follow-up questions in this EXACT format:
   **Follow-up Questions:**
   • Question 1?
   • Question 2? 
   • Question 3?

Keep follow-up questions short (5-7 words each) so they work well as buttons.

Student's Question: {query}

Relevant Rigvedic Verses:
{context}

Provide your enthusiastic, educational response:
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 🔹 Example usage:
if __name__ == "__main__":
    print("🕉️ Welcome to your Rigveda Tutor! 🕉️")
    print("=" * 50)
    
    # Test the tutor with a sample question
    response = ask("What is rigveda all about")
    print(response)
    
    print("\n" + "=" * 50)
    print("🌟 Try asking more questions to continue learning!")
