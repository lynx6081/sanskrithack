from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import faiss
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI()

# Global variables for loaded data
index = None
verses = None

def load_data():
    """Load the FAISS index and verses metadata"""
    global index, verses
    try:
        # Adjust paths according to your file structure
        index = faiss.read_index("./databse/rigveda.index")
        with open("./databse/rigveda_meta.pkl", "rb") as f:
            verses = pickle.load(f)
        print("✅ Index and metadata loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def embed(texts):
    """Create embeddings using OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

def search(query, topk=5):
    """Search for relevant verses"""
    if index is None or verses is None:
        raise Exception("Index not loaded")
    
    q = embed([query])
    D, I = index.search(q, topk)
    return [(verses[i], float(D[0][j])) for j, i in enumerate(I[0])]

def ask(query, topk=5, model="gpt-4o-mini"):
    """Get answer using RAG (Retrieval Augmented Generation)"""
    # 1. Retrieve relevant verses
    results = search(query, topk=topk)
    
    # 2. Create context
    context = "\n".join([
        f"RV {r['mandala']}.{r['sukta']}.{r['verse']}: {r['text_sa']}"
        for r, _ in results
    ])
    
    # 3. Create prompt for LLM
    prompt = f"""
You are an expert on the Rigveda. Use ONLY the following verses to answer the question. 
If the answer is not present in the verses, give the most relevant answer based on the the query.

Question: {query}

Relevant Verses:
{context}

Answer:
"""
    
    # 4. Get response from OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content, [r for r, _ in results]

# Load data when the app starts
if not load_data():
    print("Warning: Could not load data files. Make sure rigveda.index and rigveda_meta.pkl exist in ./databse/ directory")

@app.route('/')
def home():
    """Serve the frontend"""
    # Read the HTML file we created
    try:
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # If template file doesn't exist, return a simple message
        return """
        <h1>Rigveda Chatbot Backend</h1>
        <p>Backend is running! Create a templates/index.html file or use the frontend separately.</p>
        <p>API endpoint: POST /api/ask</p>
        """

@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        topk = data.get('topk', 5)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if index is None or verses is None:
            return jsonify({'error': 'Database not loaded. Please check server configuration.'}), 500
        
        # Get answer and relevant verses
        answer, relevant_verses = ask(query, topk=topk)
        
        # Format verse references for frontend
        verses_info = [
            {
                'mandala': v['mandala'],
                'sukta': v['sukta'],
                'verse': v['verse'],
                'text_sa': v['text_sa']
            }
            for v in relevant_verses
        ]
        
        return jsonify({
            'answer': answer,
            'verses': verses_info,
            'query': query
        })
        
    except Exception as e:
        print(f"Error in API: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if (index is not None and verses is not None) else 'degraded',
        'database_loaded': index is not None and verses is not None,
        'total_verses': len(verses) if verses else 0
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting Rigveda Chatbot Server...")
    print("📚 Make sure your database files are in ./databse/ directory")
    print("🌐 Frontend will be available at http://localhost:5000")
    print("🔌 API available at http://localhost:5000/api/ask")
    
    app.run(debug=True, host='0.0.0.0', port=5000)