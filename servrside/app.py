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

def ask(query, topk=5, model="gpt-4o-mini", is_intro=False):
    """Get answer using RAG (Retrieval Augmented Generation)"""
    
    if is_intro:
        # Special introduction message - no need to search
        intro_response = """
🕉️ **Namaste and welcome, dear seeker of wisdom!** 🕉️

I am your passionate Rigveda tutor, and I'm absolutely thrilled that you've chosen to embark on this incredible journey into the world's oldest sacred text! The Rigveda is like a magnificent treasure chest filled with 3,500-year-old wisdom, beautiful hymns, and profound insights about life, nature, and the divine.

Think of me as your friendly guide who will help you understand these ancient Sanskrit verses in the simplest way possible - whether you're 8 or 80 years old! 

**What would you love to explore today?** Here are some fascinating topics we could dive into:

🔥 **Fire & Agni** - Learn about the sacred fire god who connects earth and heaven
⚡ **Indra the Mighty** - Discover the thunder god who defeats demons and brings rain  
🌙 **Soma & Sacred Rituals** - Understand the mysterious divine drink and ceremonies
🌍 **Creation Stories** - Explore how the universe began according to Rigvedic seers
🎵 **Hymns & Poetry** - Appreciate the beautiful language and metaphors
⚖️ **Dharma & Ethics** - Learn about righteous living and moral principles

Just click on any topic above, or feel free to ask me anything that sparks your curiosity! I love questions like "Why is fire so important?" or "What makes Rigveda special?" - no question is too simple or too complex!

**What shall we discover together today?** 🌟
"""
        return intro_response, []
    
    # 1. Retrieve relevant verses
    results = search(query, topk=topk)
    
    # 2. Create context
    context = "\n".join([
        f"RV {r['mandala']}.{r['sukta']}.{r['verse']}: {r['text_sa']}"
        for r, _ in results
    ])
    
    # 3. Create engaging tutor prompt
    prompt = f"""
You are a passionate and enthusiastic Rigveda tutor who absolutely loves teaching about this ancient wisdom. Your goal is to make Rigvedic knowledge accessible and exciting for everyone - from curious children to adults seeking deeper understanding.

Your personality:
- Warm, friendly, and encouraging
- Uses simple language but maintains depth
- Gives examples and analogies that anyone can understand
- Always follows up with engaging questions to keep the conversation going
- Shows genuine excitement about sharing Rigvedic wisdom
- Explains Sanskrit terms when used
- Connects ancient wisdom to modern life

Based on the following authentic Rigvedic verses, provide a comprehensive, engaging answer that:
1. Explains the topic clearly and simply
2. Uses analogies or examples when helpful
3. Shares the cultural and spiritual significance
4. Ends with a follow-up question to encourage deeper learning

Student's Question: {query}

Relevant Rigvedic Verses:
{context}

Provide your enthusiastic, educational response:
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
        is_intro = data.get('is_intro', False)
        
        if not query and not is_intro:
            return jsonify({'error': 'Query is required'}), 400
        
        if not is_intro and (index is None or verses is None):
            return jsonify({'error': 'Database not loaded. Please check server configuration.'}), 500
        
        # Get answer and relevant verses
        answer, relevant_verses = ask(query, topk=topk, is_intro=is_intro)
        
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
            'query': query,
            'is_intro': is_intro
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