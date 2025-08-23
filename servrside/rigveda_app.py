from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import faiss
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import uuid

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

# In-memory conversation storage (in production, use a proper database)
conversations = {}
user_quiz_states = {}

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

def extract_topics_from_conversation(conversation_history):
    """Extract main topics discussed in the conversation using GPT"""
    if not conversation_history or len(conversation_history) < 2:
        return []
    
    # Get recent conversation for topic extraction
    recent_messages = conversation_history[-6:]  # Last 6 messages
    context = "\n".join([
        f"{'User' if msg['sender'] == 'user' else 'Tutor'}: {msg['text']}" 
        for msg in recent_messages
    ])
    
    topic_extraction_prompt = f"""
Based on this Rigveda tutoring conversation, identify the main topics and concepts discussed. 
Focus on specific Rigvedic themes, deities, concepts, or practices mentioned.

Conversation:
{context}

List the main topics as a simple comma-separated list (max 5 topics). Examples:
Agni, fire rituals, Indra, creation myths, dharma, soma, hymns, cosmic order

Topics discussed:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": topic_extraction_prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        topics_text = response.choices[0].message.content.strip()
        topics = [topic.strip() for topic in topics_text.split(',') if topic.strip()]
        return topics[:5]  # Max 5 topics
        
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []

def generate_mcq_quiz(topics, conversation_context):
    """Generate MCQ quiz based on discussed topics"""
    if not topics:
        return None
    
    topics_str = ", ".join(topics)
    
    quiz_prompt = f"""
You are creating a quiz for a student who has been learning about Rigveda. Based on the topics they discussed: {topics_str}

Create 3 multiple choice questions (easy to moderate level) about these Rigvedic topics. 
Each question should have 4 options (A, B, C, D) with exactly one correct answer.

Format your response as valid JSON:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "A",
      "explanation": "Brief explanation of why this answer is correct"
    }}
  ]
}}

Focus on:
- Basic concepts and facts about Rigveda
- Deities and their attributes  
- Important themes and symbols
- Cultural significance

Keep questions clear and educational. Avoid overly complex Sanskrit terms without explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": quiz_prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        quiz_text = response.choices[0].message.content.strip()
        # Remove potential markdown formatting
        if quiz_text.startswith('```json'):
            quiz_text = quiz_text.replace('```json', '').replace('```', '').strip()
        elif quiz_text.startswith('```'):
            quiz_text = quiz_text.replace('```', '').strip()
            
        quiz_data = json.loads(quiz_text)
        return quiz_data
        
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return None

def should_trigger_quiz(session_id):
    """Check if quiz should be triggered based on conversation count"""
    if session_id not in user_quiz_states:
        user_quiz_states[session_id] = {
            'message_count': 0,
            'last_quiz_at': 0,
            'quiz_frequency': 3  # Every 3 meaningful exchanges (reduced from 4)
        }
    
    state = user_quiz_states[session_id]
    state['message_count'] += 1
    
    # Trigger quiz every 3 meaningful exchanges
    if state['message_count'] - state['last_quiz_at'] >= state['quiz_frequency']:
        state['last_quiz_at'] = state['message_count']
        return True
    
    return False

def ask(query, topk=5, model="gpt-4o-mini", is_intro=False, session_id=None):
    """Get answer using RAG with conversation tracking"""
    
    if is_intro:
        # Special introduction message
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
        return intro_response, [], False
    
    # Initialize conversation tracking
    if session_id and session_id not in conversations:
        conversations[session_id] = []
    
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
    
    # 4. Get response from OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    
    # 5. Track conversation and check for quiz trigger
    quiz_triggered = False
    if session_id:
        # Add to conversation history
        conversations[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'sender': 'user',
            'text': query
        })
        conversations[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'sender': 'bot',
            'text': answer
        })
        
        # Check if quiz should be triggered
        quiz_triggered = should_trigger_quiz(session_id)
    
    return answer, [r for r, _ in results], quiz_triggered

# Load data when the app starts
if not load_data():
    print("Warning: Could not load data files. Make sure rigveda.index and rigveda_meta.pkl exist in ./database/ directory")

@app.route('/')
def home():
    """Serve the frontend"""
    try:
        # Try to serve from templates/rigveda/ directory
        with open('templates/rigveda/index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        try:
            # Fallback to templates/ directory
            with open('templates/index.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return """
            <h1>🔥 Rigveda Chatbot Backend with Quiz Feature</h1>
            <p>Backend is running! Please create the frontend file at:</p>
            <ul>
                <li><code>templates/rigveda/index.html</code> - Recommended location</li>
                <li><code>templates/index.html</code> - Fallback location</li>
            </ul>
            <p>API endpoints:</p>
            <ul>
                <li>POST /api/rigveda/ask - For chat functionality</li>
                <li>POST /api/rigveda/generate-quiz - For quiz generation</li>
                <li>POST /api/rigveda/submit-quiz - For quiz submission</li>
                <li>GET /api/health - Health check</li>
            </ul>
            """

# Fixed API endpoints to match the main platform expectations
@app.route('/api/rigveda/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        topk = data.get('topk', 5)
        is_intro = data.get('is_intro', False)
        session_id = data.get('session_id', 'default')
        
        if not query and not is_intro:
            return jsonify({'error': 'Query is required'}), 400
        
        if not is_intro and (index is None or verses is None):
            return jsonify({'error': 'Database not loaded. Please check server configuration.'}), 500
        
        # Get answer and check for quiz trigger
        answer, relevant_verses, quiz_triggered = ask(query, topk=topk, is_intro=is_intro, session_id=session_id)
        
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
            'is_intro': is_intro,
            'quiz_triggered': quiz_triggered,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error in API: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/api/rigveda/generate-quiz', methods=['POST'])
def api_generate_quiz():
    """API endpoint for generating quiz"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in conversations:
            return jsonify({'error': 'No conversation history found'}), 400
        
        # Extract topics from conversation
        topics = extract_topics_from_conversation(conversations[session_id])
        
        if not topics:
            return jsonify({'error': 'No topics found in conversation'}), 400
        
        # Generate quiz
        quiz_data = generate_mcq_quiz(topics, conversations[session_id])
        
        if not quiz_data:
            return jsonify({'error': 'Failed to generate quiz'}), 500
        
        return jsonify({
            'quiz': quiz_data,
            'topics': topics,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return jsonify({'error': 'Failed to generate quiz'}), 500

@app.route('/api/rigveda/submit-quiz', methods=['POST'])
def api_submit_quiz():
    """API endpoint for submitting quiz answers"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        answers = data.get('answers', {})
        quiz_questions = data.get('quiz_questions', [])
        
        # Calculate score
        correct_count = 0
        results = []
        
        for i, question in enumerate(quiz_questions):
            question_id = str(i)
            user_answer = answers.get(question_id)
            correct_answer = question['correct_answer']
            is_correct = user_answer == correct_answer
            
            if is_correct:
                correct_count += 1
            
            results.append({
                'question': question['question'],
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'explanation': question['explanation']
            })
        
        total_questions = len(quiz_questions)
        score_percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # Generate encouraging feedback
        if score_percentage >= 80:
            feedback = "🎉 Excellent work! You have a great understanding of Rigvedic wisdom!"
        elif score_percentage >= 60:
            feedback = "👏 Well done! You're making good progress in your Rigveda studies!"
        elif score_percentage >= 40:
            feedback = "💪 Good effort! Keep exploring and learning more about Rigveda!"
        else:
            feedback = "📚 Don't worry! Learning takes time. Let's continue our Rigveda journey together!"
        
        return jsonify({
            'score': correct_count,
            'total': total_questions,
            'percentage': score_percentage,
            'feedback': feedback,
            'results': results,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error submitting quiz: {e}")
        return jsonify({'error': 'Failed to submit quiz'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if (index is not None and verses is not None) else 'degraded',
        'database_loaded': index is not None and verses is not None,
        'total_verses': len(verses) if verses else 0,
        'active_conversations': len(conversations),
        'veda': 'rigveda'
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting Rigveda Chatbot Server with Quiz Feature...")
    print("📚 Make sure your database files are in ./database/ directory")
    print("🌐 Frontend will be available at http://localhost:5001")
    print("🔌 API available at http://localhost:5001/api/rigveda/ask")
    print("🧠 Quiz API available at http://localhost:5001/api/rigveda/generate-quiz")
    
    app.run(debug=True, host='0.0.0.0', port=5001)