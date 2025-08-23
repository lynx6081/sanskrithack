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
        # Fixed paths with multiple fallbacks
        possible_paths = [
            ("./database/yajurveda.index", "./database/yajurveda_meta.pkl"),
            ("./databse/yajurveda.index", "./databse/yajurveda_meta.pkl"),  # Keep typo version as fallback
            ("database/yajurveda.index", "database/yajurveda_meta.pkl"),
            ("databse/yajurveda.index", "databse/yajurveda_meta.pkl")
        ]
        
        for index_path, meta_path in possible_paths:
            try:
                if os.path.exists(index_path) and os.path.exists(meta_path):
                    index = faiss.read_index(index_path)
                    with open(meta_path, "rb") as f:
                        verses = pickle.load(f)
                    print(f"✅ Yajurveda index and metadata loaded successfully from {index_path}")
                    return True
            except Exception as e:
                continue
        
        print("⚠ Could not find database files in any expected location")
        return False
    except Exception as e:
        print(f"⚠ Error loading data: {e}")
        return False

def embed(texts):
    """Create embeddings using OpenAI"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return np.array([d.embedding for d in response.data], dtype="float32")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        # Fallback with dummy embeddings for testing
        return np.random.rand(len(texts) if isinstance(texts, list) else 1, 3072).astype("float32")

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
        return ["ritual procedures", "sacred mantras", "fire ceremonies"]  # Default topics
    
    # Get recent conversation for topic extraction
    recent_messages = conversation_history[-8:]  # Last 8 messages for better context
    context = "\n".join([
        f"{'User' if msg['sender'] == 'user' else 'Tutor'}: {msg['text']}" 
        for msg in recent_messages
    ])
    
    topic_extraction_prompt = f"""
Based on this Yajurveda tutoring conversation, identify the main topics and concepts discussed. 
Focus on specific Yajurvedic themes, rituals, sacrificial procedures, mantras, or ceremonial practices mentioned.

Conversation:
{context}

List the main topics as a simple comma-separated list (max 5 topics). Examples:
sacrificial rituals, mantras, fire ceremonies, ritual procedures, priests, altar construction, offerings, ceremonial implements, yajna

If no specific topics are clear, use: ritual procedures, sacred mantras, fire ceremonies

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
        
        # Ensure we always return at least one topic
        if not topics:
            topics = ["ritual procedures", "sacred mantras", "fire ceremonies"]
            
        return topics[:5]  # Max 5 topics
        
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return ["ritual procedures", "sacred mantras", "fire ceremonies"]  # Return default topics on error

def generate_mcq_quiz(topics, conversation_context):
    """Generate MCQ quiz based on discussed topics"""
    if not topics:
        topics = ["ritual procedures", "sacred mantras", "fire ceremonies"]
    
    topics_str = ", ".join(topics)
    
    quiz_prompt = f"""
You are creating a quiz for a student who has been learning about Yajurveda. Based on the topics they discussed: {topics_str}

Create exactly 3 multiple choice questions (easy to moderate level) about these Yajurvedic topics. 
Each question should have exactly 4 options (A, B, C, D) with exactly one correct answer.

Format your response as valid JSON only, no other text:
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
    }},
    {{
      "question": "Second question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "B",
      "explanation": "Brief explanation of why this answer is correct"
    }},
    {{
      "question": "Third question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "C",
      "explanation": "Brief explanation of why this answer is correct"
    }}
  ]
}}

Focus on:
- Sacrificial rituals and procedures
- Types of offerings and ceremonies
- Priestly duties and roles
- Ritual implements and altar construction
- Mantras and their applications
- Different types of yajna (fire sacrifices)

Keep questions clear and educational. Avoid overly complex Sanskrit terms without explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": quiz_prompt}],
            temperature=0.7,
            max_tokens=1200
        )
        
        quiz_text = response.choices[0].message.content.strip()
        
        # Clean up the response - remove any markdown formatting
        if quiz_text.startswith('```json'):
            quiz_text = quiz_text.replace('```json', '').replace('```', '').strip()
        elif quiz_text.startswith('```'):
            quiz_text = quiz_text.replace('```', '').strip()
        
        # Find the JSON part if there's extra text
        json_start = quiz_text.find('{')
        json_end = quiz_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            quiz_text = quiz_text[json_start:json_end]
            
        quiz_data = json.loads(quiz_text)
        
        # Validate the structure
        if not isinstance(quiz_data, dict) or 'questions' not in quiz_data:
            raise ValueError("Invalid quiz structure")
        
        if not isinstance(quiz_data['questions'], list) or len(quiz_data['questions']) == 0:
            raise ValueError("No questions in quiz")
            
        return quiz_data
        
    except Exception as e:
        print(f"Error generating quiz: {e}")
        # Return a fallback quiz
        return {
            "questions": [
                {
                    "question": "What is the primary focus of Yajurveda?",
                    "options": {
                        "A": "Philosophical discussions",
                        "B": "Ritual procedures and sacrificial ceremonies",
                        "C": "Historical narratives",
                        "D": "Poetic compositions"
                    },
                    "correct_answer": "B",
                    "explanation": "Yajurveda primarily contains mantras and procedures for conducting sacrificial rituals and ceremonies."
                },
                {
                    "question": "What does 'yajna' refer to in Yajurveda?",
                    "options": {
                        "A": "Sacred texts",
                        "B": "Fire sacrifice or ritual offering",
                        "C": "Temple architecture",
                        "D": "Meditation practice"
                    },
                    "correct_answer": "B",
                    "explanation": "Yajna refers to fire sacrifices or ritual offerings, which are central to Yajurvedic practices."
                },
                {
                    "question": "Who typically performed Yajurvedic rituals?",
                    "options": {
                        "A": "Common people",
                        "B": "Kings only",
                        "C": "Trained priests",
                        "D": "Merchants"
                    },
                    "correct_answer": "C",
                    "explanation": "Yajurvedic rituals were typically performed by trained priests who knew the precise procedures and mantras."
                }
            ]
        }

def should_trigger_quiz(session_id):
    """Check if quiz should be triggered based on conversation count"""
    if session_id not in user_quiz_states:
        user_quiz_states[session_id] = {
            'message_count': 0,
            'last_quiz_at': 0,
            'quiz_frequency': 3  # Every 3 meaningful exchanges (matching Rigveda)
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
        # Special introduction message for Yajurveda
        intro_response = """
🔥 **Namaste and welcome, dear seeker of ritual wisdom!** 🔥

I am your passionate Yajurveda tutor, and I'm absolutely thrilled that you've chosen to explore the most practical and ritualistic of all Vedas! The Yajurveda is like a comprehensive manual for sacred ceremonies, containing the precise procedures, mantras, and instructions that ancient priests used to conduct powerful fire sacrifices and religious rituals.

Think of me as your knowledgeable guide who will help you understand these sacred procedures, ritual significance, and ceremonial practices - whether you're interested in ancient traditions, spiritual practices, or religious studies!

**What ritual journey shall we begin today?** Here are some fascinating topics we could explore:

🔥 **Fire Sacrifices** - Learn about yajna ceremonies and their spiritual significance
📜 **Ritual Procedures** - Discover the precise steps in Vedic ceremonies  
🛕 **Altar Construction** - Understand the sacred geometry of ritual spaces
🕉️ **Sacred Mantras** - Explore the powerful chants used in rituals
👨‍🔬 **Priestly Duties** - Meet the different types of ritual specialists
🥄 **Offerings & Implements** - Learn about ceremonial tools and substances
⚡ **Different Yajnas** - Discover various types of fire ceremonies
🌟 **Ritual Symbolism** - Understand the deeper meanings behind ceremonies

Just click on any topic above, or ask me anything that sparks your curiosity about Vedic rituals! I love questions like "How were fire altars built?" or "What makes Yajurveda different from other Vedas?" - no question is too simple or too complex!

**What sacred ceremony shall we discover together today?** 🌟
"""
        return intro_response, [], False
    
    # Initialize conversation tracking
    if session_id and session_id not in conversations:
        conversations[session_id] = []
    
    # Handle case when database is not loaded
    if index is None or verses is None:
        # Provide a general response without database search
        context = "General Yajurveda knowledge without specific verse references"
        results = []
    else:
        # 1. Retrieve relevant verses
        try:
            results = search(query, topk=topk)
            # 2. Create context
            context = "\n".join([
                f"YV {r.get('chapter', '?')}.{r.get('verse', '?')}: {r.get('text_sa', r.get('text', ''))}"
                for r, _ in results
            ])
        except Exception as e:
            print(f"Search error: {e}")
            results = []
            context = "General Yajurveda knowledge (database search unavailable)"
    
    # 3. Create engaging tutor prompt for Yajurveda
    prompt = f"""
You are a passionate and enthusiastic Yajurveda tutor who absolutely loves teaching about sacred rituals and ceremonial practices. Your goal is to make Yajurvedic knowledge accessible and exciting for everyone - from spiritual seekers to students of ancient traditions.

Your personality:
- Warm, reverent, and encouraging (like a wise ritual expert)
- Uses simple language but maintains depth about ritual and ceremonial concepts
- Gives examples and analogies related to ceremonies and sacred practices
- Shows genuine excitement about sharing Yajurvedic ritual wisdom
- Explains Sanskrit terms and ritual concepts when used
- Connects ancient ceremonial traditions to spirituality and cultural practices today

Based on the following context about Yajurveda, provide a clear, engaging answer that:
1. Keep your response concise (2-3 short paragraphs maximum)
2. Explain the topic clearly with focus on ritual/ceremonial aspects when relevant
3. Use ceremonial analogies or examples when helpful
4. Share the spiritual and cultural significance briefly
5. End with 2-3 short follow-up questions in this EXACT format:
   **Follow-up Questions:**
   • Question 1?
   • Question 2? 
   • Question 3?

Keep follow-up questions short (5-7 words each) so they work well as buttons.

Student's Question: {query}

Relevant Context:
{context}

Provide your enthusiastic, knowledgeable response:
"""
    
    # 4. Get response from OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        # Fallback response
        answer = f"""
I understand you're asking about "{query}" in relation to Yajurveda! While I'm experiencing some technical difficulties accessing my full knowledge base, I can share that Yajurveda is fundamentally about ritual procedures and sacred ceremonies.

The Yajurveda contains detailed instructions for conducting fire sacrifices (yajna), altar construction, and ceremonial practices that were central to ancient Vedic traditions. These rituals were believed to maintain cosmic order and facilitate communication between humans and the divine.

**Follow-up Questions:**
• What are different types of yajna?
• How were altars constructed?
• What role did priests play?
"""
    
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
    
    # Return results (handle case when results might not exist)
    return answer, [r for r, _ in results], quiz_triggered

# Load data when the app starts
if not load_data():
    print("Warning: Could not load data files. App will run with limited functionality.")

@app.route('/')
def home():
    """Serve the frontend"""
    try:
        # Updated path checking like Rigveda app
        possible_html_paths = [
            'templates/yajurveda/index.html',
            'templates/yajurveda_index.html',
            'templates/index.html',
            'yajurveda/index.html',
            'index.html'
        ]
        
        for html_path in possible_html_paths:
            try:
                if os.path.exists(html_path):
                    with open(html_path, 'r', encoding='utf-8') as f:
                        print(f"✅ Serving HTML from: {html_path}")
                        return f.read()
            except Exception as e:
                continue
                
        raise FileNotFoundError("HTML file not found in any expected location")
        
    except FileNotFoundError:
        return """
        <h1>🔥 Yajurveda Chatbot Backend with Quiz Feature</h1>
        <p>Backend is running! Frontend HTML file not found.</p>
        <p>Expected locations checked:</p>
        <ul>
            <li>templates/yajurveda/index.html</li>
            <li>templates/yajurveda_index.html</li>
            <li>templates/index.html</li>
            <li>yajurveda/index.html</li>
            <li>index.html</li>
        </ul>
        <p>API endpoints available:</p>
        <ul>
            <li>POST /api/yajurveda/ask - For chat messages</li>
            <li>POST /api/yajurveda/generate-quiz - For quiz generation</li>
            <li>POST /api/yajurveda/submit-quiz - For quiz submission</li>
            <li>GET /api/health - Health check</li>
        </ul>
        <p>Current working directory: """ + os.getcwd() + """</p>
        """

# Fixed API endpoints to match the Rigveda pattern exactly
@app.route('/api/yajurveda/ask', methods=['POST'])
def api_ask():
    """API endpoint for asking questions"""
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        data = request.json
        query = data.get('query', '').strip()
        topk = data.get('topk', 5)
        is_intro = data.get('is_intro', False)
        session_id = data.get('session_id', 'default')
        
        print(f"✅ Received request - Query: '{query}', Intro: {is_intro}, Session: {session_id}")
        
        if not query and not is_intro:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get answer and check for quiz trigger (works even without database)
        answer, relevant_verses, quiz_triggered = ask(query, topk=topk, is_intro=is_intro, session_id=session_id)
        
        # Format verse references for frontend
        verses_info = []
        for v in relevant_verses:
            verse_info = {
                'chapter': v.get('chapter', '?'),
                'verse': v.get('verse', '?'),
                'text_sa': v.get('text_sa', v.get('text', ''))
            }
            verses_info.append(verse_info)
        
        return jsonify({
            'answer': answer,
            'verses': verses_info,
            'query': query,
            'is_intro': is_intro,
            'quiz_triggered': quiz_triggered,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"❌ Error in API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/yajurveda/generate-quiz', methods=['POST'])
def api_generate_quiz():
    """API endpoint for generating quiz - Fixed to match Rigveda pattern"""
    try:
        print("✅ Generate quiz endpoint called")
        
        # Validate request
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        session_id = data.get('session_id', 'default')
        
        print(f"🎯 Generating quiz for session: {session_id}")
        print(f"📊 Available sessions: {list(conversations.keys())}")
        
        # Check if session exists, if not create default
        if session_id not in conversations:
            print("⚠ Creating default conversation for quiz")
            conversations[session_id] = [
                {
                    'timestamp': datetime.now().isoformat(),
                    'sender': 'user',
                    'text': 'Tell me about Yajurveda rituals'
                },
                {
                    'timestamp': datetime.now().isoformat(),
                    'sender': 'bot',
                    'text': 'Yajurveda contains detailed procedures for conducting fire sacrifices and ceremonial practices.'
                }
            ]
        
        conversation_history = conversations[session_id]
        print(f"💬 Conversation length: {len(conversation_history)}")
        
        # Extract topics from conversation
        topics = extract_topics_from_conversation(conversation_history)
        print(f"🏷️ Extracted topics: {topics}")
        
        if not topics:
            topics = ["ritual procedures", "sacred mantras", "fire ceremonies"]
        
        # Generate quiz
        print("🔥 Generating quiz...")
        quiz_data = generate_mcq_quiz(topics, conversation_history)
        
        if not quiz_data:
            return jsonify({'error': 'Failed to generate quiz'}), 500
        
        print(f"🎉 Quiz successfully created with {len(quiz_data.get('questions', []))} questions")
        
        return jsonify({
            'quiz': quiz_data,
            'topics': topics,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"💥 Error generating quiz: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate quiz: {str(e)}'}), 500

@app.route('/api/yajurveda/submit-quiz', methods=['POST'])
def api_submit_quiz():
    """API endpoint for submitting quiz answers - Fixed to match Rigveda pattern"""
    try:
        print("✅ Submit quiz endpoint called")
        
        # Validate request has JSON data
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        session_id = data.get('session_id', 'default')
        answers = data.get('answers', {})
        quiz_questions = data.get('quiz_questions', [])
        
        # Validate required data
        if not quiz_questions:
            return jsonify({'error': 'No quiz questions provided'}), 400
        
        if not isinstance(quiz_questions, list):
            return jsonify({'error': 'Quiz questions must be a list'}), 400
        
        if not isinstance(answers, dict):
            return jsonify({'error': 'Answers must be a dictionary'}), 400
        
        print(f"📝 Processing quiz submission for session: {session_id}")
        print(f"❓ Number of questions: {len(quiz_questions)}")
        print(f"✏️ Number of answers: {len(answers)}")
        
        # Calculate score
        correct_count = 0
        results = []
        
        for i, question in enumerate(quiz_questions):
            question_id = str(i)
            user_answer = answers.get(question_id)
            
            # Validate question structure
            if not isinstance(question, dict):
                return jsonify({'error': f'Question {i} has invalid format'}), 400
            
            if 'correct_answer' not in question:
                return jsonify({'error': f'Question {i} missing correct_answer'}), 400
            
            correct_answer = question['correct_answer']
            is_correct = user_answer == correct_answer
            
            if is_correct:
                correct_count += 1
            
            results.append({
                'question': question.get('question', f'Question {i+1}'),
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'explanation': question.get('explanation', 'No explanation provided')
            })
        
        total_questions = len(quiz_questions)
        score_percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # Generate encouraging feedback
        if score_percentage >= 80:
            feedback = "🔥 Excellent mastery! You understand the sacred rituals of Yajurveda beautifully!"
        elif score_percentage >= 60:
            feedback = "📜 Well performed! Your knowledge of Yajurvedic traditions is growing steadily!"
        elif score_percentage >= 40:
            feedback = "🛕 Good foundation! Keep studying the ritual wisdom of Yajurveda!"
        else:
            feedback = "🕉️ Every ritual expert begins as a student! Let's continue our ceremonial journey together!"
        
        print(f"🏆 Quiz completed: {correct_count}/{total_questions} ({score_percentage:.1f}%)")
        
        return jsonify({
            'score': correct_count,
            'total': total_questions,
            'percentage': round(score_percentage, 1),
            'feedback': feedback,
            'results': results,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"❌ Error submitting quiz: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to submit quiz: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if (index is not None and verses is not None) else 'degraded',
        'database_loaded': index is not None and verses is not None,
        'total_verses': len(verses) if verses else 0,
        'active_conversations': len(conversations),
        'veda_type': 'yajurveda',
        'working_directory': os.getcwd(),
        'available_routes': [
            'GET /',
            'POST /api/yajurveda/ask', 
            'POST /api/yajurveda/generate-quiz',
            'POST /api/yajurveda/submit-quiz',
            'GET /api/health'
        ]
    }
    print(f"🩺 Health check: {status['status']}")
    return jsonify(status)

# Route debugging function
@app.before_request
def log_request_info():
    print(f"🌐 {request.method} {request.path}")

if __name__ == '__main__':
    print("🚀 Starting Enhanced Yajurveda Chatbot Server with Quiz Feature...")
    print("🔥 Make sure your database files are in ./database/ directory")
    print("🌐 Frontend will be available at http://localhost:5000")
    print("🔌 API available at:")
    print("   - GET / (Frontend)")
    print("   - POST /api/yajurveda/ask")
    print("   - POST /api/yajurveda/generate-quiz")
    print("   - POST /api/yajurveda/submit-quiz")
    print("   - GET /api/health")
    
    # Print registered routes for debugging
    print("\n📋 Registered Flask Routes:")
    for rule in app.url_map.iter_rules():
        print(f"   {rule.methods} {rule.rule}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)