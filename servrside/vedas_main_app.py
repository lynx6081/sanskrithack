from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    """Serve the main Vedas selection page"""
    try:
        with open('templates/vedas_landing.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>🕉️ Vedic Wisdom Hub</h1>
        <p>Welcome to the Vedic Learning Platform!</p>
        <p>Please create <code>templates/vedas_landing.html</code> with the main landing page.</p>
        <ul>
            <li><a href="/rigveda">Rigveda Tutor (Available)</a></li>
            <li><a href="/samaveda">Samaveda Tutor (Coming Soon)</a></li>
            <li><a href="/yajurveda">Yajurveda Tutor (Coming Soon)</a></li>
            <li><a href="/atharvaveda">Atharvaveda Tutor (Coming Soon)</a></li>
        </ul>
        """

@app.route('/rigveda')
def rigveda_tutor():
    """Serve the Rigveda tutor page"""
    try:
        # Try enhanced version first, then fallback to basic
        try:
            with open('templates/rigveda/enhanced_index.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            with open('templates/rigveda/index.html', 'r', encoding='utf-8') as f:
                return f.read()
    except FileNotFoundError:
        return """
        <h1>🔥 Rigveda Tutor</h1>
        <p>Rigveda tutor files not found!</p>
        <p>Please create the Rigveda tutor files in <code>templates/rigveda/</code></p>
        <p><a href="/">← Back to Vedas Hub</a></p>
        """

@app.route('/samaveda')
def samaveda_tutor():
    """Samaveda tutor - Coming soon"""
    return create_coming_soon_page('Samaveda', '🎵', 'melodies and chants')

@app.route('/yajurveda')
def yajurveda_tutor():
    """Yajurveda tutor - Coming soon"""
    return create_coming_soon_page('Yajurveda', '🔱', 'rituals and procedures')

@app.route('/atharvaveda')
def atharvaveda_tutor():
    """Atharvaveda tutor - Coming soon"""
    return create_coming_soon_page('Atharvaveda', '🌿', 'practical wisdom and daily life')

def create_coming_soon_page(veda_name, icon, description):
    """Create a beautiful coming soon page for future Vedas"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{veda_name} Tutor - Coming Soon</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            
            .container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 25px;
                padding: 60px 40px;
                text-align: center;
                max-width: 600px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
                backdrop-filter: blur(15px);
            }}
            
            .icon {{
                font-size: 5em;
                margin-bottom: 30px;
                display: block;
                filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.2));
                animation: bounce 2s infinite;
            }}
            
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-20px); }}
                60% {{ transform: translateY(-10px); }}
            }}
            
            .title {{
                font-size: 3em;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 20px;
            }}
            
            .description {{
                font-size: 1.3em;
                color: #666;
                margin-bottom: 30px;
                line-height: 1.6;
            }}
            
            .status {{
                background: linear-gradient(135deg, #f39c12, #e67e22);
                color: white;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 1.2em;
                font-weight: 600;
                display: inline-block;
                margin-bottom: 30px;
            }}
            
            .back-button {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                text-decoration: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 1.1em;
                font-weight: 600;
                display: inline-block;
                transition: all 0.3s ease;
            }}
            
            .back-button:hover {{
                transform: translateY(-3px);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            }}
            
            .progress {{
                margin: 30px 0;
                color: #667eea;
                font-weight: 600;
            }}
            
            .progress-bar {{
                width: 100%;
                height: 8px;
                background: #e0e6ed;
                border-radius: 4px;
                margin: 15px 0;
                overflow: hidden;
            }}
            
            .progress-fill {{
                height: 100%;
                background: linear-gradient(135deg, #f39c12, #e67e22);
                border-radius: 4px;
                animation: loading 3s ease-in-out infinite;
            }}
            
            @keyframes loading {{
                0% {{ width: 0%; }}
                50% {{ width: 70%; }}
                100% {{ width: 0%; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">{icon}</div>
            <h1 class="title">{veda_name}</h1>
            <p class="description">
                Discover the wisdom of {veda_name}, the sacred Veda of {description}.
                Our AI-powered tutor is being carefully crafted to bring you an immersive learning experience.
            </p>
            
            <div class="status">🚧 Coming Soon</div>
            
            <div class="progress">
                <div>Development in Progress...</div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
            
            <a href="/" class="back-button">← Back to Vedas Hub</a>
        </div>
    </body>
    </html>
    """

# API Routes for Rigveda (proxy to existing Rigveda app)
@app.route('/api/rigveda/ask', methods=['POST'])
def rigveda_api_ask():
    """Proxy API calls to Rigveda tutor"""
    # Import and use your existing Rigveda logic here
    # For now, return a placeholder response
    try:
        from rigveda_app import ask as rigveda_ask
        data = request.json
        query = data.get('query', '').strip()
        topk = data.get('topk', 5)
        is_intro = data.get('is_intro', False)
        session_id = data.get('session_id', 'default')
        
        answer, relevant_verses, quiz_triggered = rigveda_ask(
            query, topk=topk, is_intro=is_intro, session_id=session_id
        )
        
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
        # Fallback response if Rigveda app is not available
        return jsonify({
            'error': 'Rigveda tutor temporarily unavailable. Please check server configuration.',
            'details': str(e)
        }), 500

@app.route('/api/rigveda/generate-quiz', methods=['POST'])
def rigveda_api_generate_quiz():
    """Proxy quiz generation to Rigveda tutor"""
    try:
        from rigveda_app import generate_mcq_quiz, extract_topics_from_conversation, conversations
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in conversations:
            return jsonify({'error': 'No conversation history found'}), 400
        
        topics = extract_topics_from_conversation(conversations[session_id])
        if not topics:
            return jsonify({'error': 'No topics found in conversation'}), 400
        
        quiz_data = generate_mcq_quiz(topics, conversations[session_id])
        if not quiz_data:
            return jsonify({'error': 'Failed to generate quiz'}), 500
        
        return jsonify({
            'quiz': quiz_data,
            'topics': topics,
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Quiz generation temporarily unavailable.',
            'details': str(e)
        }), 500

@app.route('/api/rigveda/submit-quiz', methods=['POST'])
def rigveda_api_submit_quiz():
    """Proxy quiz submission to Rigveda tutor"""
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
        
        # Generate feedback
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
        return jsonify({
            'error': 'Quiz submission failed.',
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check for the main platform"""
    return jsonify({
        'status': 'healthy',
        'platform': 'Vedic Wisdom Hub',
        'available_vedas': ['rigveda'],
        'coming_soon': ['samaveda', 'yajurveda', 'atharvaveda']
    })

@app.route('/about')
def about():
    """About page for the platform"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>About - Vedic Wisdom Hub</title>
        <style>
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                margin: 0;
            }
            .container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 25px;
                padding: 40px;
                max-width: 800px;
                backdrop-filter: blur(15px);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
            }
            h1 { color: #2c3e50; margin-bottom: 30px; }
            .back-link {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 20px;
                display: inline-block;
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-link">← Back to Home</a>
            <h1>🕉️ About Vedic Wisdom Hub</h1>
            <p>Welcome to the future of Vedic learning! Our platform uses advanced AI technology to make ancient Sanskrit texts accessible to modern learners.</p>
            
            <h2>📚 What We Offer:</h2>
            <ul>
                <li><strong>Interactive AI Tutors</strong> - Personalized guidance through ancient texts</li>
                <li><strong>Smart Quizzes</strong> - Test your knowledge with adaptive questions</li>
                <li><strong>Modern Interface</strong> - Beautiful, user-friendly design</li>
                <li><strong>Authentic Content</strong> - Based on original Sanskrit sources</li>
            </ul>
            
            <h2>🎯 Our Mission:</h2>
            <p>To bridge the gap between ancient wisdom and modern learning, making Vedic knowledge accessible to seekers worldwide.</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("🕉️ Starting Vedic Wisdom Hub...")
    print("🌐 Main platform available at http://localhost:5000")
    print("🔥 Rigveda tutor available at http://localhost:5000/rigveda")
    print("🎵 Other Vedas coming soon!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)