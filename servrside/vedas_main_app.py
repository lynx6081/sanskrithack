from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_cors import CORS
import os
import importlib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Dictionary to store loaded Veda apps
veda_apps = {}

def load_veda_app(veda_name):
    """Dynamically load a Veda app module"""
    try:
        if veda_name not in veda_apps:
            module_name = f"{veda_name}_app"
            veda_apps[veda_name] = importlib.import_module(module_name)
        return veda_apps[veda_name]
    except ImportError as e:
        print(f"Warning: Could not load {veda_name}_app.py - {e}")
        return None

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
            <li><a href="/rigveda">Rigveda Tutor</a></li>
            <li><a href="/samaveda">Samaveda Tutor</a></li>
            <li><a href="/yajurveda">Yajurveda Tutor</a></li>
            <li><a href="/atharvaveda">Atharvaveda Tutor</a></li>
        </ul>
        """

def serve_veda_page(veda_name, icon):
    """Generic function to serve a Veda tutor page"""
    try:
        # Try enhanced version first, then fallback to basic
        try:
            with open(f'templates/{veda_name}/enhanced_index.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            with open(f'templates/{veda_name}/index.html', 'r', encoding='utf-8') as f:
                return f.read()
    except FileNotFoundError:
        return f"""
        <h1>{icon} {veda_name.title()} Tutor</h1>
        <p>{veda_name.title()} tutor files not found!</p>
        <p>Please create the {veda_name.title()} tutor files in <code>templates/{veda_name}/</code></p>
        <p>Expected files: <code>index.html</code> or <code>enhanced_index.html</code></p>
        <p><a href="/">← Back to Vedas Hub</a></p>
        """

@app.route('/rigveda')
def rigveda_tutor():
    """Serve the Rigveda tutor page"""
    return serve_veda_page('rigveda', '🔥')

@app.route('/samaveda')
def samaveda_tutor():
    """Serve the Samaveda tutor page"""
    return serve_veda_page('samaveda', '🎵')

@app.route('/yajurveda')
def yajurveda_tutor():
    """Serve the Yajurveda tutor page"""
    return serve_veda_page('yajurveda', '🔱')

@app.route('/atharvaveda')
def atharvaveda_tutor():
    """Serve the Atharvaveda tutor page"""
    return serve_veda_page('atharvaveda', '🌿')

def create_api_routes(veda_name):
    """Create API routes for a specific Veda"""
    
    @app.route(f'/api/{veda_name}/ask', methods=['POST'], endpoint=f'{veda_name}_ask')
    def veda_api_ask():
        """API endpoint for asking questions"""
        veda_app = load_veda_app(veda_name)
        if not veda_app:
            return jsonify({
                'error': f'{veda_name.title()} tutor temporarily unavailable. Please check server configuration.',
                'details': f'Could not load {veda_name}_app.py'
            }), 500
        
        try:
            data = request.json
            query = data.get('query', '').strip()
            topk = data.get('topk', 5)
            is_intro = data.get('is_intro', False)
            session_id = data.get('session_id', 'default')
            
            # Call the ask function from the specific Veda app
            answer, relevant_verses, quiz_triggered = veda_app.ask(
                query, topk=topk, is_intro=is_intro, session_id=session_id
            )
            
            # Format verse information
            verses_info = []
            if relevant_verses:
                for v in relevant_verses:
                    if isinstance(v, dict):
                        verses_info.append({
                            'mandala': v.get('mandala', ''),
                            'sukta': v.get('sukta', ''),
                            'verse': v.get('verse', ''),
                            'text_sa': v.get('text_sa', '')
                        })
            
            return jsonify({
                'answer': answer,
                'verses': verses_info,
                'query': query,
                'is_intro': is_intro,
                'quiz_triggered': quiz_triggered,
                'session_id': session_id
            })
            
        except Exception as e:
            print(f"Error in {veda_name} API: {e}")
            return jsonify({
                'error': f'{veda_name.title()} tutor encountered an error.',
                'details': str(e)
            }), 500
    
    @app.route(f'/api/{veda_name}/generate-quiz', methods=['POST'], endpoint=f'{veda_name}_generate_quiz')
    def veda_api_generate_quiz():
        """API endpoint for generating quiz"""
        veda_app = load_veda_app(veda_name)
        if not veda_app:
            return jsonify({
                'error': f'{veda_name.title()} quiz temporarily unavailable.',
                'details': f'Could not load {veda_name}_app.py'
            }), 500
        
        try:
            data = request.json
            session_id = data.get('session_id', 'default')
            
            # Check if the veda app has the required functions
            if not hasattr(veda_app, 'conversations'):
                return jsonify({'error': 'No conversation history found'}), 400
            
            if session_id not in veda_app.conversations:
                return jsonify({'error': 'No conversation history found'}), 400
            
            topics = veda_app.extract_topics_from_conversation(veda_app.conversations[session_id])
            if not topics:
                return jsonify({'error': 'No topics found in conversation'}), 400
            
            quiz_data = veda_app.generate_mcq_quiz(topics, veda_app.conversations[session_id])
            if not quiz_data:
                return jsonify({'error': 'Failed to generate quiz'}), 500
            
            return jsonify({
                'quiz': quiz_data,
                'topics': topics,
                'session_id': session_id
            })
            
        except Exception as e:
            print(f"Error generating {veda_name} quiz: {e}")
            return jsonify({
                'error': f'{veda_name.title()} quiz generation failed.',
                'details': str(e)
            }), 500
    
    @app.route(f'/api/{veda_name}/submit-quiz', methods=['POST'], endpoint=f'{veda_name}_submit_quiz')
    def veda_api_submit_quiz():
        """API endpoint for submitting quiz"""
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
                feedback = f"🎉 Excellent work! You have a great understanding of {veda_name.title()} wisdom!"
            elif score_percentage >= 60:
                feedback = f"👏 Well done! You're making good progress in your {veda_name.title()} studies!"
            elif score_percentage >= 40:
                feedback = f"💪 Good effort! Keep exploring and learning more about {veda_name.title()}!"
            else:
                feedback = f"📚 Don't worry! Learning takes time. Let's continue our {veda_name.title()} journey together!"
            
            return jsonify({
                'score': correct_count,
                'total': total_questions,
                'percentage': score_percentage,
                'feedback': feedback,
                'results': results,
                'session_id': session_id
            })
            
        except Exception as e:
            print(f"Error submitting {veda_name} quiz: {e}")
            return jsonify({
                'error': f'{veda_name.title()} quiz submission failed.',
                'details': str(e)
            }), 500

# Create API routes for all Vedas
for veda in ['rigveda', 'samaveda', 'yajurveda', 'atharvaveda']:
    create_api_routes(veda)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check for the main platform"""
    # Check which Vedas are available
    available_vedas = []
    unavailable_vedas = []
    
    for veda in ['rigveda', 'samaveda', 'yajurveda', 'atharvaveda']:
        app_module = load_veda_app(veda)
        if app_module:
            available_vedas.append(veda)
        else:
            unavailable_vedas.append(veda)
    
    return jsonify({
        'status': 'healthy',
        'platform': 'Vedic Wisdom Hub',
        'available_vedas': available_vedas,
        'unavailable_vedas': unavailable_vedas,
        'total_vedas': 4
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
                <li><strong>All Four Vedas</strong> - Complete coverage of Rigveda, Samaveda, Yajurveda, and Atharvaveda</li>
            </ul>
            
            <h2>🎯 Our Mission:</h2>
            <p>To bridge the gap between ancient wisdom and modern learning, making all four Vedas accessible to seekers worldwide through AI-powered tutoring.</p>
            
            <h2>🕉️ The Four Vedas:</h2>
            <ul>
                <li><strong>Rigveda 🔥</strong> - Hymns and praises to deities</li>
                <li><strong>Samaveda 🎵</strong> - Melodies and chants for rituals</li>
                <li><strong>Yajurveda 🔱</strong> - Ritual procedures and mantras</li>
                <li><strong>Atharvaveda 🌿</strong> - Practical wisdom and daily life</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.route('/api/veda-status')
def veda_status():
    """API endpoint to check status of all Vedas"""
    status = {}
    
    for veda in ['rigveda', 'samaveda', 'yajurveda', 'atharvaveda']:
        app_module = load_veda_app(veda)
        template_exists = (
            os.path.exists(f'templates/{veda}/index.html') or 
            os.path.exists(f'templates/{veda}/enhanced_index.html')
        )
        
        status[veda] = {
            'app_available': app_module is not None,
            'template_available': template_exists,
            'fully_functional': app_module is not None and template_exists
        }
    
    return jsonify(status)

if __name__ == '__main__':
    print("🕉️ Starting Vedic Wisdom Hub - All Vedas Platform...")
    print("🌐 Main platform available at http://localhost:5000")
    print("🔥 Rigveda tutor: http://localhost:5000/rigveda")
    print("🎵 Samaveda tutor: http://localhost:5000/samaveda")
    print("🔱 Yajurveda tutor: http://localhost:5000/yajurveda")
    print("🌿 Atharvaveda tutor: http://localhost:5000/atharvaveda")
    print("📌 Health check: http://localhost:5000/api/health")
    print("📊 Status check: http://localhost:5000/api/veda-status")
    
    app.run(debug=True, host='0.0.0.0', port=5000)