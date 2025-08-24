# sanskrithack
**About** <br>
We build AI powered Vedic Wisdom Hub that contains specialised chatbot as tutors for specific vedas. We build an interactive platform with good user interface where the user can select a particular Veda and then start their learning journey from scratch. The main motive of this Vedic wisdom Hub is to make learning veda simple and interactive for everyone. With this platform, you don't need to go through the Vedic Text just to get any sort of information about the vedas, just one click and enthusiasm is enough to get started. <br> <br>
**Features**
- Platform is build for any type of user from beginner to advanced.
- A good, simple, easy to understand UI for anyone to get started.
- Specialised Tutors for specific Vedas. These tutor will guide you throughout your learning journey.
- Real time suggested interactive follow up questions based on the user query so that user can dive deeper.
- Real time generated small quizzes after every 2-3 conversations to test how much the user understood with proper scoring and corrections
- Easy to understand yet detailed answer to the query by the AI tutor.
<br><br>

**Tech Stack**
-  Frontend: html, css, javascript, streamlit
-  Backend: flask
-  database: faiss (vectorDB)
-  APIs: openai, bs4
-  technology- Retrieval Augmented Generation

**Getting Started:** <br>
**1. Clone this repo** <br>
git clone https://github.com/lynx6081/sanskrithack.git<br>
cd sanskrithack <br>
**2. Install dependencies** <br>
pip install -r requirements.txt <br>
**3. Load api key:** <br>
create file with name .env <br>
upload openai_api_key with format: OPENAI_API_KEY = "*your api key here in double quotes*" <br>
**4. Change directory to servrside** <br>
cd servrside <br>
**5. Run the project** <br>
python vedas_main_app.py <br>
**6. Local Host link:** <br>
(ctrl + click) on the url that will appear in the terminal here: 🌐 Main platform available at *http://localhost:5000* <br>
**7. for Audio Bot**<br>
*Go to root directory:* cd .. <br>
*run the file:* streamlit run audio_bot.py <br>

**Usage:** 
When you will run the vedas_main_app.py file, and then ctrl+click on the url, the main page will open.you can select whatever tutor you want and start interaction. You can either click on already provided topics or ask quuery of your own. <br>

**Demo:**
- Live Demo:https://youtu.be/YbnblXgDScI?feature=shared

**Future Improvements:** <br>
- We will try to integrate other sanskrit data such as other literature texts, grammar texts, mythology texts, etc
- We will try to make understand and communicate real time in sanskrit with the user so that the user can learn sanskrit by practically speaking and learning by recommended corrections system.
- We tried to integrate audio to the tutors so that they can recite the texts many times to make users more comfortable with the language.The Audio Bot contains the file to initate a bot where we can input any sanskrit text in sa script and it will give its proper text to speech version in audio file. But even after trying several times, we were unable to integrate the audio due to time shortage.










  

  

  
