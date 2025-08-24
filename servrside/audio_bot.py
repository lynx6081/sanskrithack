import streamlit as st
from gtts import gTTS
import tempfile

st.title("🕉 Sanskrit Text-to-Speech Generator")

st.markdown("Enter any Sanskrit text (Devanagari or IAST) and convert it into speech.")

text_input = st.text_area("📝 Enter Sanskrit text:", "योगः कर्मसु कौशलम्")

if st.button("🔊 Generate Speech"):
    if text_input.strip():
        # Generate temporary audio file
        # tts = gTTS(text=text_input, lang="sa", slow=False)
        tts = gTTS(text=text_input, lang="hi", slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        
        # Play in browser
        audio_file = open(temp_file.name, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
        
        st.success("✅ Audio generated successfully!")
    else:
        st.error("Please enter some Sanskrit text.")