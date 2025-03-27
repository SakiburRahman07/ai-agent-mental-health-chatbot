import streamlit as st
from mental_health_assistant import chat_with_mental_health_assistant
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
import re
import cv2
import numpy as np
from deepface import DeepFace
import time
from PIL import Image
import io
import datetime
import pandas as pd
import altair as alt
import os
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Bangladesh Mental Health Support Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None

if "emotion_score" not in st.session_state:
    st.session_state.emotion_score = None

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "emotion_notes" not in st.session_state:
    st.session_state.emotion_notes = {}

if "voice_emotion" not in st.session_state:
    st.session_state.voice_emotion = None

if "voice_confidence" not in st.session_state:
    st.session_state.voice_confidence = None

if "temp_audio_file" not in st.session_state:
    st.session_state.temp_audio_file = None

# Create a mock model for voice emotion detection (since we don't have a real model)
def create_mock_voice_model():
    """Create a simple mock model for demonstration purposes"""
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, 'voice_emotion_model.pkl')
    
    if not os.path.exists(model_path):
        # Map of voice features to emotions - this is just a placeholder
        voice_emotions = {
            'calm': [0.1, 0.2, 0.3, 0.4],
            'happy': [0.5, 0.6, 0.7, 0.8],
            'sad': [0.9, 1.0, 1.1, 1.2],
            'angry': [1.3, 1.4, 1.5, 1.6],
            'fearful': [1.7, 1.8, 1.9, 2.0],
            'neutral': [2.1, 2.2, 2.3, 2.4]
        }
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(voice_emotions, f)
    
    return model_path

# Voice recording and analysis functions
def record_audio(duration=5, sample_rate=22050):
    """Record audio for a specified duration"""
    st.write("Recording... Please speak clearly.")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Recording logic
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    
    # Update progress bar
    for i in range(100):
        time.sleep(duration/100)
        progress_bar.progress(i + 1)
    
    sd.wait()
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'voice_analysis_temp.wav')
    write(temp_file, sample_rate, recording)
    
    st.session_state.temp_audio_file = temp_file
    return temp_file

def analyze_voice(file_path):
    """
    Analyze voice to detect emotion
    This is a simplified version that would work with a real model
    """
    # For demonstration, we'll return a random emotion with a confidence score
    # In a real implementation, this would use the actual model to predict
    
    emotions = ['happy', 'sad', 'angry', 'fearful', 'neutral', 'calm']
    import random
    
    # Load audio and extract features (simplified)
    try:
        y, sr = librosa.load(file_path, duration=5)
        
        # Extract some basic audio features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        
        # In a real implementation, you would use your model here
        # model = pickle.load(open('voice_emotion_model.pkl', 'rb'))
        # prediction = model.predict([mfcc_processed])
        
        # For demonstration, randomly select an emotion
        prediction = random.choice(emotions)
        confidence = random.uniform(60, 95)
        
        # Add to audio features to history
        timestamp = datetime.datetime.now()
        
        # In a real implementation, you'd use the model's confidence
        return prediction, confidence
        
    except Exception as e:
        st.error(f"Error analyzing voice: {str(e)}")
        return "neutral", 50.0

# Voice emotion descriptions and recommendations
VOICE_EMOTION_INSIGHTS = {
    "happy": {
        "description": "A happy tone in your voice can indicate positive emotional state and well-being.",
        "connection": "In Bangladesh, expressing happiness in your voice is generally culturally appropriate and can strengthen social bonds.",
        "recommendations": [
            "Notice what made you feel happy and try to incorporate more of it in your life",
            "Share this positive feeling with others who might benefit from your support",
            "Use this moment of positivity to address any challenging tasks"
        ]
    },
    "sad": {
        "description": "Sadness in your voice can reflect grief, disappointment, or low mood states.",
        "connection": "In Bangladesh, particularly in rural areas, expressing sadness verbally may be seen as seeking community support.",
        "recommendations": [
            "Allow yourself to experience this emotion without judgment",
            "Reach out to trusted family members or friends",
            "Consider gentle physical activities like walking to help process feelings",
            "If persistent, speaking with a mental health professional can help"
        ]
    },
    "angry": {
        "description": "An angry tone can reflect frustration, perceived injustice, or unmet needs.",
        "connection": "In Bangladeshi culture, anger expression is often moderated, especially in public settings.",
        "recommendations": [
            "Take deep breaths before responding to situations",
            "Identify the source of your anger and whether it relates to deeper issues",
            "Find appropriate outlets for expressing feelings, like writing or physical activity",
            "Consider culturally-appropriate ways to address concerns constructively"
        ]
    },
    "fearful": {
        "description": "Fear in your voice can indicate anxiety, worry, or feeling threatened.",
        "connection": "In Bangladesh, anxiety about social perception and family matters may be reflected in voice patterns.",
        "recommendations": [
            "Practice grounding techniques to reduce immediate anxiety",
            "Identify specific triggers for your fear",
            "Share your concerns with someone you trust",
            "Break down overwhelming situations into manageable steps"
        ]
    },
    "neutral": {
        "description": "A neutral tone may indicate emotional balance or sometimes emotional suppression.",
        "connection": "In professional and formal settings in Bangladesh, a neutral tone is often valued.",
        "recommendations": [
            "Check if you're actually feeling neutral or suppressing emotions",
            "Practice mindfulness to maintain awareness of your emotional state",
            "Consider if this is your typical speaking pattern or a response to current circumstances"
        ]
    },
    "calm": {
        "description": "A calm voice tone suggests emotional regulation and composure.",
        "connection": "In Bangladesh, a calm demeanor in speech is highly valued, especially among elders and leaders.",
        "recommendations": [
            "Notice what helps you maintain this state of calm",
            "Use this composure to address any challenges effectively",
            "Share techniques that help you maintain calm with others who might benefit"
        ]
    }
}

# Mental health insights for different emotions in Bangladesh context
EMOTION_MENTAL_HEALTH_INSIGHTS = {
    "happy": {
        "description": "Happiness can be a sign of good mental wellbeing, but sometimes it may mask underlying issues.",
        "connection": "In Bangladesh, expressing happiness openly is culturally encouraged in many situations, but it's also important to acknowledge all emotions.",
        "recommendations": [
            "Take time to appreciate positive moments",
            "Consider writing down what's making you happy to revisit during challenging times",
            "Check if your happiness feels genuine or if you're suppressing other feelings"
        ],
        "local_resources": "Community gatherings and family events can be good support systems to maintain positive emotions."
    },
    "sad": {
        "description": "Sadness is a natural emotion and can indicate grief, loss, or depression if persistent.",
        "connection": "In Bangladesh, sadness may sometimes be internalized due to cultural expectations of emotional resilience, especially in rural areas.",
        "recommendations": [
            "Allow yourself to experience sadness without judgment",
            "Consider speaking with a trusted elder or family member",
            "Engage in community or religious activities that provide comfort",
            "If sadness persists for more than two weeks, consider speaking with a mental health professional"
        ],
        "local_resources": "Kaan Pete Roi's emotional support line provides confidential support in Bangla."
    },
    "angry": {
        "description": "Anger can be a response to perceived injustice, frustration, or unmet needs.",
        "connection": "In Bangladeshi culture, managing anger appropriately is highly valued, but suppressed anger can lead to mental health challenges.",
        "recommendations": [
            "Practice deep breathing for 5 minutes",
            "Write down what triggered your anger",
            "Consider cultural practices like taking a short walk or reciting calming prayers",
            "Find a private space to express frustration safely"
        ],
        "local_resources": "Local community mediators (such as village elders) can sometimes help resolve interpersonal conflicts."
    },
    "fear": {
        "description": "Fear is a protective emotion but can develop into anxiety disorders if persistent.",
        "connection": "In Bangladesh, fears related to natural disasters, economic insecurity, or social judgment are common stressors.",
        "recommendations": [
            "Practice grounding techniques using the 5-4-3-2-1 method",
            "Talk about your fears with someone you trust",
            "Consider how realistic your fears are and what evidence supports or contradicts them",
            "Gradually face minor fears in a controlled way"
        ],
        "local_resources": "The National Institute of Mental Health (NIMH) in Bangladesh provides services for anxiety disorders."
    },
    "surprise": {
        "description": "Surprise indicates something unexpected and can trigger stress responses if startling.",
        "connection": "In fast-changing Bangladeshi urban environments, constant surprises can sometimes contribute to adjustment stress.",
        "recommendations": [
            "Take a moment to process unexpected information",
            "Consider if the surprise has triggered any other emotions",
            "Practice adaptability through mindful acceptance"
        ],
        "local_resources": "Community support groups can help those adjusting to major life changes."
    },
    "neutral": {
        "description": "Neutral expressions may indicate emotional balance or sometimes emotional suppression.",
        "connection": "In Bangladesh, maintaining neutrality might be a cultural value in certain contexts, especially in professional settings.",
        "recommendations": [
            "Check in with yourself about what you're actually feeling",
            "Consider if you're suppressing emotions for cultural reasons",
            "Practice mindfulness to increase emotional awareness"
        ],
        "local_resources": "Meditation groups in urban areas like Dhaka can help with emotional awareness."
    },
    "disgust": {
        "description": "Disgust can relate to moral judgments, traumatic memories, or physical aversion.",
        "connection": "In Bangladesh, disgust related to environmental conditions or certain social situations may be common stressors.",
        "recommendations": [
            "Identify exactly what's triggering the disgust response",
            "Consider if this relates to any past experiences",
            "For environmental triggers, focus on what's within your control to change"
        ],
        "local_resources": "Environmental improvement community groups can help address some common disgust triggers."
    }
}

# Title and description
st.title("🧠 Bangladesh Mental Health Support Assistant")
st.markdown("""
This assistant is here to provide mental health support and information for people in Bangladesh. 
It's designed to be empathetic and helpful, but remember it's not a replacement for professional help.
""")

# Function to display YouTube videos
def display_youtube_video(url):
    # Extract video ID from URL
    video_id_match = re.search(r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\s]+)', url)
    if video_id_match:
        video_id = video_id_match.group(1)
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.markdown(f"[Watch Video]({url})")

# Update the styling for the reasoning/thinking display
def display_thinking(content):
    """Display thinking process with typewriter-like styling"""
    st.markdown("""
    <style>
    .thinking-box {
        background-color: #f9f9ff;
        border-left: 4px solid #6e7bff;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9em;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="thinking-box">{content}</div>', unsafe_allow_html=True)

# Save emotion to history
def save_emotion_to_history(emotion, score, note="", source="face"):
    if emotion:
        timestamp = datetime.datetime.now()
        emotion_data = {
            "timestamp": timestamp,
            "emotion": emotion,
            "score": score,
            "note": note,
            "source": source  # 'face' or 'voice'
        }
        st.session_state.emotion_history.append(emotion_data)
        
        # Keep only the last 30 entries
        if len(st.session_state.emotion_history) > 30:
            st.session_state.emotion_history = st.session_state.emotion_history[-30:]

# Facial emotion detection function
def detect_face_emotion(image):
    try:
        # Convert the image to numpy array for DeepFace
        img_array = np.array(image)
        
        # Analyze the image with DeepFace
        result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
        
        if result and len(result) > 0:
            dominant_emotion = result[0]['dominant_emotion']
            emotion_score = result[0]['emotion'][dominant_emotion]
            
            return dominant_emotion, emotion_score
        else:
            return None, None
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return None, None

# Display emotion tracking history chart
def display_emotion_history_chart():
    if not st.session_state.emotion_history:
        st.info("No emotion history data yet. Use the facial emotion detection to start tracking.")
        return
    
    # Create dataframe from emotion history
    df = pd.DataFrame(st.session_state.emotion_history)
    
    # Map emotions to numeric values for visualization
    emotion_map = {
        "happy": 6, 
        "surprise": 5, 
        "neutral": 4, 
        "disgust": 3, 
        "fear": 2, 
        "angry": 1, 
        "sad": 0,
        "calm": 7,
        "fearful": 2  # map fearful to fear
    }
    
    df["emotion_value"] = df["emotion"].map(lambda x: emotion_map.get(x.lower(), 3))
    
    # Create chart with different colors for face vs voice
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('emotion_value:Q', 
                scale=alt.Scale(domain=[0, 7]),
                axis=alt.Axis(
                    values=[0, 1, 2, 3, 4, 5, 6, 7],
                    labelExpr="datum.value == 0 ? 'Sad' : datum.value == 1 ? 'Angry' : datum.value == 2 ? 'Fear' : datum.value == 3 ? 'Disgust' : datum.value == 4 ? 'Neutral' : datum.value == 5 ? 'Surprise' : datum.value == 6 ? 'Happy' : 'Calm'"
                ),
                title='Emotion'),
        color=alt.Color('source:N', scale=alt.Scale(
            domain=['face', 'voice'],
            range=['#1f77b4', '#ff7f0e']
        ), title="Detection Method"),
        tooltip=['timestamp:T', 'emotion:N', 'score:Q', 'note:N', 'source:N']
    ).properties(
        title='Your Emotion History',
        width=600,
        height=300
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # Filter based on source
    with st.expander("Filter by detection method"):
        col1, col2 = st.columns(2)
        with col1:
            show_face = st.checkbox("Show facial emotions", value=True)
        with col2:
            show_voice = st.checkbox("Show voice emotions", value=True)
        
        filtered_df = df
        if not show_face:
            filtered_df = filtered_df[filtered_df['source'] != 'face']
        if not show_voice:
            filtered_df = filtered_df[filtered_df['source'] != 'voice']
        
        if len(filtered_df) > 0:
            filtered_chart = alt.Chart(filtered_df).mark_line(point=True).encode(
                x=alt.X('timestamp:T', title='Time'),
                y=alt.Y('emotion_value:Q', 
                        scale=alt.Scale(domain=[0, 7]),
                        axis=alt.Axis(
                            values=[0, 1, 2, 3, 4, 5, 6, 7],
                            labelExpr="datum.value == 0 ? 'Sad' : datum.value == 1 ? 'Angry' : datum.value == 2 ? 'Fear' : datum.value == 3 ? 'Disgust' : datum.value == 4 ? 'Neutral' : datum.value == 5 ? 'Surprise' : datum.value == 6 ? 'Happy' : 'Calm'"
                        ),
                        title='Emotion'),
                color=alt.Color('source:N', title="Detection Method"),
                tooltip=['timestamp:T', 'emotion:N', 'score:Q', 'note:N', 'source:N']
            ).properties(
                title='Filtered Emotion History',
                width=600,
                height=300
            ).interactive()
            
            st.altair_chart(filtered_chart, use_container_width=True)
    
    # Show pattern analysis if we have enough data
    if len(df) >= 5:
        st.subheader("Emotion Patterns")
        
        # Calculate most frequent emotion
        most_common = df['emotion'].value_counts().idxmax()
        
        # Calculate time patterns
        df['hour'] = df['timestamp'].dt.hour
        
        morning_emotions = df[df['hour'].between(5, 11)]['emotion'].value_counts().idxmax() if not df[df['hour'].between(5, 11)].empty else "No data"
        afternoon_emotions = df[df['hour'].between(12, 17)]['emotion'].value_counts().idxmax() if not df[df['hour'].between(12, 17)].empty else "No data"
        evening_emotions = df[df['hour'].between(18, 23)]['emotion'].value_counts().idxmax() if not df[df['hour'].between(18, 23)].empty else "No data"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📊 Most frequent emotion: **{most_common}**")
            
            # Mental health insight for most common emotion
            insight_dict = EMOTION_MENTAL_HEALTH_INSIGHTS
            if most_common.lower() == "fearful":
                most_common = "fear"  # Map fearful to fear for insights
            if most_common.lower() == "calm":
                insight_dict = VOICE_EMOTION_INSIGHTS
                
            if most_common.lower() in insight_dict:
                insight = insight_dict[most_common.lower()]
                with st.expander(f"Mental Health Insight for {most_common.capitalize()}"):
                    st.write(insight["description"])
                    st.write(f"**Bangladesh Context:** {insight['connection']}")
                    st.write("**Recommendations:**")
                    for rec in insight["recommendations"]:
                        st.write(f"• {rec}")
        
        with col2:
            st.info("**Time of Day Patterns**")
            st.write(f"🌅 Morning (5AM-11AM): **{morning_emotions}**")
            st.write(f"☀️ Afternoon (12PM-5PM): **{afternoon_emotions}**")
            st.write(f"🌙 Evening (6PM-11PM): **{evening_emotions}**")

# Create tabs for different features
main_tab, emotion_tab = st.tabs(["Chat Assistant", "Emotion Tracker"])

with emotion_tab:
    st.header("Multimodal Emotion Detection & Tracking")
    st.write("This tool helps you track your emotions through facial expressions and voice tone, providing insights into your mental wellbeing.")
    
    detection_tab1, detection_tab2 = st.tabs(["Facial Emotion Detection", "Voice Tone Analysis"])
    
    with detection_tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Detect Your Facial Emotion")
            camera_image = st.camera_input("Take a photo to detect your emotion")
            
            if camera_image is not None:
                # Process the image
                image = Image.open(camera_image)
                emotion, score = detect_face_emotion(image)
                
                if emotion and score:
                    st.session_state.detected_emotion = emotion
                    st.session_state.emotion_score = score
                    
                    # Display the emotion with an emoji
                    emotion_emoji = {
                        "happy": "😊",
                        "sad": "😢",
                        "angry": "😠",
                        "fear": "😨",
                        "surprise": "😲",
                        "neutral": "😐",
                        "disgust": "🤢"
                    }
                    
                    emoji = emotion_emoji.get(emotion.lower(), "")
                    st.success(f"Detected: {emotion} {emoji}\nConfidence: {score:.1f}%")
                    
                    # Get optional note
                    note = st.text_area("Add a note about what you're feeling (optional):", key="face_note")
                    
                    # Save button
                    if st.button("Save to Emotion Journal", key="save_face"):
                        save_emotion_to_history(emotion, score, note, source="face")
                        st.success("Facial emotion saved to your journal!")
                    
                    # Use in chat button
                    if st.button("Discuss this emotion with assistant", key="discuss_face"):
                        # Add message to chat
                        emotion_message = f"Based on my facial emotion detection, I'm feeling {emotion.lower()}. {note}"
                        st.session_state.messages.append({"role": "user", "content": emotion_message})
                        
                        # Switch to chat tab
                        st.rerun()
                else:
                    st.warning("No face or emotion detected. Please try again.")
        
        with col2:
            st.subheader("What Your Expression Means")
            if st.session_state.detected_emotion:
                emotion = st.session_state.detected_emotion.lower()
                if emotion in EMOTION_MENTAL_HEALTH_INSIGHTS:
                    insight = EMOTION_MENTAL_HEALTH_INSIGHTS[emotion]
                    
                    with st.expander("What does this emotion mean for mental health?", expanded=True):
                        st.write(insight["description"])
                    
                    with st.expander("Bangladesh Cultural Context"):
                        st.write(insight["connection"])
                    
                    with st.expander("Recommendations"):
                        for rec in insight["recommendations"]:
                            st.write(f"• {rec}")
                    
                    with st.expander("Local Resources"):
                        st.write(insight["local_resources"])
            else:
                st.info("Take a photo to see what your facial expression reveals about your emotional state.")
    
    with detection_tab2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Analyze Your Voice Tone")
            
            if st.button("Record Voice (5 seconds)", key="record_voice"):
                # Record audio
                audio_file = record_audio(duration=5)
                
                # Analyze the voice
                emotion, confidence = analyze_voice(audio_file)
                
                # Store in session state
                st.session_state.voice_emotion = emotion
                st.session_state.voice_confidence = confidence
                
                # Create a placeholder to play the audio
                st.audio(audio_file)
            
            if st.session_state.voice_emotion:
                emotion = st.session_state.voice_emotion
                confidence = st.session_state.voice_confidence
                
                # Display the emotion with an emoji
                emotion_emoji = {
                    "happy": "😊",
                    "sad": "😢",
                    "angry": "😠",
                    "fearful": "😨",
                    "neutral": "😐",
                    "calm": "😌"
                }
                
                emoji = emotion_emoji.get(emotion.lower(), "")
                st.success(f"Voice Tone: {emotion} {emoji}\nConfidence: {confidence:.1f}%")
                
                # Get optional note
                note = st.text_area("Add a note about what you're feeling (optional):", key="voice_note")
                
                # Save button
                if st.button("Save to Emotion Journal", key="save_voice"):
                    save_emotion_to_history(emotion, confidence, note, source="voice")
                    st.success("Voice emotion saved to your journal!")
                
                # Use in chat button
                if st.button("Discuss this voice tone with assistant", key="discuss_voice"):
                    # Add message to chat
                    emotion_message = f"Based on my voice tone analysis, I'm feeling {emotion.lower()}. {note}"
                    st.session_state.messages.append({"role": "user", "content": emotion_message})
                    
                    # Switch to chat tab
                    st.rerun()
        
        with col2:
            st.subheader("What Your Voice Tone Means")
            if st.session_state.voice_emotion:
                emotion = st.session_state.voice_emotion.lower()
                if emotion in VOICE_EMOTION_INSIGHTS:
                    insight = VOICE_EMOTION_INSIGHTS[emotion]
                    
                    with st.expander("What does this voice tone indicate?", expanded=True):
                        st.write(insight["description"])
                    
                    with st.expander("Bangladesh Cultural Context"):
                        st.write(insight["connection"])
                    
                    with st.expander("Recommendations"):
                        for rec in insight["recommendations"]:
                            st.write(f"• {rec}")
            else:
                st.info("Record your voice to analyze your emotional tone.")
    
    # Show combined emotion history
    st.header("Your Emotion Journal")
    display_emotion_history_chart()

with main_tab:
    # Chat interface
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                # Check if there are YouTube links in the message
                content = message["content"]
                st.write(content)
                
                # Extract and display YouTube videos if present
                youtube_links = re.findall(r'https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([^\s&]+)', content)
                if youtube_links:
                    for link in youtube_links[:1]:  # Limit to first video to avoid cluttering
                        full_url = f"https://www.youtube.com/watch?v={link}"
                        display_youtube_video(full_url)
        elif message["role"] == "function":
            with st.chat_message("assistant", avatar="💭"):
                display_thinking(message["content"])

    # Function to process assistant responses
    def process_assistant_response(result):
        current_messages = st.session_state.messages.copy()
        
        # Store all messages in session state
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                # Skip human messages as we've already added the user input
                continue
                
            elif isinstance(msg, AIMessage):
                if len(current_messages) > 0 and any(m["role"] == "assistant" and m["content"] == msg.content for m in current_messages):
                    continue
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": msg.content})
                
                # Display only the LAST assistant message
                if msg == result["messages"][-1] or (isinstance(result["messages"][-1], FunctionMessage) and msg == [m for m in result["messages"] if isinstance(m, AIMessage)][-1]):
                    with st.chat_message("assistant"):
                        st.write(msg.content)
                        
                        # Extract and display YouTube videos if present
                        youtube_links = re.findall(r'https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([^\s&]+)', msg.content)
                        if youtube_links:
                            for link in youtube_links[:1]:  # Limit to first video to avoid cluttering
                                full_url = f"https://www.youtube.com/watch?v={link}"
                                display_youtube_video(full_url)
            
            elif isinstance(msg, FunctionMessage):
                if len(current_messages) > 0 and any(m["role"] == "function" and m["content"] == msg.content for m in current_messages):
                    continue
                
                # Add function message (reasoning) to chat history but don't display it by default
                st.session_state.messages.append({"role": "function", "content": msg.content, "name": msg.name})
                
                # Only display reasoning if explicitly enabled
                if st.session_state.get("reasoning_visible", True):
                    with st.chat_message("assistant", avatar="💭"):
                        display_thinking(msg.content)

    # Emotion integration in chat UI
    chat_container = st.container()
    
    with chat_container:
        # Create a row for detected emotions
        emotion_col1, emotion_col2 = st.columns(2)
        
        # Display face emotion info if detected
        with emotion_col1:
            if st.session_state.detected_emotion:
                emotion = st.session_state.detected_emotion
                score = st.session_state.emotion_score
                emotion_emoji = {
                    "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
                    "surprise": "😲", "neutral": "😐", "disgust": "🤢"
                }
                emoji = emotion_emoji.get(emotion.lower(), "")
                
                st.info(f"Facial emotion: {emotion} {emoji}")

        # Display voice emotion info if detected
        with emotion_col2:
            if st.session_state.voice_emotion:
                emotion = st.session_state.voice_emotion
                score = st.session_state.voice_confidence
                emotion_emoji = {
                    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
                    "neutral": "😐", "calm": "😌"
                }
                emoji = emotion_emoji.get(emotion.lower(), "")
                
                st.info(f"Voice tone: {emotion} {emoji}")
        
        # Emotion inclusion options
        include_col1, include_col2 = st.columns(2)
        with include_col1:
            include_face = st.checkbox("Include facial emotion", value=False, key="include_face_in_chat", 
                                      disabled=not st.session_state.detected_emotion)
        with include_col2:
            include_voice = st.checkbox("Include voice tone", value=False, key="include_voice_in_chat",
                                       disabled=not st.session_state.voice_emotion)

        # User input
        user_input = st.chat_input("আপনি আজ কেমন বোধ করছেন? (How are you feeling today?)")

        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Build context with emotions if included
            context_with_emotion = user_input
            emotions_added = []
            
            # Add facial emotion if selected
            if st.session_state.detected_emotion and include_face:
                emotion = st.session_state.detected_emotion
                emotion_emoji = {
                    "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
                    "surprise": "😲", "neutral": "😐", "disgust": "🤢"
                }
                emoji = emotion_emoji.get(emotion.lower(), "")
                emotions_added.append(f"facial expression: {emotion} {emoji}")
            
            # Add voice emotion if selected
            if st.session_state.voice_emotion and include_voice:
                emotion = st.session_state.voice_emotion
                emotion_emoji = {
                    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
                    "neutral": "😐", "calm": "😌"
                }
                emoji = emotion_emoji.get(emotion.lower(), "")
                emotions_added.append(f"voice tone: {emotion} {emoji}")
            
            # Add emotions to context if any were selected
            if emotions_added:
                context_with_emotion = f"{user_input} [Detected {', '.join(emotions_added)}]"
            
            # Get response from assistant
            with st.spinner("Thinking..."):
                result = chat_with_mental_health_assistant(context_with_emotion, st.session_state.agent_state)
                st.session_state.agent_state = result
                
                # Process and display the result
                process_assistant_response(result)

# Add resources sidebar
with st.sidebar:
    st.header("Bangladesh Mental Health Resources")
    st.markdown("""
    **Crisis Resources:**
    - National Mental Health Helpline (Bangladesh): 01688-709965, 01688-709966
    - Kaan Pete Roi (Emotional Support): 9612119911
    - Bangladesh Emergency Services: 999
    
    **Mental Health Organizations in Bangladesh:**
    - [National Institute of Mental Health (NIMH)](https://nimhbd.com/)
    - [Bangladesh Association of Psychiatrists](http://www.bap.org.bd/)
    - [Dhaka Community Hospital](http://dchtrust.org/)
    - [Mental Health & Psychosocial Support Network Bangladesh](https://www.mhinnovation.net/organisations/mental-health-psychosocial-support-network-bangladesh)
    
    **Settings:**
    """)
    
    # Emotion detection quick access
    st.subheader("Quick Emotion Detection")
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button("📷 Detect Face", help="Take a photo to detect your facial emotion"):
            st.switch_page("app.py")
            emotion_tab.selectbox("Emotion Tracker")
            detection_tab1.selectbox("Facial Emotion Detection")
    
    with quick_col2:
        if st.button("🎙️ Analyze Voice", help="Record your voice to analyze your tone"):
            st.switch_page("app.py")
            emotion_tab.selectbox("Emotion Tracker")
            detection_tab2.selectbox("Voice Tone Analysis")
    
    # More prominently displayed reasoning toggle
    st.subheader("Thinking Process")
    show_reasoning = st.toggle(
        "Show assistant's detailed thought process", 
        value=True,
        help="Toggle to see how the assistant analyzes your messages and forms responses"
    )
    
    if show_reasoning != st.session_state.get("show_reasoning", True):
        st.session_state["show_reasoning"] = show_reasoning
        if st.session_state.agent_state:
            st.session_state.agent_state["reasoning_visible"] = show_reasoning
            st.info("Reasoning visibility updated! Will apply to next messages.")
            
    st.markdown("""
    **Remember:** This assistant provides support but does not replace professional help.
    
    **Cultural Context:** This assistant has been adapted for Bangladesh, taking into consideration local cultural norms and mental health perspectives.
    """) 