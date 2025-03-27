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
def save_emotion_to_history(emotion, score, note=""):
    if emotion:
        timestamp = datetime.datetime.now()
        emotion_data = {
            "timestamp": timestamp,
            "emotion": emotion,
            "score": score,
            "note": note
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
        "sad": 0
    }
    
    df["emotion_value"] = df["emotion"].map(lambda x: emotion_map.get(x.lower(), 3))
    
    # Create chart
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('emotion_value:Q', 
                scale=alt.Scale(domain=[0, 6]),
                axis=alt.Axis(
                    values=[0, 1, 2, 3, 4, 5, 6],
                    labelExpr="datum.value == 0 ? 'Sad' : datum.value == 1 ? 'Angry' : datum.value == 2 ? 'Fear' : datum.value == 3 ? 'Disgust' : datum.value == 4 ? 'Neutral' : datum.value == 5 ? 'Surprise' : 'Happy'"
                ),
                title='Emotion'),
        color=alt.Color('emotion:N', legend=None),
        tooltip=['timestamp:T', 'emotion:N', 'score:Q', 'note:N']
    ).properties(
        title='Your Emotion History',
        width=600,
        height=300
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
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
            if most_common.lower() in EMOTION_MENTAL_HEALTH_INSIGHTS:
                insight = EMOTION_MENTAL_HEALTH_INSIGHTS[most_common.lower()]
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
    st.header("Facial Emotion Detection & Tracking")
    st.write("This tool helps you track your emotions over time, which can provide insights into your mental wellbeing.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Detect Your Emotion")
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
                note = st.text_area("Add a note about what you're feeling (optional):")
                
                # Save button
                if st.button("Save to Emotion Journal"):
                    save_emotion_to_history(emotion, score, note)
                    st.success("Emotion saved to your journal!")
                
                # Use in chat button
                if st.button("Discuss this emotion with assistant"):
                    # Add message to chat
                    emotion_message = f"Based on my facial emotion detection, I'm feeling {emotion.lower()}. {note}"
                    st.session_state.messages.append({"role": "user", "content": emotion_message})
                    
                    # Switch to chat tab
                    st.rerun()
            else:
                st.warning("No face or emotion detected. Please try again.")
    
    with col2:
        st.subheader("Your Emotion Journal")
        display_emotion_history_chart()
        
        # Mental health insights based on current emotion
        if st.session_state.detected_emotion:
            emotion = st.session_state.detected_emotion.lower()
            if emotion in EMOTION_MENTAL_HEALTH_INSIGHTS:
                insight = EMOTION_MENTAL_HEALTH_INSIGHTS[emotion]
                
                st.subheader(f"Mental Health Insights for {emotion.capitalize()}")
                
                with st.expander("What does this emotion mean for mental health?", expanded=True):
                    st.write(insight["description"])
                
                with st.expander("Bangladesh Cultural Context"):
                    st.write(insight["connection"])
                
                with st.expander("Recommendations"):
                    for rec in insight["recommendations"]:
                        st.write(f"• {rec}")
                
                with st.expander("Local Resources"):
                    st.write(insight["local_resources"])

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
        
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                # Skip human messages as we've already added the user input
                continue
                
            elif isinstance(msg, AIMessage):
                if len(current_messages) > 0 and any(m["role"] == "assistant" and m["content"] == msg.content for m in current_messages):
                    continue
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": msg.content})
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.write(msg.content)
                    
                    # Extract and display YouTube videos if present
                    youtube_links = re.findall(r'https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([^\s&]+)', msg.content)
                    if youtube_links:
                        for link in youtube_links[:1]:
                            full_url = f"https://www.youtube.com/watch?v={link}"
                            display_youtube_video(full_url)
            
            elif isinstance(msg, FunctionMessage):
                if len(current_messages) > 0 and any(m["role"] == "function" and m["content"] == msg.content for m in current_messages):
                    continue
                
                # Add function message (reasoning) to chat history
                st.session_state.messages.append({"role": "function", "content": msg.content, "name": msg.name})
                
                # Display function message (reasoning)
                with st.chat_message("assistant", avatar="💭"):
                    display_thinking(msg.content)

    # Emotion integration in chat UI
    chat_container = st.container()
    
    with chat_container:
        # Display emotion info if detected
        if st.session_state.detected_emotion:
            emotion = st.session_state.detected_emotion
            score = st.session_state.emotion_score
            emotion_emoji = {
                "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
                "surprise": "😲", "neutral": "😐", "disgust": "🤢"
            }
            emoji = emotion_emoji.get(emotion.lower(), "")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Detected emotion: {emotion} {emoji} ({score:.1f}%)")
            with col2:
                include_emotion = st.checkbox("Include in message", value=False, key="include_emotion_in_chat")

        # User input
        user_input = st.chat_input("আপনি আজ কেমন বোধ করছেন? (How are you feeling today?)")

        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # If there's a detected emotion and the checkbox is checked, include it in the context
            context_with_emotion = user_input
            
            if st.session_state.detected_emotion and st.session_state.get("include_emotion_in_chat", False):
                emotion = st.session_state.detected_emotion
                emotion_emoji = {
                    "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
                    "surprise": "😲", "neutral": "😐", "disgust": "🤢"
                }
                emoji = emotion_emoji.get(emotion.lower(), "")
                context_with_emotion = f"{user_input} [Detected emotion: {emotion} {emoji}, score: {st.session_state.emotion_score:.1f}%]"
            
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