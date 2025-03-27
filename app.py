import streamlit as st
from mental_health_assistant import chat_with_mental_health_assistant
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
import re

st.set_page_config(
    page_title="Mental Health Support Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

# Title and description
st.title("🧠 Mental Health Support Assistant")
st.markdown("""
This assistant is here to provide mental health support and information. 
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

# User input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get response from assistant
    with st.spinner("Thinking..."):
        result = chat_with_mental_health_assistant(user_input, st.session_state.agent_state)
        st.session_state.agent_state = result
        
        # Process all messages in the result
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

# Add sidebar with important resources
with st.sidebar:
    st.header("Important Resources")
    st.markdown("""
    **Crisis Resources:**
    - National Suicide Prevention Lifeline: 988 or 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    - Emergency Services: 911 (US)
    
    **Self-Care Resources:**
    - [Mental Health America](https://www.mhanational.org/)
    - [National Alliance on Mental Illness](https://www.nami.org/)
    - [Healthline: Mental Health Resources](https://www.healthline.com/health/mental-health/resources)
    
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
    """) 