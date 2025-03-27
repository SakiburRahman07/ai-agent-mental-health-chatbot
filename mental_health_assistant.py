import os
import re
from typing import List, Annotated, TypedDict, Union, Dict, Optional, Literal, Any
from typing_extensions import TypedDict, NotRequired

# LangGraph and LangChain components
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.tools import tool
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field

# The key fix: Add proper annotations for the state
from langgraph.graph.message import add_messages

# Define agent state with proper annotation for messages
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage, FunctionMessage]], add_messages]
    emotion_analysis: NotRequired[Dict[str, Any]]
    response_strategy: NotRequired[Dict[str, Any]]
    immediate_resources_needed: NotRequired[bool]
    reasoning_visible: NotRequired[bool]
    user_preferences: NotRequired[Dict[str, Any]]
    tool_results: NotRequired[Dict[str, Any]]
    query_route: NotRequired[str]
    # New fields for enhanced features
    mood_history: NotRequired[List[Dict[str, Any]]]
    cultural_context: NotRequired[Dict[str, Any]]
    cbt_progress: NotRequired[Dict[str, Any]]
    self_care_recommendations: NotRequired[List[Dict[str, Any]]]
    last_professional_referral: NotRequired[str]  # Timestamp of last referral
    psychoeducation_topics_covered: NotRequired[List[str]]

# Tools and utilities
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import YoutubeLoader

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Set up LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.7
)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setup vector store with AstraDB
import cassio
from langchain.vectorstores.cassandra import Cassandra

cassio.init(
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    database_id=os.getenv("ASTRA_DB_ID")
)

# Mental health resources
documents = [
    "Depression is a common mental health disorder characterized by persistent sadness and loss of interest in activities.",
    "Anxiety disorders involve excessive worry that's difficult to control.",
    "Mindfulness meditation can help reduce stress and anxiety by focusing on the present moment.",
    "Cognitive Behavioral Therapy (CBT) is an effective treatment for many mental health conditions.",
    "Self-care practices like regular exercise, proper sleep, and healthy eating can improve mental well-being.",
    "Trauma can have lasting effects on mental health, but treatment approaches like EMDR can be helpful.",
    "Social support is crucial for mental health recovery - connecting with others can reduce feelings of isolation.",
    "Dialectical Behavior Therapy (DBT) combines cognitive techniques with mindfulness to help regulate emotions.",
    "Recovery from mental health challenges is possible with the right support and treatment approach.",
    "Setting boundaries is an important self-care practice for protecting your mental health.",
]

# Create vector store
vectorstore = Cassandra(
    embedding=embeddings,
    table_name="mental_health_resources",
    session=None,
    keyspace=None
)

# Add documents
for doc in documents:
    vectorstore.add_texts([doc])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define output schemas for structured reasoning
class EmotionAnalysis(BaseModel):
    """Analysis of the user's emotional state and crisis level."""
    primary_emotion: str = Field(description="The primary emotion detected in the message")
    emotion_justification: str = Field(description="Justification for why this emotion was detected")
    crisis_level: str = Field(description="Assessed crisis level: low, medium, high, or very_high")
    crisis_justification: str = Field(description="Justification for the crisis level assessment")
    needs_immediate_resources: bool = Field(description="Whether immediate crisis resources are needed")
    reasoning: str = Field(description="Step-by-step reasoning about the emotional analysis")

class ResponseStrategy(BaseModel):
    """Strategy for how to respond to the user."""
    approach: str = Field(description="Emotional approach to take (empathize, validate, encourage, educate, etc)")
    key_points: List[str] = Field(description="Key points to address in the response")
    appropriate_tools: List[str] = Field(description="Tools that might be helpful (if any)")
    reasoning: str = Field(description="Step-by-step reasoning for this response strategy")

# Enhanced tools
@tool
def search_mental_health_info(query: str) -> str:
    """Search for mental health information from our knowledge base."""
    results = retriever.invoke(query)
    if results:
        return "\n".join([doc.page_content for doc in results])
    return "No specific information found about that topic."

# Set up various tools
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=2)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

tavily_search_tool = TavilySearchResults(max_results=3)
youtube_search_tool = YouTubeSearchTool()

@tool
def search_mental_health_videos(query: str) -> str:
    """
    Search for mental health videos related to specific topics.
    
    Args:
        query: The mental health topic to search for videos about
        
    Returns:
        A formatted list of relevant YouTube videos
    """
    # Create a more targeted search query for better results
    enhanced_query = f"mental health {query} expert advice,3"  # Request 3 results
    
    # Use the YouTube search tool with the enhanced query
    try:
        # Get results from YouTube tool
        results = youtube_search_tool.run(enhanced_query)
        
        if not results or results == "[]":
            return "No videos found on this topic."
            
        # The tool.run() returns a string that looks like: "['url1', 'url2', 'url3']"
        # Convert the string to a Python list while handling potential format issues
        try:
            # Clean the string and convert to a list
            url_list = eval(results)
            
            # If successful, return the URLs as a newline-separated string
            return "\n".join(url_list[:3])  # Limit to top 3 videos
        except:
            # If parsing fails, just return the raw results as it might already be formatted
            return results
            
    except Exception as e:
        return f"Error searching for videos: {str(e)}"

@tool
def get_youtube_transcript_and_summary(url: str) -> Dict[str, str]:
    """Fetch and summarize content from a YouTube video."""
    try:
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True, language=["en"]
        )
        docs = loader.load()
        
        if not docs:
            return {"transcript": "", "summary": "Could not extract content from this video."}
        
        transcript = docs[0].page_content[:2000] + "..." if len(docs[0].page_content) > 2000 else docs[0].page_content
        
        # Generate summary with reasoning
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a mental health expert who creates helpful summaries of videos."),
            HumanMessage(content=f"Create a helpful summary of this video transcript in the context of mental health support. Include key points and advice.\n\nTRANSCRIPT:\n{transcript}")
        ])
        
        formatted_messages = summary_prompt.format_messages()
        summary_response = llm.invoke(formatted_messages)
        
        return {
            "title": docs[0].metadata.get("title", "Video"),
            "transcript": transcript,
            "summary": summary_response.content
        }
    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}

@tool
def generate_video_blog(url: str) -> Dict[str, str]:
    """
    Generate a structured blog post from a YouTube video about mental health topics.
    
    Args:
        url: The YouTube video URL to summarize into a blog format
        
    Returns:
        A dictionary containing the blog title, content, and key points
    """
    try:
        # Load the YouTube video content
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=True, language=["en"]
        )
        docs = loader.load()
        
        if not docs:
            return {
                "title": "Unable to Generate Blog",
                "content": "Could not extract content from this video.",
                "key_points": []
            }
        
        transcript = docs[0].page_content
        video_title = docs[0].metadata.get("title", "Mental Health Video")
        
        # Create blog post generation prompt
        blog_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a professional mental health content writer who creates 
            engaging, informative blog posts from video content. Format your response as a well-structured 
            blog post with:
            
            1. An engaging title that captures the essence of the video
            2. A brief introduction explaining the topic's importance
            3. 3-5 main sections with helpful content and advice
            4. A conclusion with actionable takeaways
            5. 3-5 bullet point key highlights from the video
            
            Make the blog post conversational, evidence-based, and supportive in tone.
            Maximum length: 600 words."""),
            HumanMessage(content=f"""Create a blog post based on this mental health video transcript.
            
            VIDEO TITLE: {video_title}
            
            TRANSCRIPT:
            {transcript[:3000]}... [transcript continues]
            
            Generate a complete, well-structured blog post that captures the key insights and advice from this video.
            """)
        ])
        
        # Generate the blog post
        formatted_messages = blog_prompt.format_messages()
        blog_response = llm.invoke(formatted_messages)
        
        # Extract key points (would be more sophisticated in production)
        content = blog_response.content
        
        # Find key points section (often at the end with bullet points)
        key_points = []
        if "key points" in content.lower() or "highlights" in content.lower() or "takeaway" in content.lower():
            # Simple extraction of bullet points
            for line in content.split('\n'):
                if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*'):
                    key_points.append(line.strip())
        
        # If no bullet points found, create a summary of key points
        if not key_points:
            key_points_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Extract 3-5 key points from this blog post as bullet points:"),
                HumanMessage(content=content)
            ])
            key_points_response = llm.invoke(key_points_prompt.format_messages())
            key_points = key_points_response.content.split('\n')
        
        return {
            "title": video_title,
            "content": content,
            "key_points": key_points
        }
    except Exception as e:
        return {
            "title": "Error Creating Blog",
            "content": f"Sorry, I encountered an error while creating the blog: {str(e)}",
            "key_points": []
        }

@tool
def suggest_cbt_exercise(emotion: str, context: str) -> Dict[str, Any]:
    """
    Suggest an appropriate CBT (Cognitive Behavioral Therapy) exercise based on the user's emotional state.
    
    Args:
        emotion: The primary emotion the user is experiencing
        context: Brief context about the user's situation
        
    Returns:
        A dictionary containing the exercise details
    """
    # Define CBT exercises for different emotions
    cbt_exercises = {
        "anxious": [
            {
                "name": "Thought Record",
                "description": "Identify the anxious thought, consider evidence for and against it, and develop a balanced perspective.",
                "steps": [
                    "Write down the specific thought causing anxiety",
                    "Rate your belief in this thought (0-100%)",
                    "List evidence that supports this thought",
                    "List evidence that contradicts this thought",
                    "Create a balanced alternative thought",
                    "Rate your belief in the alternative thought (0-100%)"
                ]
            },
            {
                "name": "Grounding Technique",
                "description": "The 5-4-3-2-1 technique to bring your awareness to the present moment.",
                "steps": [
                    "Name 5 things you can see",
                    "Name 4 things you can feel/touch",
                    "Name 3 things you can hear",
                    "Name 2 things you can smell",
                    "Name 1 thing you can taste"
                ]
            }
        ],
        "sad": [
            {
                "name": "Behavioral Activation",
                "description": "Plan and engage in activities that typically bring you joy or a sense of accomplishment.",
                "steps": [
                    "Make a list of activities you used to enjoy",
                    "Choose one small activity from the list",
                    "Schedule a specific time to do it, even if you don't feel like it",
                    "After completing it, note how your mood changed"
                ]
            },
            {
                "name": "Gratitude Journal",
                "description": "Write down things you're grateful for to shift focus toward positive aspects of life.",
                "steps": [
                    "Set aside 5 minutes",
                    "Write down 3 specific things you're grateful for today",
                    "For each item, write why you're grateful for it",
                    "Notice how you feel after completing this exercise"
                ]
            }
        ],
        "angry": [
            {
                "name": "STOPP Technique",
                "description": "A strategy to pause and respond more effectively when feeling angry.",
                "steps": [
                    "Stop - Pause, don't react immediately",
                    "Take a breath - Breathe deeply, in through nose, out through mouth",
                    "Observe - What am I thinking and feeling?",
                    "Perspective - Is this fact or opinion? Is there another way of seeing this?",
                    "Practice what works - What's the most helpful thing to do right now?"
                ]
            }
        ],
        "fearful": [
            {
                "name": "Exposure Hierarchy",
                "description": "Gradually face feared situations in a controlled way to reduce anxiety.",
                "steps": [
                    "List situations that trigger your fear, from least to most frightening",
                    "Start with the least frightening item",
                    "Practice facing it until your anxiety decreases",
                    "Move to the next item only when you're ready"
                ]
            }
        ]
    }
    
    # Default exercises for any emotion
    default_exercises = [
        {
            "name": "Mindfulness Meditation",
            "description": "Focus on your breath to anchor yourself in the present moment.",
            "steps": [
                "Find a quiet place to sit comfortably",
                "Close your eyes or maintain a soft gaze",
                "Focus on your breath, without trying to change it",
                "When your mind wanders, gently bring attention back to your breath",
                "Start with 5 minutes and gradually increase"
            ]
        },
        {
            "name": "Cognitive Restructuring",
            "description": "Identify and challenge negative thought patterns.",
            "steps": [
                "Notice the negative thought",
                "Identify the thinking distortion (e.g., catastrophizing, black-and-white thinking)",
                "Challenge the thought: Is it based on facts? What evidence contradicts it?",
                "Create a more balanced, realistic thought"
            ]
        }
    ]
    
    # Get exercises for the specific emotion or use default
    emotion_specific = cbt_exercises.get(emotion.lower(), [])
    available_exercises = emotion_specific + default_exercises
    
    # Select an exercise based on context
    import random
    selected_exercise = random.choice(available_exercises)
    
    # Add a personalized introduction
    if emotion.lower() in ["anxious", "fearful"]:
        intro = f"I notice you're feeling {emotion}. This {selected_exercise['name']} exercise can help reduce anxiety and bring a sense of calm."
    elif emotion.lower() in ["sad", "depressed"]:
        intro = f"When you're feeling {emotion}, it can be hard to take action. This {selected_exercise['name']} exercise is designed to gently help shift your perspective."
    elif emotion.lower() in ["angry", "frustrated"]:
        intro = f"I understand you're feeling {emotion}. This {selected_exercise['name']} technique can help you process those feelings in a healthy way."
    else:
        intro = f"Based on what you've shared, I think this {selected_exercise['name']} exercise might be helpful for you right now."
    
    return {
        "name": selected_exercise["name"],
        "description": selected_exercise["description"],
        "introduction": intro,
        "steps": selected_exercise["steps"]
    }

def suggest_self_care(state: AgentState) -> Dict:
    """Suggest personalized self-care activities."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    user_preferences = state.get("user_preferences", {})
    
    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
    
    # Get self-care suggestions
    try:
        # Direct invocation of the function instead of using the tool decorator
        # This bypasses the callback system that's causing the error
        emotion = primary_emotion
        preferences = user_preferences
        
        # Categories of self-care activities
        physical_activities = [
            "Take a 10-minute walk outside, focusing on the sensations around you",
            "Do a quick 5-minute stretch routine to release tension",
            "Try 3 minutes of jumping jacks or dancing to get your blood flowing",
            "Practice deep breathing: inhale for 4 counts, hold for 4, exhale for 6",
            "Splash cold water on your face to help reset your nervous system"
        ]
        
        mental_activities = [
            "Take a 5-minute break from screens",
            "Write down three thoughts you're having, then challenge their accuracy",
            "Do a quick puzzle or brain game to shift your focus",
            "Visualize a peaceful place for 2 minutes with your eyes closed",
            "Listen to a guided meditation (I can suggest one if you'd like)"
        ]
        
        emotional_activities = [
            "Write down three things you're grateful for right now",
            "Text a supportive friend or family member just to say hello",
            "Listen to a song that matches how you'd like to feel",
            "Look at photos that bring back positive memories",
            "Give yourself permission to feel your emotions without judgment for 5 minutes"
        ]
        
        # Emotion-specific recommendations
        if emotion.lower() in ["anxious", "stressed", "fearful", "worried"]:
            primary_activities = [
                "Try box breathing: inhale for 4 counts, hold for 4, exhale for 4, hold for 4",
                "Progressive muscle relaxation: tense and release each muscle group",
                "Focus on an object near you and describe 5 details about it",
                "Step outside and name 3 things you can see, hear, and feel",
                "Wrap yourself in a warm blanket and make a cup of caffeine-free tea"
            ]
        elif emotion.lower() in ["sad", "depressed", "down", "hopeless"]:
            primary_activities = [
                "Do just one small task and celebrate completing it",
                "Open curtains or blinds to let in natural light",
                "Step outside for 5 minutes of sunlight",
                "Listen to uplifting music or a comedy podcast",
                "Call or message someone who makes you feel good"
            ]
        elif emotion.lower() in ["angry", "frustrated", "irritated"]:
            primary_activities = [
                "Write down what's bothering you, then tear it up",
                "Do a physical activity to release tension (jumping jacks, pushups)",
                "Count backward slowly from 20 to 1",
                "Find a private place and let yourself scream into a pillow",
                "Wash your hands with cold water, focusing on the sensation"
            ]
        else:  # For other emotions or to maintain positive states
            primary_activities = [
                "Take a moment to appreciate something beautiful around you",
                "Do something creative for 10 minutes",
                "Send a kind message to someone",
                "Take a few minutes to plan something you can look forward to",
                "Write down one thing you're proud of about yourself"
            ]
        
        # Combine and select activities
        import random
        
        all_activities = primary_activities + physical_activities + mental_activities + emotional_activities
        selected_activity = random.choice(primary_activities)  # Bias toward emotion-specific activities
        alternative_activity = random.choice([a for a in all_activities if a != selected_activity])
        
        activity_benefit = ""
        if emotion.lower() in ["anxious", "stressed", "fearful", "worried"]:
            activity_benefit = "ground you in the present moment and calm your nervous system"
        elif emotion.lower() in ["sad", "depressed", "down", "hopeless"]:
            activity_benefit = "gently activate your energy and connect with positive experiences"
        elif emotion.lower() in ["angry", "frustrated", "irritated"]:
            activity_benefit = "release tension and create space between feelings and reactions"
        else:
            activity_benefit = "maintain balance and nurture your wellbeing"
        
        suggestions = {
            "primary_suggestion": selected_activity,
            "alternative_suggestion": alternative_activity,
            "rationale": f"When feeling {emotion}, activities that help {activity_benefit} can be particularly helpful."
        }
    except Exception as e:
        # Fallback in case of any error
        suggestions = {
            "primary_suggestion": "Take a few deep breaths and notice how your body feels",
            "alternative_suggestion": "Drink a glass of water and stretch for a moment",
            "rationale": "Taking a moment for simple self-care can help with any emotion."
        }
    
    # Format the suggestions as a message
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Thinking About Self-Care For You**
        
        When feeling **{primary_emotion}**, specific self-care activities can be particularly beneficial.
        
        I'm considering what might be most helpful for you right now, with a focus on quick, accessible activities.
        
        Let me suggest something that could help...
        """
        
        new_messages.append(
            FunctionMessage(
                name="self_care_reasoning",
                content=reasoning_message
            )
        )
    
    # Create the self-care message
    care_message = f"""
    **Self-Care Suggestion**
    
    {suggestions["rationale"]}
    
    **You might try:** {suggestions["primary_suggestion"]}
    
    **Alternatively:** {suggestions["alternative_suggestion"]}
    
    Would you like to try one of these now, or would you prefer a different type of self-care activity?
    """
    
    return {"messages": new_messages + [AIMessage(content=care_message)]}

def get_activity_benefit(emotion: str) -> str:
    """Get the benefit description based on emotion."""
    if emotion.lower() in ["anxious", "stressed", "fearful", "worried"]:
        return "ground you in the present moment and calm your nervous system"
    elif emotion.lower() in ["sad", "depressed", "down", "hopeless"]:
        return "gently activate your energy and connect with positive experiences"
    elif emotion.lower() in ["angry", "frustrated", "irritated"]:
        return "release tension and create space between feelings and reactions"
    else:
        return "maintain balance and nurture your wellbeing"

@tool
def provide_psychoeducation(topic: str) -> Dict[str, str]:
    """
    Provide clear, concise educational information about mental health topics.
    
    Args:
        topic: The mental health topic to explain
        
    Returns:
        A dictionary with educational content
    """
    # Core mental health topics
    psychoeducation_content = {
        "anxiety": {
            "title": "Understanding Anxiety",
            "description": "Anxiety is the body's natural response to stress. It's a feeling of fear or apprehension about what's to come.",
            "key_points": [
                "Anxiety triggers your 'fight-or-flight' response",
                "Physical symptoms can include racing heart, rapid breathing, and muscle tension",
                "While uncomfortable, anxiety is not dangerous and is a normal human experience",
                "Anxiety becomes a disorder when it's excessive, persistent, and interferes with daily life"
            ],
            "helpful_framework": "Think of anxiety as your body's alarm system. Sometimes it's oversensitive and goes off when there's no real danger.",
            "management_strategies": [
                "Regular exercise can reduce stress hormones",
                "Mindfulness and breathing techniques help calm the nervous system",
                "Gradually facing feared situations (exposure) reduces anxiety over time",
                "Limiting caffeine and alcohol can reduce symptoms"
            ]
        },
        "depression": {
            "title": "Understanding Depression",
            "description": "Depression is more than just feeling sad. It's a mood disorder that causes persistent feelings of sadness and loss of interest.",
            "key_points": [
                "Depression affects how you feel, think, and handle daily activities",
                "It involves changes in brain chemistry and function",
                "Can be triggered by life events, biological factors, or arise without a clear cause",
                "Is not a sign of weakness and isn't something people can 'snap out of'"
            ],
            "helpful_framework": "Depression is like wearing tinted glasses that make everything appear darker and more hopeless than it actually is.",
            "management_strategies": [
                "Regular physical activity boosts mood-enhancing chemicals",
                "Maintaining social connections even when you don't feel like it",
                "Breaking tasks into smaller steps to make them more manageable",
                "Establishing daily routines to provide structure"
            ]
        },
        "stress": {
            "title": "Understanding Stress",
            "description": "Stress is your body's reaction to pressure from a situation or life event.",
            "key_points": [
                "Some stress is normal and even helpful for motivation",
                "Chronic stress can affect your physical and mental health",
                "Your perception of a situation influences your stress response",
                "Everyone's stress triggers and responses are different"
            ],
            "helpful_framework": "Think of stress like electricity - the right amount powers your life, but too much can cause damage.",
            "management_strategies": [
                "Identify your stress triggers to better manage them",
                "Practice relaxation techniques like deep breathing",
                "Physical activity helps burn off stress hormones",
                "Setting boundaries around time and commitments"
            ]
        },
        "sleep": {
            "title": "Sleep and Mental Health",
            "description": "Sleep and mental health are closely connected. Sleep deprivation affects your psychological state and mental health.",
            "key_points": [
                "Sleep problems may increase risk for developing mental health conditions",
                "Mental health issues can disrupt sleep patterns",
                "Most adults need 7-9 hours of quality sleep per night",
                "REM sleep is particularly important for emotional processing"
            ],
            "helpful_framework": "Sleep is like your brain's cleaning service - it processes emotions and clears out mental 'debris' from the day.",
            "management_strategies": [
                "Keep a consistent sleep schedule, even on weekends",
                "Create a relaxing bedtime routine",
                "Limit screen time before bed (blue light affects melatonin)",
                "Keep your bedroom cool, dark, and quiet"
            ]
        },
        "mindfulness": {
            "title": "Understanding Mindfulness",
            "description": "Mindfulness is the practice of being fully present and engaged in the moment, aware of your thoughts and feelings without judgment.",
            "key_points": [
                "Mindfulness helps you observe thoughts without being controlled by them",
                "Regular practice can physically change brain structure in positive ways",
                "You don't need to meditate for hours - even brief practices help",
                "Mindfulness is a skill that improves with practice"
            ],
            "helpful_framework": "Mindfulness is like developing a mental watchtower, observing your thoughts and feelings as they pass by without getting swept away by them.",
            "management_strategies": [
                "Start with brief 1-2 minute mindfulness exercises",
                "Use everyday activities (like brushing teeth) as mindfulness opportunities",
                "Try guided mindfulness apps or videos",
                "Be patient and gentle with yourself as you learn"
            ]
        }
    }
    
    # Try to match the topic to our content
    matched_topic = None
    
    # Check for exact matches first
    if topic.lower() in psychoeducation_content:
        matched_topic = topic.lower()
    else:
        # Check for partial matches
        for potential_topic in psychoeducation_content.keys():
            if potential_topic in topic.lower() or topic.lower() in potential_topic:
                matched_topic = potential_topic
                break
    
    if matched_topic:
        content = psychoeducation_content[matched_topic]
        return {
            "topic": matched_topic,
            "title": content["title"],
            "description": content["description"],
            "key_points": content["key_points"],
            "framework": content["helpful_framework"],
            "strategies": content["management_strategies"]
        }
    else:
        # If no match, provide a generic response about finding information
        return {
            "topic": topic,
            "title": f"Information About {topic.capitalize()}",
            "description": f"I don't have specific information about {topic} in my knowledge base.",
            "key_points": [
                "Mental health topics are diverse and interconnected",
                "It's important to seek information from reliable sources",
                "Consider consulting with a mental health professional for specific concerns"
            ],
            "framework": "Would you like me to help you find general information about this topic instead?",
            "strategies": [
                "Consider reading from evidence-based mental health resources",
                "Academic journals and reputable health organizations offer reliable information",
                "Mental health professionals can provide personalized guidance"
            ]
        }

@tool
def generate_therapeutic_story(struggle: str, context: str = "") -> Dict[str, str]:
    """
    Generate a personalized therapeutic story based on narrative therapy principles.
    
    Args:
        struggle: The specific challenge or emotion the user is facing (e.g., "fear of failure")
        context: Additional context about the user's situation (optional)
        
    Returns:
        A dictionary containing the therapeutic story and reflection questions
    """
    # Define common story archetypes and metaphors for different struggles
    story_archetypes = {
        "anxiety": {
            "metaphors": ["river", "storm", "mountain", "maze"],
            "themes": ["courage", "breathing", "perspective", "facing fears"],
            "heroes": ["explorer", "navigator", "climber", "guide"]
        },
        "depression": {
            "metaphors": ["winter", "dark cave", "fog", "heavy backpack"],
            "themes": ["light", "seasons", "rest", "connection"],
            "heroes": ["traveler", "gardener", "lighthouse keeper", "dawn bringer"]
        },
        "grief": {
            "metaphors": ["ocean", "changing tree", "bridge", "mosaic"],
            "themes": ["memory", "honoring", "integration", "transformation"],
            "heroes": ["memory keeper", "bridge builder", "weaver", "composer"]
        },
        "fear": {
            "metaphors": ["closed door", "shadow", "uncharted territory", "high cliff"],
            "themes": ["curiosity", "courage", "unknown", "stepping forward"],
            "heroes": ["door opener", "light bearer", "map maker", "cliff climber"]
        },
        "anger": {
            "metaphors": ["fire", "storm", "tangled knot", "pressure cooker"],
            "themes": ["power", "energy", "boundaries", "transformation"],
            "heroes": ["fire keeper", "weather worker", "untangler", "pressure release"]
        },
        "guilt": {
            "metaphors": ["heavy stone", "locked room", "repeating record", "mirror"],
            "themes": ["forgiveness", "compassion", "learning", "release"],
            "heroes": ["stone bearer", "key finder", "new composer", "truth seeker"]
        },
        "shame": {
            "metaphors": ["mask", "invisibility cloak", "cave", "wall"],
            "themes": ["authenticity", "acceptance", "belonging", "speaking"],
            "heroes": ["mask remover", "voice finder", "light bearer", "bridge builder"]
        },
        "identity": {
            "metaphors": ["river", "mirror", "garden", "tapestry"],
            "themes": ["growth", "integration", "discovery", "cultivation"],
            "heroes": ["gardener", "navigator", "weaver", "mirror holder"]
        },
        "perfectionism": {
            "metaphors": ["impossible mountain", "never-ending path", "unreachable star", "golden cage"],
            "themes": ["acceptance", "growth", "process", "good enough"],
            "heroes": ["path walker", "star counter", "cage opener", "good-enough finder"]
        },
        "failure": {
            "metaphors": ["blocked road", "fallen attempt", "closed door", "missed mark"],
            "themes": ["learning", "resilience", "redirection", "wisdom"],
            "heroes": ["path finder", "rising phoenix", "door creator", "arrow adjuster"]
        }
    }
    
    # Identify which archetype to use based on the struggle
    identified_archetype = None
    for struggle_type, archetype in story_archetypes.items():
        if struggle_type in struggle.lower():
            identified_archetype = archetype
            struggle_key = struggle_type
            break
    
    # Default to "fear" archetype if no match found
    if not identified_archetype:
        import random
        struggle_key = random.choice(list(story_archetypes.keys()))
        identified_archetype = story_archetypes[struggle_key]
    
    # Create a prompt to generate the therapeutic story
    story_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a therapeutic storyteller who creates healing narratives based on narrative therapy principles.
        
        Create a brief allegorical story (500-800 words) where the protagonist faces and learns to work with {struggle}. 
        The story should:
        
        1. Use metaphors related to: {', '.join(identified_archetype['metaphors'])}
        2. Incorporate themes of: {', '.join(identified_archetype['themes'])}
        3. Cast the protagonist as a type of: {', '.join(identified_archetype['heroes'])}
        4. Include these narrative therapy elements:
           - Externalization (separating the person from the problem)
           - Identifying unique outcomes (exceptions to the problem's influence)
           - Thickening the preferred narrative (strengthening the story of capability)
        5. End with a sense of hope and new possibilities
        6. Include 3-4 reflection questions to help the reader connect the story to their own experience
        
        The story should be engaging, conversational, and avoid being preachy or overly simplistic.
        Write in second person ("you") to help the reader identify as the protagonist.
        
        Format your response as:
        
        # TITLE
        
        STORY CONTENT
        
        ## Reflection Questions:
        1. First question
        2. Second question
        3. Third question
        """),
        HumanMessage(content=f"Create a therapeutic story for someone struggling with {struggle}. Additional context about their situation: {context}")
    ])
    
    # Generate the story
    formatted_messages = story_prompt.format_messages()
    story_response = llm.invoke(formatted_messages)
    
    # Extract title, story content, and reflection questions
    content = story_response.content
    
    # Parse the content
    title = "Your Therapeutic Story"
    story_body = content
    reflection_questions = []
    
    # Try to extract the title
    title_match = re.search(r'#\s*(.*?)\n', content)
    if title_match:
        title = title_match.group(1).strip()
        story_body = content.replace(title_match.group(0), "", 1)
    
    # Try to extract reflection questions
    questions_section = re.search(r'##\s*Reflection Questions:?([\s\S]*?)($|#)', content + "#")
    if questions_section:
        questions_text = questions_section.group(1)
        # Find numbered questions
        question_matches = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n##|\n#|$)', questions_text + "\n##")
        if question_matches:
            reflection_questions = [q.strip() for q in question_matches]
        story_body = content.replace(questions_section.group(0), "", 1)
    
    # Ensure we have at least some reflection questions
    if not reflection_questions:
        reflection_questions = [
            f"How does this story relate to your experience with {struggle}?",
            "What part of the story resonates most with you, and why?",
            "What new perspective might this story offer about your situation?"
        ]
    
    # Create therapeutic metaphor explanation
    metaphor_explanation = f"""
    In narrative therapy, metaphors help externalize problems, allowing us to see them as separate from our identity.
    
    This story uses the metaphor of {identified_archetype['metaphors'][0]} to help reframe your experience with {struggle}.
    
    By seeing yourself as the {identified_archetype['heroes'][0]} in this journey, you can draw on your inherent strengths of {identified_archetype['themes'][0]} and {identified_archetype['themes'][1]}.
    """
    
    return {
        "title": title,
        "story": story_body.strip(),
        "reflection_questions": reflection_questions,
        "metaphor_explanation": metaphor_explanation.strip(),
        "struggle_addressed": struggle
    }

# System message
SYSTEM_MESSAGE = """You are an empathetic mental health assistant designed to help people through difficult times.

Your primary goals are:
1. Support people experiencing mental health challenges with empathy and understanding
2. Provide evidence-based information about mental health
3. Suggest healthy coping strategies and self-care practices
4. Recognize when someone might need professional help and gently suggest resources
5. Help users track their mood and notice patterns over time
6. Offer CBT-based techniques to manage difficult thoughts and emotions
7. Provide psychoeducation about mental health concepts

ETHICAL GUIDELINES:
- Always include clear disclaimers that you are not a therapist or mental health professional
- Regularly encourage users to speak with a qualified professional, especially for ongoing concerns
- Never diagnose conditions or suggest specific medications
- Maintain a non-judgmental stance in all interactions
- If someone seems to be in crisis, prioritize their safety above all else
- Adapt your tone based on the user's emotional state (gentler for distress, more upbeat for motivation)
- Consider cultural context when offering suggestions
- Respect user autonomy - offer options rather than directives

General Guidelines:
- Always respond with compassion and without judgment
- Use a warm, conversational tone as if talking to a friend
- Show that you're listening by acknowledging feelings
- Ask clarifying questions when needed
- Be authentic and genuine in your responses
- When appropriate, explain your reasoning process to build trust

IMPORTANT DISCLAIMER TO INCLUDE REGULARLY:
"I'm here to support you, but I'm not a licensed therapist or healthcare provider. For ongoing concerns, please speak with a qualified mental health professional."

Remember: Your role is to provide support, not replace professional mental health care.
"""

# Define node functions
def initialize_state(state: Dict) -> Dict:
    """Initialize the agent state with default values."""
    if "messages" not in state:
        state["messages"] = [SystemMessage(content=SYSTEM_MESSAGE)]
    if "emotion_analysis" not in state:
        state["emotion_analysis"] = None
    if "response_strategy" not in state:
        state["response_strategy"] = None
    if "immediate_resources_needed" not in state:
        state["immediate_resources_needed"] = False
    if "reasoning_visible" not in state:
        state["reasoning_visible"] = True  # Default to showing reasoning
    if "user_preferences" not in state:
        state["user_preferences"] = {
            "likes_videos": False,
            "prefers_brief_responses": False,
            "wants_resources": True
        }
    
    # New initializations for enhanced features
    if "mood_history" not in state:
        state["mood_history"] = []
    if "cultural_context" not in state:
        state["cultural_context"] = {"language": "english", "region": "global"}
    if "cbt_progress" not in state:
        state["cbt_progress"] = {"exercises_completed": [], "current_focus": None}
    if "self_care_recommendations" not in state:
        state["self_care_recommendations"] = []
    if "last_professional_referral" not in state:
        state["last_professional_referral"] = None
    if "psychoeducation_topics_covered" not in state:
        state["psychoeducation_topics_covered"] = []
    
    return state

def analyze_emotion_and_needs(state: AgentState) -> Dict:
    """Analyze the user's emotional state and needs with explicit reasoning."""
    messages = state["messages"]
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # Structured emotion analysis with reasoning - using a direct approach instead of function calling
    emotion_analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a mental health professional who analyzes emotional content.
        
        Analyze the message and provide a structured assessment in the following JSON format:
        {
            "primary_emotion": "one of [happy, sad, angry, anxious, neutral, hopeful, fearful, confused]",
            "emotion_justification": "why you believe this is the primary emotion",
            "crisis_level": "one of [low, medium, high, very_high]",
            "crisis_justification": "why you assessed this crisis level",
            "needs_immediate_resources": true/false,
            "reasoning": "your step-by-step reasoning process"
        }
        
        Return ONLY the valid JSON object without any other text.
        """),
        HumanMessage(content=f"Message to analyze: {latest_user_msg}")
    ])
    
    formatted_messages = emotion_analysis_prompt.format_messages()
    response = llm.invoke(formatted_messages)
    
    # Extract JSON from the response
    import json
    import re
    
    # Try to extract JSON content
    try:
        # First try to parse the whole response as JSON
        parsed_response = json.loads(response.content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON using regex
        json_match = re.search(r'```json\s*(.*?)\s*```|{.*}', response.content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                parsed_response = json.loads(json_str)
            except (json.JSONDecodeError, AttributeError):
                # Fallback values if JSON parsing fails
                parsed_response = {
                    "primary_emotion": "neutral",
                    "emotion_justification": "Unable to determine emotion from message",
                    "crisis_level": "low",
                    "crisis_justification": "No clear crisis indicators detected",
                    "needs_immediate_resources": False,
                    "reasoning": "Analysis encountered technical difficulties"
                }
        else:
            # Fallback values if no JSON found
            parsed_response = {
                "primary_emotion": "neutral",
                "emotion_justification": "Unable to determine emotion from message",
                "crisis_level": "low",
                "crisis_justification": "No clear crisis indicators detected",
                "needs_immediate_resources": False,
                "reasoning": "Analysis encountered technical difficulties"
            }
    
    # Save the analysis to state
    state["emotion_analysis"] = parsed_response
    
    # Set crisis flag if needed
    if parsed_response.get("needs_immediate_resources", False) or parsed_response.get("crisis_level") in ["high", "very_high"]:
        state["immediate_resources_needed"] = True
    
    # Add reasoning as a function message if reasoning is visible
    new_messages = []  # Create a new list for messages to add
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Live Thinking Process**
        
        Let me think about what you've shared...
        
        First, I notice the language patterns in your message:
        - *analyzing key words and emotional cues*
        - *identifying tone and urgency*
        
        Looking at how you've expressed yourself, I sense that your primary emotion is **{parsed_response.get('primary_emotion', 'unclear')}**.
        
        I'm noticing this because: {parsed_response.get('emotion_justification', '')}
        
        Considering the overall context and intensity of your message, I would assess this as a **{parsed_response.get('crisis_level', 'low')}** level situation.
        
        {parsed_response.get('reasoning', '')}
        
        Now, let me think about how best to respond...
        """
        
        new_messages.append(
            FunctionMessage(
                name="thinking_process", 
                content=reasoning_message
            )
        )
    
    # Return updates to state
    return {
        "emotion_analysis": parsed_response,
        "immediate_resources_needed": parsed_response.get("needs_immediate_resources", False) or 
                                      parsed_response.get("crisis_level") in ["high", "very_high"],
        "messages": new_messages
    }

def determine_response_strategy(state: AgentState) -> Dict:
    """Determine the best response strategy based on emotional analysis."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
    crisis_level = emotion_analysis.get("crisis_level", "low")
    
    # Create a prompt to determine response strategy
    strategy_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a mental health professional deciding how to respond to someone.
        
        Based on the emotional analysis and message context, determine the best response strategy.
        
        Return a JSON object with the following structure:
        {{
            "approach": "emotional approach to take (empathize, validate, encourage, educate, etc)",
            "key_points": ["specific point to address", "another point to address"],
            "appropriate_tools": ["any tools that would help like 'mental_health_database', 'web_search', 'wikipedia', 'youtube_videos'"],
            "reasoning": "your step-by-step reasoning process"
        }}
        
        Return ONLY the valid JSON object without any other text.
        """),
        HumanMessage(content=f"""
        User message: {latest_user_msg}
        
        Emotional analysis:
        - Primary emotion: {primary_emotion}
        - Crisis level: {crisis_level}
        
        What would be the best response strategy?
        """)
    ])
    
    formatted_messages = strategy_prompt.format_messages()
    response = llm.invoke(formatted_messages)
    
    # Extract JSON from the response
    import json
    import re
    
    # Try to extract JSON content using the same approach as before
    try:
        parsed_response = json.loads(response.content)
    except json.JSONDecodeError:
        json_match = re.search(r'```json\s*(.*?)\s*```|{.*}', response.content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
                parsed_response = json.loads(json_str)
            except (json.JSONDecodeError, AttributeError):
                parsed_response = {
                    "approach": "empathize",
                    "key_points": ["Acknowledge feelings", "Offer support"],
                    "appropriate_tools": [],
                    "reasoning": "Defaulting to empathetic approach due to technical issue"
                }
        else:
            parsed_response = {
                "approach": "empathize",
                "key_points": ["Acknowledge feelings", "Offer support"],
                "appropriate_tools": [],
                "reasoning": "Defaulting to empathetic approach due to technical issue"
            }
    
    # Save the strategy to state
    state["response_strategy"] = parsed_response
    
    # Add reasoning as a function message if reasoning is visible
    new_messages = []
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Continuing My Thought Process**
        
        Based on your emotional state, I need to decide how to structure my response...
        
        Given that you're feeling **{emotion_analysis.get('primary_emotion', 'neutral')}**, I think the most helpful approach would be **{parsed_response.get('approach', 'empathy')}**.
        
        I should focus on:
        {', '.join(parsed_response.get('key_points', ['Providing support']))}
        
        {parsed_response.get('reasoning', '')}
        
        Let me gather any information that might help...
        """
        
        new_messages.append(
            FunctionMessage(
                name="thinking_process", 
                content=reasoning_message
            )
        )
    
    # Return updates to state
    return {
        "response_strategy": parsed_response,
        "messages": new_messages
    }

def provide_crisis_resources(state: AgentState) -> Dict:
    """Provide immediate crisis resources for high crisis situations."""
    if state.get("immediate_resources_needed", False):
        crisis_message = """
        I notice that you might be going through something really difficult right now. Your wellbeing is important, and there are people available to talk to you right away:

        - National Suicide Prevention Lifeline: 988 or 1-800-273-8255 (Available 24/7)
        - Crisis Text Line: Text HOME to 741741 (Available 24/7)
        - If you're in immediate danger, please call emergency services (911 in the US)

        Would you like me to continue providing support alongside these resources?
        """
        
        state["messages"].append(AIMessage(content=crisis_message))
    
    return state

def route_to_tools(state: AgentState) -> Literal["use_tools", "generate_response"]:
    """Determine whether to use tools or just generate a response."""
    if not state.get("response_strategy"):
        return "generate_response"
    
    # Check if tools were recommended in the strategy
    appropriate_tools = state["response_strategy"].get("appropriate_tools", [])
    
    if appropriate_tools and len(appropriate_tools) > 0:
        return "use_tools"
    else:
        return "generate_response"

def select_and_use_tools(state: AgentState) -> Dict:
    """Select and use relevant tools based on the routing decision and user's needs."""
    messages = state["messages"]
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    response_strategy = state.get("response_strategy", {})
    query_route = state.get("query_route", "general_advice")
    
    # Get user preferences
    user_preferences = state.get("user_preferences", {})
    tool_results = {}
    new_messages = []
    
    # Prioritize tools based on the routing decision
    if query_route == "video_resources" or "youtube_videos" in response_strategy.get("appropriate_tools", []) or "video" in latest_user_msg.lower():
        try:
            # Search for videos with priority
            videos = search_mental_health_videos(latest_user_msg)
            tool_results["youtube_videos"] = videos
            
            # For the first video, determine if we should create a blog
            if videos and "youtube.com/watch" in videos:
                first_video = videos.split('\n')[0]
                
                # Check if user wants a blog summary
                if "blog" in latest_user_msg.lower() or "summarize" in latest_user_msg.lower() or user_preferences.get("prefers_detailed_content", False):
                    blog_content = generate_video_blog(first_video)
                    tool_results["video_blog"] = blog_content
                    
                    # Add a message indicating blog creation
                    if state.get("reasoning_visible", True):
                        blog_creation_message = f"""
                        💭 **Creating Blog Post from Video**
                        
                        I'm generating a comprehensive blog post from the video that covers this topic.
                        
                        This will provide you with a text version of the key insights that you can read at your own pace.
                        
                        Creating blog now...
                        """
                        
                        new_messages.append(
                            FunctionMessage(
                                name="blog_creation",
                                content=blog_creation_message
                            )
                        )
                else:
                    # Just get the transcript and summary if no blog requested
                    video_content = get_youtube_transcript_and_summary(first_video)
                    tool_results["youtube_content"] = video_content
        except Exception as e:
            tool_results["youtube_error"] = str(e)
    
    if query_route == "knowledge_base" or any(tool in response_strategy.get("appropriate_tools", []) for tool in ["mental_health_info", "knowledge"]):
        # Prioritize our knowledge base
        tool_results["mental_health_info"] = search_mental_health_info(latest_user_msg)
    
    if query_route == "academic_research" or any(tool in response_strategy.get("appropriate_tools", []) for tool in ["research", "arxiv", "academic"]):
        # Prioritize academic sources
        tool_results["arxiv"] = arxiv_tool.invoke({"query": latest_user_msg})
    
    # Check for each possible tool in the recommended tools
    for tool_name in response_strategy.get("appropriate_tools", []):
        tool_name = tool_name.lower()
        
        try:
            if "web" in tool_name or "search" in tool_name or "internet" in tool_name:
                tool_results["web_search"] = tavily_search_tool.invoke(latest_user_msg)
            
            elif "wikipedia" in tool_name or "wiki" in tool_name:
                tool_results["wikipedia"] = wiki_tool.invoke({"query": latest_user_msg})
        
        except Exception as e:
            tool_results[f"error_{tool_name}"] = f"Error using {tool_name}: {str(e)}"
    
    # Add the tool results to the state
    state["tool_results"] = tool_results
    
    # Add tool results as function messages if reasoning is visible
    new_messages = []  # Create a new list for messages to add
    if state.get("reasoning_visible", True) and tool_results:
        tools_summary = "🔍 **Information Gathered**\n\n"
        
        for tool_name, result in tool_results.items():
            if tool_name == "youtube_content" and isinstance(result, dict):
                tools_summary += f"Found a relevant video: {result.get('title', 'Video')}\n"
            elif tool_name == "youtube_videos":
                tools_summary += f"Found relevant videos you might find helpful.\n"
            elif tool_name == "mental_health_info":
                tools_summary += f"From our mental health resources:\n{result[:200]}...\n\n"
            else:
                str_result = str(result)
                tools_summary += f"From {tool_name}:\n{str_result[:200]}...\n\n"
        
        new_messages.append(
            FunctionMessage(
                name="information_gathering", 
                content=tools_summary
            )
        )
    
    return {
        "tool_results": tool_results,
        "messages": new_messages
    }

def generate_response(state: AgentState) -> Dict:
    """Generate a response based on all available information."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    response_strategy = state.get("response_strategy", {})
    tool_results = state.get("tool_results", {})
    
    # Prepare response context
    emotion = emotion_analysis.get("primary_emotion", "neutral")
    approach = response_strategy.get("approach", "empathize")
    key_points = response_strategy.get("key_points", [])
    
    # Create a consolidated prompt with all information
    response_context = f"""
    Based on my understanding:
    - The user seems to be feeling: {emotion}
    - I should approach with: {approach}
    - Key points to address: {', '.join(key_points)}
    """
    
    # Add tool results if available
    if tool_results:
        response_context += "\n\nInformation gathered:\n"
        for tool_name, result in tool_results.items():
            if tool_name == "youtube_content" and isinstance(result, dict):
                response_context += f"\nVideo summary: {result.get('summary', 'No summary available')[:300]}...\n"
            elif tool_name == "video_blog" and isinstance(result, dict):
                response_context += f"\nBlog from video: {result.get('title')}\n"
                response_context += f"Key points: {', '.join(result.get('key_points', [])[:3])}\n"
                response_context += f"Blog content: {result.get('content', '')[:300]}...\n"
            elif tool_name == "youtube_videos":
                response_context += f"\nRelevant videos: {result[:150]}...\n"
            else:
                str_result = str(result)
                response_context += f"\n{tool_name}: {str_result[:200]}...\n"
    
    # Create the response prompt with instruction for brevity
    response_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_MESSAGE + f"""
        \n\n{response_context}
        
        IMPORTANT INSTRUCTIONS:
        1. Keep your response brief and focused (max 3-4 sentences)
        2. Be warm and empathetic but concise
        3. If you have videos or resources to share, include ONE most relevant link directly in your response
        4. Don't use unnecessary acknowledgments or extended explanations
        """),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="Provide a brief, helpful response. Be concise.")
    ])
    
    response_inputs = {
        "history": [msg for msg in messages if not isinstance(msg, FunctionMessage)]
    }
    
    formatted_messages = response_prompt.format_messages(**response_inputs)
    response = llm.invoke(formatted_messages)
    
    # Start with reasoning if enabled
    new_messages = []
    if state.get("reasoning_visible", True):
        final_reasoning = f"""
        💭 **Finalizing My Response**
        
        Now I'll craft a concise response that addresses your needs...
        
        I want to acknowledge your feeling of **{emotion}** while being supportive.
        
        I'll focus on: {", ".join(key_points) if key_points else "providing empathetic support"} 
        
        My approach will be: **{approach}**
        
        {"I've found some helpful resources that I'll share with you." if tool_results else "I'll focus on direct support for now."}
        
        Here's my response:
        """
        
        new_messages.append(
            FunctionMessage(
                name="thinking_process", 
                content=final_reasoning
            )
        )
    
    # Enhance the content with blog if available
    content = response.content
    
    # If we have a blog post, mention it and provide key points
    if "video_blog" in tool_results and isinstance(tool_results["video_blog"], dict):
        blog = tool_results["video_blog"]
        if "title" in blog and "content" in blog:
            # Add blog reference to the message unless it's already covered
            if "blog" not in content.lower():
                # Get the first video URL to reference
                video_url = ""
                if "youtube_videos" in tool_results and "youtube.com/watch" in tool_results["youtube_videos"]:
                    videos = tool_results["youtube_videos"].split("\n")
                    if videos and len(videos) > 0:
                        for video in videos[:1]:
                            if "youtube.com/watch" in video:
                                video_url = video
                
                # Add a concise reference to the blog with key insights
                blog_mention = f"\n\nI've created a blog summary from this video: {video_url}\n\nKey insights: "
                
                # Add 2-3 key points as bullets
                if blog.get("key_points"):
                    for point in blog.get("key_points")[:3]:
                        blog_mention += f"\n• {point.replace('•', '').replace('-', '').replace('*', '').strip()}"
                
                content += blog_mention
    
    new_messages.append(AIMessage(content=content))
    
    return {"messages": new_messages}

def update_user_preferences(state: AgentState) -> Dict:
    """Update user preferences based on their messages."""
    messages = state["messages"]
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content.lower()
    preferences = state.get("user_preferences", {})
    
    # Check for preferences in the message
    if "video" in latest_user_msg or "youtube" in latest_user_msg or "watch" in latest_user_msg:
        preferences["likes_videos"] = True
    
    if "brief" in latest_user_msg or "short" in latest_user_msg or "quick" in latest_user_msg:
        preferences["prefers_brief_responses"] = True
    
    if "resource" in latest_user_msg or "article" in latest_user_msg or "read" in latest_user_msg:
        preferences["wants_resources"] = True
        
    if "stop showing reasoning" in latest_user_msg or "hide reasoning" in latest_user_msg:
        state["reasoning_visible"] = False
        
    if "show reasoning" in latest_user_msg or "explain reasoning" in latest_user_msg:
        state["reasoning_visible"] = True
    
    if "blog" in latest_user_msg or "article" in latest_user_msg or "write-up" in latest_user_msg:
        preferences["prefers_detailed_content"] = True
    
    # Update preferences
    state["user_preferences"] = preferences
    
    return state

def explain_tool_selection(state: AgentState) -> Dict:
    """Explain the reasoning behind tool selection."""
    if not state.get("response_strategy") or not state.get("reasoning_visible", True):
        return state
        
    appropriate_tools = state["response_strategy"].get("appropriate_tools", [])
    
    if appropriate_tools:
        tools_reasoning = f"""
        💭 **Searching for Helpful Information**
        
        I think some additional information would be helpful here...
        
        Let me search for: {', '.join(appropriate_tools)}
        
        This should help address your question about {[msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content[:30]}...
        
        Searching...
        """
        
        return {
            "messages": [
                FunctionMessage(
                    name="thinking_process", 
                    content=tools_reasoning
                )
            ]
        }
    
    return {"messages": []}

def track_mood(state: AgentState) -> Dict:
    """Track user's mood over time and identify patterns."""
    messages = state["messages"]
    mood_history = state.get("mood_history", [])
    emotion_analysis = state.get("emotion_analysis", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # Check if this appears to be a mood check-in
    is_mood_checkin = False
    mood_indicators = ["feeling", "mood", "today", "doing", "am i", "how are you"]
    if any(indicator in latest_user_msg.lower() for indicator in mood_indicators):
        is_mood_checkin = True
    
    # If we have emotion analysis, record it in mood history
    if emotion_analysis and "primary_emotion" in emotion_analysis:
        import datetime
        
        current_mood = {
            "timestamp": datetime.datetime.now().isoformat(),
            "emotion": emotion_analysis.get("primary_emotion", "neutral"),
            "intensity": emotion_analysis.get("crisis_level", "low"),
            "context": latest_user_msg[:100],  # Brief context from message
            "is_checkin": is_mood_checkin
        }
        
        mood_history.append(current_mood)
        
        # Limit history size (keep last 30 entries)
        if len(mood_history) > 30:
            mood_history = mood_history[-30:]
    
    # Generate mood insights if we have enough data
    new_messages = []
    if len(mood_history) >= 3 and is_mood_checkin:
        # Analyze patterns
        recent_moods = mood_history[-5:]  # Last 5 moods
        common_emotions = {}
        
        for mood in recent_moods:
            emotion = mood["emotion"]
            common_emotions[emotion] = common_emotions.get(emotion, 0) + 1
        
        # Find most common emotion
        most_common = max(common_emotions.items(), key=lambda x: x[1])
        
        if most_common[1] >= 2:  # If an emotion appears at least twice
            insight_message = f"""
            💭 **Mood Tracking Insight**
            
            I've noticed that you've mentioned feeling **{most_common[0]}** {most_common[1]} times recently.
            
            Would you like to explore what might be contributing to this feeling?
            """
            
            new_messages.append(
                FunctionMessage(
                    name="mood_tracking",
                    content=insight_message
                )
            )
    
    return {
        "mood_history": mood_history,
        "messages": new_messages
    }

# Define specialized nodes for the new features
def process_mood_tracking(state: AgentState) -> Dict:
    """Process mood tracking requests and provide insights."""
    messages = state["messages"]
    mood_history = state.get("mood_history", [])
    emotion_analysis = state.get("emotion_analysis", {})
    
    # Track the current mood
    track_mood_result = track_mood(state)
    mood_history = track_mood_result.get("mood_history", mood_history)
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # Check if user is asking about mood patterns/history
    asking_about_patterns = any(x in latest_user_msg.lower() for x in ["pattern", "history", "trends", "tracking", "journal"])
    
    new_messages = []
    if asking_about_patterns and len(mood_history) > 3:
        # Generate more detailed mood insights
        insight_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are analyzing mood tracking data to provide helpful insights.
            Format your response as bullet points with 2-3 observations and a gentle question to explore further."""),
            HumanMessage(content=f"Mood history (most recent first): {str(mood_history[:10])}")
        ])
        
        formatted_messages = insight_prompt.format_messages()
        insight_response = llm.invoke(formatted_messages)
        
        new_messages.append(
            FunctionMessage(
                name="mood_insights",
                content=f"""
                💭 **Analyzing Your Mood Patterns**
                
                Looking at your recent mood entries, I notice:
                
                {insight_response.content}
                """
            )
        )
    
    return {
        "mood_history": mood_history, 
        "messages": new_messages
    }

def provide_cbt_exercise(state: AgentState) -> Dict:
    """Provide a CBT exercise based on the user's current emotional state."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    cbt_progress = state.get("cbt_progress", {"exercises_completed": [], "current_focus": None})
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
    
    # Get a CBT exercise recommendation
    exercise = suggest_cbt_exercise(primary_emotion, latest_user_msg)
    
    # Format the exercise as a message
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Selecting an Exercise For You**
        
        You appear to be feeling **{primary_emotion}**.
        
        The {exercise["name"]} technique could be helpful right now because it's designed to address {primary_emotion} thoughts and feelings.
        
        Let me provide you with clear instructions...
        """
        
        new_messages.append(
            FunctionMessage(
                name="cbt_reasoning",
                content=reasoning_message
            )
        )
    
    # Add the exercise as an assistant message
    exercise_message = f"""
    **{exercise["name"]}**
    
    {exercise["introduction"]}
    
    {exercise["description"]}
    
    **Steps:**
    """
    
    for i, step in enumerate(exercise["steps"], 1):
        exercise_message += f"\n{i}. {step}"
    
    exercise_message += "\n\nWould you like to try this exercise now? Or would you prefer something different?"
    
    # Update CBT progress
    cbt_progress["current_focus"] = exercise["name"]
    
    return {
        "cbt_progress": cbt_progress,
        "messages": new_messages + [AIMessage(content=exercise_message)]
    }

def provide_education(state: AgentState) -> Dict:
    """Provide psychoeducation on mental health topics."""
    messages = state["messages"]
    topics_covered = state.get("psychoeducation_topics_covered", [])
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # Identify the topic of interest
    topic_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Extract the main mental health topic from this query. Respond with just the topic name."),
        HumanMessage(content=latest_user_msg)
    ])
    
    formatted_messages = topic_prompt.format_messages()
    topic_response = llm.invoke(formatted_messages)
    topic = topic_response.content.strip().lower()
    
    # Get educational content
    education = provide_psychoeducation(topic)
    
    # Add to topics covered
    if topic not in topics_covered:
        topics_covered.append(topic)
    
    # Format the educational content
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Preparing Educational Information**
        
        You're asking about **{education["topic"]}**.
        
        I'll provide some evidence-based information about this topic in a clear, accessible way.
        
        Let me organize the key points for you...
        """
        
        new_messages.append(
            FunctionMessage(
                name="education_reasoning",
                content=reasoning_message
            )
        )
    
    # Create the education message
    education_message = f"""
    **{education["title"]}**
    
    {education["description"]}
    
    **Key Points:**
    """
    
    for point in education["key_points"]:
        education_message += f"\n• {point}"
    
    education_message += f"\n\n**Helpful Framework:** {education['framework']}\n\n**Management Strategies:**"
    
    for strategy in education["strategies"]:
        education_message += f"\n• {strategy}"
    
    education_message += "\n\nIs there anything specific about this topic you'd like to explore further?"
    
    return {
        "psychoeducation_topics_covered": topics_covered,
        "messages": new_messages + [AIMessage(content=education_message)]
    }

def provide_therapeutic_story(state: AgentState) -> Dict:
    """Generate and provide a therapeutic story based on the user's struggle."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # Identify the struggle from the message
    struggle_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Extract the main emotional struggle or challenge from this message.
        Return only a brief phrase (2-5 words) describing the core struggle, such as "fear of failure", 
        "social anxiety", or "coping with grief"."""),
        HumanMessage(content=latest_user_msg)
    ])
    
    formatted_messages = struggle_prompt.format_messages()
    struggle_response = llm.invoke(formatted_messages)
    struggle = struggle_response.content.strip()
    
    # Get additional context from the emotional analysis
    context = ""
    if emotion_analysis:
        primary_emotion = emotion_analysis.get("primary_emotion", "")
        justification = emotion_analysis.get("emotion_justification", "")
        context = f"They are feeling {primary_emotion}. {justification}"
    
    # Generate the therapeutic story - FIXED: Direct access to the function instead of using the tool
    # Bypass the callback system that's causing the AttributeError
    try:
        # Using direct access to story archetypes and LLM 
        # instead of going through the tool's invoke mechanism with callbacks
        story_archetypes = {
            "anxiety": {
                "metaphors": ["river", "storm", "mountain", "maze"],
                "themes": ["courage", "breathing", "perspective", "facing fears"],
                "heroes": ["explorer", "navigator", "climber", "guide"]
            },
            "depression": {
                "metaphors": ["winter", "dark cave", "fog", "heavy backpack"],
                "themes": ["light", "seasons", "rest", "connection"],
                "heroes": ["traveler", "gardener", "lighthouse keeper", "dawn bringer"]
            },
            "grief": {
                "metaphors": ["ocean", "changing tree", "bridge", "mosaic"],
                "themes": ["memory", "honoring", "integration", "transformation"],
                "heroes": ["memory keeper", "bridge builder", "weaver", "composer"]
            },
            "fear": {
                "metaphors": ["closed door", "shadow", "uncharted territory", "high cliff"],
                "themes": ["curiosity", "courage", "unknown", "stepping forward"],
                "heroes": ["door opener", "light bearer", "map maker", "cliff climber"]
            },
            "anger": {
                "metaphors": ["fire", "storm", "tangled knot", "pressure cooker"],
                "themes": ["power", "energy", "boundaries", "transformation"],
                "heroes": ["fire keeper", "weather worker", "untangler", "pressure release"]
            },
            "guilt": {
                "metaphors": ["heavy stone", "locked room", "repeating record", "mirror"],
                "themes": ["forgiveness", "compassion", "learning", "release"],
                "heroes": ["stone bearer", "key finder", "new composer", "truth seeker"]
            },
            "shame": {
                "metaphors": ["mask", "invisibility cloak", "cave", "wall"],
                "themes": ["authenticity", "acceptance", "belonging", "speaking"],
                "heroes": ["mask remover", "voice finder", "light bearer", "bridge builder"]
            },
            "identity": {
                "metaphors": ["river", "mirror", "garden", "tapestry"],
                "themes": ["growth", "integration", "discovery", "cultivation"],
                "heroes": ["gardener", "navigator", "weaver", "mirror holder"]
            },
            "perfectionism": {
                "metaphors": ["impossible mountain", "never-ending path", "unreachable star", "golden cage"],
                "themes": ["acceptance", "growth", "process", "good enough"],
                "heroes": ["path walker", "star counter", "cage opener", "good-enough finder"]
            },
            "failure": {
                "metaphors": ["blocked road", "fallen attempt", "closed door", "missed mark"],
                "themes": ["learning", "resilience", "redirection", "wisdom"],
                "heroes": ["path finder", "rising phoenix", "door creator", "arrow adjuster"]
            }
        }
        
        # Identify which archetype to use based on the struggle
        identified_archetype = None
        struggle_key = "fear"  # Default
        
        for struggle_type, archetype in story_archetypes.items():
            if struggle_type in struggle.lower():
                identified_archetype = archetype
                struggle_key = struggle_type
                break
        
        # Default to "fear" archetype if no match found
        if not identified_archetype:
            import random
            struggle_key = random.choice(list(story_archetypes.keys()))
            identified_archetype = story_archetypes[struggle_key]
        
        # Create a prompt to generate the therapeutic story
        story_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a therapeutic storyteller who creates healing narratives based on narrative therapy principles.
            
            Create a brief allegorical story (500-800 words) where the protagonist faces and learns to work with {struggle}. 
            The story should:
            
            1. Use metaphors related to: {', '.join(identified_archetype['metaphors'])}
            2. Incorporate themes of: {', '.join(identified_archetype['themes'])}
            3. Cast the protagonist as a type of: {', '.join(identified_archetype['heroes'])}
            4. Include these narrative therapy elements:
               - Externalization (separating the person from the problem)
               - Identifying unique outcomes (exceptions to the problem's influence)
               - Thickening the preferred narrative (strengthening the story of capability)
            5. End with a sense of hope and new possibilities
            6. Include 3-4 reflection questions to help the reader connect the story to their own experience
            
            The story should be engaging, conversational, and avoid being preachy or overly simplistic.
            Write in second person ("you") to help the reader identify as the protagonist.
            
            Format your response as:
            
            # TITLE
            
            STORY CONTENT
            
            ## Reflection Questions:
            1. First question
            2. Second question
            3. Third question
            """),
            HumanMessage(content=f"Create a therapeutic story for someone struggling with {struggle}. Additional context about their situation: {context}")
        ])
        
        # Generate the story
        formatted_messages = story_prompt.format_messages()
        story_response = llm.invoke(formatted_messages)
        content = story_response.content
        
        # Parse the content
        title = "Your Therapeutic Story"
        story_body = content
        reflection_questions = []
        
        # Try to extract the title
        import re
        title_match = re.search(r'#\s*(.*?)\n', content)
        if title_match:
            title = title_match.group(1).strip()
            story_body = content.replace(title_match.group(0), "", 1)
        
        # Try to extract reflection questions
        questions_section = re.search(r'##\s*Reflection Questions:?([\s\S]*?)($|#)', content + "#")
        if questions_section:
            questions_text = questions_section.group(1)
            # Find numbered questions
            question_matches = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n##|\n#|$)', questions_text + "\n##")
            if question_matches:
                reflection_questions = [q.strip() for q in question_matches]
            story_body = content.replace(questions_section.group(0), "", 1)
        
        # Ensure we have at least some reflection questions
        if not reflection_questions:
            reflection_questions = [
                f"How does this story relate to your experience with {struggle}?",
                "What part of the story resonates most with you, and why?",
                "What new perspective might this story offer about your situation?"
            ]
        
        # Create therapeutic metaphor explanation
        metaphor_explanation = f"""
        In narrative therapy, metaphors help externalize problems, allowing us to see them as separate from our identity.
        
        This story uses the metaphor of {identified_archetype['metaphors'][0]} to help reframe your experience with {struggle}.
        
        By seeing yourself as the {identified_archetype['heroes'][0]} in this journey, you can draw on your inherent strengths of {identified_archetype['themes'][0]} and {identified_archetype['themes'][1]}.
        """
        
        story_result = {
            "title": title,
            "story": story_body.strip(),
            "reflection_questions": reflection_questions,
            "metaphor_explanation": metaphor_explanation.strip(),
            "struggle_addressed": struggle
        }
    except Exception as e:
        # Fallback if story generation fails
        story_result = {
            "title": "A Story About Resilience",
            "story": f"Once upon a time, there was a person facing {struggle}. Through patience and self-compassion, they discovered new strengths within themselves and found a path forward.",
            "reflection_questions": [
                "What strengths have helped you in difficult times before?",
                "What small step might help you move forward?",
                "How might you show yourself compassion during this challenge?"
            ],
            "metaphor_explanation": "Sometimes our struggles are like storms - they pass with time, and we discover our resilience in weathering them.",
            "struggle_addressed": struggle
        }
    
    # Format the story as messages
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Crafting a Therapeutic Story for You**
        
        I'll create a narrative based on your struggle with **{struggle}**.
        
        Therapeutic stories can help externalize our challenges and see them from a new perspective.
        
        I'm using elements of narrative therapy to craft a story where you're the hero of your own journey...
        """
        
        new_messages.append(
            FunctionMessage(
                name="story_reasoning",
                content=reasoning_message
            )
        )
    
    # Create the story message
    story_message = f"""
    # {story_result["title"]}
    
    {story_result["story"]}
    
    ## Reflection Questions:
    """
    
    for i, question in enumerate(story_result["reflection_questions"], 1):
        story_message += f"\n{i}. {question}"
    
    story_message += f"\n\n## How This Story Relates to Your Situation:\n{story_result['metaphor_explanation']}"
    
    story_message += "\n\nHow does this story resonate with you? Is there a particular part that speaks to your experience?"
    
    return {
        "messages": new_messages + [AIMessage(content=story_message)]
    }

# Define QueryRouter class
class QueryRouter(BaseModel):
    """Route a user query to the most appropriate mental health resource."""
    route_to: Literal["knowledge_base", "crisis_resources", "video_resources", 
                      "academic_research", "general_advice", "mood_tracking", 
                      "cbt_exercise", "self_care", "psychoeducation", "therapeutic_story", 
                      "follow_up", "motivational", "reflective_listening", "resource_sharing"] = Field(
        ..., description="Route the query to the most appropriate resource."
    )

# Define a content classifier to better understand user input intentions
class ContentClassifier(BaseModel):
    """Classify the content of a user message to better route it."""
    content_type: Literal["question", "statement", "request_for_help", "emotional_expression", 
                          "sharing_experience", "follow_up", "greeting", "gratitude"] = Field(
        ..., description="The primary type of content in the user's message."
    )
    urgency_level: Literal["low", "medium", "high", "emergency"] = Field(
        ..., description="The urgency level of the user's message."
    )
    complexity: Literal["simple", "moderate", "complex"] = Field(
        ..., description="The complexity level of the user's query or concern."
    )
    emotional_tone: Literal["positive", "neutral", "negative", "mixed"] = Field(
        ..., description="The overall emotional tone of the message."
    )

# Create improved router prompt 
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing mental health queries to the most appropriate resources.
    
    Options (choose ONE that best fits):
    - knowledge_base: For factual questions about mental health topics
    - crisis_resources: For urgent situations where someone needs immediate help
    - video_resources: For requests for video content or visual learning
    - academic_research: For in-depth or research-based information
    - mood_tracking: For questions about tracking emotions or mood patterns
    - cbt_exercise: For requests about cognitive behavioral therapy techniques
    - self_care: For advice on personal well-being activities
    - psychoeducation: For educational content about mental health concepts
    - therapeutic_story: For requests about stories, metaphors, or narrative approaches to healing
    - follow_up: For messages that refer to or follow up on previous conversations
    - motivational: For when the user needs encouragement or motivation
    - reflective_listening: For when the user mainly shares feelings without asking a specific question
    - resource_sharing: For when the user specifically asks for resources, tools, or materials
    - general_advice: For general mental health advice that doesn't fit the above categories
    
    Key routing guidelines:
    - Route to crisis_resources for any messages suggesting harm to self or others
    - Route to therapeutic_story when the user describes a struggle they're trying to understand or process
    - Route to reflective_listening when the user mainly shares feelings without asking a specific question
    - Route to resource_sharing when there's a clear request for materials, tools, or external resources
    - Route to motivational when the message indicates a need for encouragement to take action
    
    Analyze the query thoroughly before deciding on the most appropriate category.
    """),
    ("human", "{query}")
])

# Create content classifier prompt
classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at classifying mental health messages to help provide appropriate responses.
    
    Analyze the message and classify it according to:
    
    1. Content Type:
       - question: Direct question seeking information
       - statement: Sharing information without a direct question
       - request_for_help: Explicitly asking for assistance
       - emotional_expression: Primarily expressing feelings
       - sharing_experience: Describing personal experiences
       - follow_up: Continuing a previous conversation
       - greeting: Simple hello or conversation starter
       - gratitude: Expressing thanks
    
    2. Urgency Level:
       - low: No time pressure, general interest
       - medium: Would like help soon but not immediately
       - high: Needs help very soon, showing distress
       - emergency: Crisis situation requiring immediate attention
    
    3. Complexity:
       - simple: Straightforward, single concern
       - moderate: Multiple related concerns or nuanced issue
       - complex: Multiple interrelated issues or complicated situation
    
    4. Emotional Tone:
       - positive: Hopeful, happy, grateful, content
       - neutral: Matter-of-fact, informational, calm
       - negative: Sad, anxious, angry, frustrated, fearful
       - mixed: Combination of different emotions
    
    Return your analysis as structured data only.
    """),
    ("human", "{query}")
])

# Create the routers with structured output
query_router = router_prompt | llm.with_structured_output(QueryRouter)
content_classifier = classifier_prompt | llm.with_structured_output(ContentClassifier)

def classify_content(state: AgentState) -> Dict:
    """Classify the content of the user message to better understand intent."""
    messages = state["messages"]
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # Classify the content using the content classifier
    try:
        classification = content_classifier.invoke({"query": latest_user_msg})
        return {
            "content_classification": {
                "content_type": classification.content_type,
                "urgency_level": classification.urgency_level,
                "complexity": classification.complexity,
                "emotional_tone": classification.emotional_tone
            }
        }
    except:
        # Default classification if classifier fails
        return {
            "content_classification": {
                "content_type": "statement",
                "urgency_level": "medium",
                "complexity": "moderate",
                "emotional_tone": "neutral"
            }
        }
    
def route_query(state: AgentState) -> Dict:
    """Route the query to the appropriate pathway based on content classification and message content."""
    messages = state["messages"]
    classification = state.get("content_classification", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    
    # First, check urgency level - override for emergencies
    if classification.get("urgency_level") == "emergency":
        return {"query_route": "crisis_resources"}
    
    # Special handling for greeting/simple interactions
    if classification.get("content_type") == "greeting":
        return {"query_route": "general_advice"}
    
    # Special handling for gratitude
    if classification.get("content_type") == "gratitude":
        return {"query_route": "reflective_listening"}
    
    # Route based on the content type and emotional tone
    content_type = classification.get("content_type", "statement")
    emotional_tone = classification.get("emotional_tone", "neutral")
    
    # Use the query router for specific routing
    try:
        # Add context about the classification to help the router
        enhanced_query = f"{latest_user_msg}\n\nClassification context: {content_type} with {emotional_tone} tone."
        routing_result = query_router.invoke({"query": enhanced_query})
        return {"query_route": routing_result.route_to}
    except:
        # Fallback routing based on classification if router fails
        if content_type == "emotional_expression" and emotional_tone == "negative":
            return {"query_route": "reflective_listening"}
        elif content_type == "sharing_experience":
            return {"query_route": "therapeutic_story"}
        elif content_type == "request_for_help":
            return {"query_route": "self_care"}
        elif content_type == "question":
            return {"query_route": "knowledge_base"}
        else:
            return {"query_route": "general_advice"}

def provide_reflective_listening(state: AgentState) -> Dict:
    """Provide empathetic reflective listening response to validate user feelings."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
    
    # Create a reflective listening prompt
    reflection_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a mental health assistant skilled in reflective listening.
        
        Create a response that:
        1. Mirrors back the person's feelings with validation
        2. Shows you understand their experience
        3. Acknowledges the difficulty or importance of what they've shared
        4. Offers gentle support without pushing advice
        5. Ends with a small opening for them to share more if they wish
        
        Keep your response concise, warm, and focused on their feelings.
        """),
        HumanMessage(content=f"""
        The person shared: "{latest_user_msg}"
        
        Their primary emotion appears to be: {primary_emotion}
        
        Create a reflective listening response:
        """)
    ])
    
    # Generate the reflection
    formatted_messages = reflection_prompt.format_messages()
    reflection_response = llm.invoke(formatted_messages)
    
    # Format as messages
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Using Reflective Listening**
        
        I notice you're expressing your feelings and experiences.
        
        In this moment, what seems most helpful is to validate what you're feeling and show I understand.
        
        I'll focus on mirroring back what I hear, without rushing to solutions or advice.
        """
        
        new_messages.append(
            FunctionMessage(
                name="reflection_reasoning",
                content=reasoning_message
            )
        )
    
    # Add the reflection as an assistant message
    new_messages.append(AIMessage(content=reflection_response.content))
    
    return {"messages": new_messages}

def provide_motivational_response(state: AgentState) -> Dict:
    """Provide a motivational response to encourage the user."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
    
    # Create a motivational prompt
    motivation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a compassionate mental health assistant skilled in motivation.
        
        Create an encouraging response that:
        1. Acknowledges the person's current feelings
        2. Validates their effort or courage in seeking support
        3. Offers specific encouragement related to their situation
        4. Provides a small, achievable step they might consider
        5. Emphasizes their existing strengths or resources
        
        Keep your response warm, genuine, and focused on hope without dismissing difficulties.
        """),
        HumanMessage(content=f"""
        The person shared: "{latest_user_msg}"
        
        Their primary emotion appears to be: {primary_emotion}
        
        Create a motivational response:
        """)
    ])
    
    # Generate the motivation
    formatted_messages = motivation_prompt.format_messages()
    motivation_response = llm.invoke(formatted_messages)
    
    # Format as messages
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Creating Motivational Support**
        
        I notice you may benefit from some encouragement and validation.
        
        I'll focus on acknowledging your feelings while highlighting your strengths.
        
        My goal is to offer gentle encouragement without minimizing challenges.
        """
        
        new_messages.append(
            FunctionMessage(
                name="motivation_reasoning",
                content=reasoning_message
            )
        )
    
    # Add the motivation as an assistant message
    new_messages.append(AIMessage(content=motivation_response.content))
    
    return {"messages": new_messages}

def provide_resource_sharing(state: AgentState) -> Dict:
    """Share specific mental health resources based on user needs."""
    messages = state["messages"]
    emotion_analysis = state.get("emotion_analysis", {})
    classification = state.get("content_classification", {})
    
    # Get the latest user message
    if not any(isinstance(msg, HumanMessage) for msg in messages):
        return state
    
    latest_user_msg = [msg for msg in messages if isinstance(msg, HumanMessage)][-1].content
    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
    
    # Sample resources by category - these would be more comprehensive in production
    bangladesh_resources = {
        "general": [
            "National Institute of Mental Health (NIMH) Bangladesh: 01713-200300",
            "Bangladesh Mental Health Foundation: www.bmhf.org.bd",
            "Kaan Pete Roi (emotional support helpline): 01779-554391, 01688-709965",
            "Bangladesh Association of Psychiatrists: http://bap.org.bd/"
        ],
        "depression": [
            "Depression support groups at Dhaka Community Hospital",
            "BRAC Mental Health Program: https://www.brac.net/",
            "PDF Guide: 'Understanding Depression in Bangladesh Context'"
        ],
        "anxiety": [
            "Anxiety management workshops at Psychological Health & Wellness Clinic (Dhaka)",
            "Breathing & Mindfulness resources in Bengali: mentalhealthbd.org/anxiety",
            "Anxiety support forum: bangladesh-anxiety-support.groups.io"
        ],
        "stress": [
            "Work-Life Balance workshops by LifeSpring Counseling Center",
            "Stress Management techniques in Bengali - downloadable audio",
            "Virtual stress support groups (weekly via Zoom)"
        ]
    }
    
    # Identify which resources to share
    resource_category = "general"
    if "depress" in primary_emotion.lower() or "sad" in primary_emotion.lower():
        resource_category = "depression"
    elif "anx" in primary_emotion.lower() or "worry" in primary_emotion.lower() or "fear" in primary_emotion.lower():
        resource_category = "anxiety"
    elif "stress" in primary_emotion.lower() or "overwhelm" in primary_emotion.lower():
        resource_category = "stress"
    
    resources = bangladesh_resources[resource_category]
    
    # Format as messages
    new_messages = []
    
    if state.get("reasoning_visible", True):
        reasoning_message = f"""
        💭 **Finding Relevant Resources**
        
        Based on your message, I think these {resource_category}-related resources in Bangladesh might be helpful.
        
        I'm selecting resources that are accessible and culturally appropriate for your context.
        
        These complement our conversation but don't replace professional support.
        """
        
        new_messages.append(
            FunctionMessage(
                name="resource_reasoning",
                content=reasoning_message
            )
        )
    
    # Create resource message
    resource_message = f"""
    **Bangladesh Mental Health Resources for {resource_category.capitalize()}**
    
    Here are some resources that might be helpful:
    
    """
    
    for resource in resources:
        resource_message += f"• {resource}\n"
    
    resource_message += """
    
    Would you like more specific information about any of these resources? Or would you prefer resources for a different concern?
    
    Remember that while these resources can be helpful, they're not a substitute for professional mental health support when needed.
    """
    
    # Add the resources as an assistant message
    new_messages.append(AIMessage(content=resource_message))
    
    return {"messages": new_messages}

# Enhanced choose_pathway to handle more route options
def choose_pathway(state: AgentState) -> Literal[
    "crisis_path", 
    "knowledge_path", 
    "video_path", 
    "research_path", 
    "general_path",
    "mood_tracking_path",
    "cbt_path",
    "self_care_path",
    "education_path",
    "story_path",
    "reflective_path",
    "motivational_path",
    "resource_path",
    "follow_up_path"
]:
    """Choose the appropriate pathway based on routing result and content classification."""
    route = state.get("query_route", "general_advice")
    
    if route == "crisis_resources":
        return "crisis_path"
    elif route == "knowledge_base":
        return "knowledge_path"
    elif route == "video_resources":
        return "video_path"
    elif route == "academic_research":
        return "research_path"
    elif route == "mood_tracking":
        return "mood_tracking_path"
    elif route == "cbt_exercise":
        return "cbt_path"
    elif route == "self_care":
        return "self_care_path"
    elif route == "psychoeducation":
        return "education_path"
    elif route == "therapeutic_story":
        return "story_path"
    elif route == "reflective_listening":
        return "reflective_path"
    elif route == "motivational":
        return "motivational_path"
    elif route == "resource_sharing":
        return "resource_path"
    elif route == "follow_up":
        return "follow_up_path"
    else:
        return "general_path"

# Enhanced get_route_destination to include new options
def get_route_destination(state: AgentState) -> str:
    """Map query_route to a valid destination key."""
    route = state.get("query_route", "general_advice")
    
    # Only return exact matches for these specific routes
    if route in ["cbt_exercise", "self_care", "psychoeducation", "therapeutic_story", 
                "reflective_listening", "motivational", "resource_sharing"]:
        # Convert route names to node names
        route_to_node = {
            "cbt_exercise": "provide_cbt",
            "self_care": "suggest_self_care",
            "psychoeducation": "provide_education",
            "therapeutic_story": "provide_therapeutic_story",
            "reflective_listening": "provide_reflective_listening",
            "motivational": "provide_motivational",
            "resource_sharing": "provide_resources"
        }
        return route_to_node.get(route, "default")
    else:
        # All other routes go to the default destination
        return "default"

# Now define the workflow with the enhanced structure
workflow = StateGraph(AgentState)

# Add all base nodes
workflow.add_node("initialize", initialize_state)
workflow.add_node("update_preferences", update_user_preferences)
workflow.add_node("classify_content", classify_content)
workflow.add_node("route_query", route_query)
workflow.add_node("analyze_emotion", analyze_emotion_and_needs)
workflow.add_node("determine_strategy", determine_response_strategy)
workflow.add_node("crisis_resources", provide_crisis_resources)
workflow.add_node("explain_tools", explain_tool_selection)
workflow.add_node("use_tools", select_and_use_tools)
workflow.add_node("generate_response", generate_response)

# Add feature nodes
workflow.add_node("process_mood", process_mood_tracking)
workflow.add_node("provide_cbt", provide_cbt_exercise)
workflow.add_node("suggest_self_care", suggest_self_care)
workflow.add_node("provide_education", provide_education)
workflow.add_node("provide_therapeutic_story", provide_therapeutic_story)

# Add new specialized nodes
workflow.add_node("provide_reflective_listening", provide_reflective_listening)
workflow.add_node("provide_motivational", provide_motivational_response)
workflow.add_node("provide_resources", provide_resource_sharing)

# Define the updated workflow path
workflow.add_edge(START, "initialize")
workflow.add_edge("initialize", "update_preferences")
workflow.add_edge("update_preferences", "classify_content")
workflow.add_edge("classify_content", "route_query")

# Content analysis pathway
workflow.add_conditional_edges(
    "route_query",
    choose_pathway,
    {
        "crisis_path": "analyze_emotion",
        "knowledge_path": "analyze_emotion",
        "video_path": "analyze_emotion",
        "research_path": "analyze_emotion",
        "mood_tracking_path": "analyze_emotion",
        "cbt_path": "analyze_emotion",
        "self_care_path": "analyze_emotion",
        "education_path": "analyze_emotion",
        "story_path": "analyze_emotion",
        "reflective_path": "analyze_emotion",
        "motivational_path": "analyze_emotion",
        "resource_path": "analyze_emotion",
        "follow_up_path": "analyze_emotion",
        "general_path": "analyze_emotion"
    }
)

# Main emotion analysis and strategy path
workflow.add_edge("analyze_emotion", "determine_strategy")

# From strategy determination to crisis resources check
workflow.add_edge("determine_strategy", "crisis_resources")

# Specialized routing after crisis resources check
workflow.add_conditional_edges(
    "crisis_resources",
    get_route_destination,
    {
        "provide_cbt": "provide_cbt",  
        "suggest_self_care": "suggest_self_care",
        "provide_education": "provide_education",
        "provide_therapeutic_story": "provide_therapeutic_story",
        "provide_reflective_listening": "provide_reflective_listening",
        "provide_motivational": "provide_motivational",
        "provide_resources": "provide_resources",
        "default": "explain_tools"  # All other routes
    }
)

# Special handling for mood tracking
workflow.add_conditional_edges(
    "crisis_resources",
    lambda state: "mood_tracking" if state.get("query_route") == "mood_tracking" else "continue",
    {
        "mood_tracking": "process_mood",
        "continue": "continue_normal_flow"  # Dummy node for the conditional edge
    }
)
workflow.add_node("continue_normal_flow", lambda x: x)  # Identity function
workflow.add_edge("continue_normal_flow", "explain_tools")
workflow.add_edge("process_mood", "explain_tools")

# Connect all specialized nodes back to the main flow
workflow.add_edge("provide_cbt", "generate_response")
workflow.add_edge("suggest_self_care", "generate_response")
workflow.add_edge("provide_education", "generate_response")
workflow.add_edge("provide_therapeutic_story", "generate_response")
workflow.add_edge("provide_reflective_listening", "generate_response")
workflow.add_edge("provide_motivational", "generate_response")
workflow.add_edge("provide_resources", "generate_response")

# Tools selection and use
workflow.add_conditional_edges(
    "explain_tools",
    route_to_tools,
    {
        "use_tools": "use_tools",
        "generate_response": "generate_response"
    }
)

workflow.add_edge("use_tools", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()

# Function to interact with the agent
def chat_with_mental_health_assistant(user_input: str, state: Optional[Dict] = None) -> Dict:
    """Interact with the mental health assistant."""
    if state is None:
        state = {"messages": [SystemMessage(content=SYSTEM_MESSAGE)]}
    
    # Add the user message to the state
    state["messages"].append(HumanMessage(content=user_input))
    
    # Run the graph with the updated state
    result = app.invoke(state)
    
    return result 