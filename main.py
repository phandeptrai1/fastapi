import os
import logging
import json
import re
import random
import asyncio
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from functools import wraps, lru_cache
from collections import defaultdict, Counter, deque
from enum import Enum, auto
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from pymongo import ASCENDING, DESCENDING

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('virtual_girlfriend.log')
    ]
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, validator, conlist
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId, json_util
from redis import asyncio as aioredis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Virtual Girlfriend AI API",
    description="API for interacting with your AI virtual girlfriend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global rate limiter
rate_limiter = RateLimiter(times=100, seconds=60)  # 100 requests per minute

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global state
redis = None
mongo = None
active_websockets = defaultdict(list)
MAX_WEBSOCKETS = 50

# Redis cache helper
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def redis_cached(ttl: int, namespace: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis:
                return await func(*args, **kwargs)
            try:
                key = f"{namespace}:{json.dumps(kwargs, sort_keys=True)}"
                cached = await redis.get(key)
                if cached:
                    return json.loads(cached)
                result = await func(*args, **kwargs)
                await redis.setex(key, ttl, json.dumps(result, cls=EnhancedJSONEncoder))
                return result
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# MongoDB setup
class MongoDBConnection:
    def __init__(self):
        try:
            uri = os.getenv("MONGODB_URI")
            if not uri:
                raise ValueError("MongoDB connection string is not set in environment variables")
            self.client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.client.get_database("chat_app")
            self.contacts = self.db.contacts
            self.messages = self.db.messages
            self.karaoke_songs = self.db.karaoke_songs
            self.karaoke_lyrics = self.db.karaoke_lyrics
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise
        
        # Collections for virtual girlfriend
        self.user_profiles = self.db.user_profiles
        self.chat_history = self.db.chat_history
        self.memories = self.db.memories
        self.mood_history = self.db.mood_history

    async def init_indexes(self):
        # Chat app indexes
        await self.contacts.create_index("id")
        await self.messages.create_index("contact_id")
        await self.karaoke_songs.create_index("videoId", unique=True)
        await self.karaoke_lyrics.create_index("videoId", unique=True)
        
        # Virtual girlfriend indexes
        await self.user_profiles.create_index("user_id", unique=True)
        await self.chat_history.create_index([("user_id", 1), ("timestamp", -1)])
        await self.memories.create_index([("user_id", 1), ("timestamp", -1)])
        await self.mood_history.create_index([("user_id", 1), ("timestamp", -1)])

# Models
class Message(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str
    timestamp: datetime

    @validator("timestamp", pre=True)
    def parse_time(cls, v):
        if isinstance(v, datetime):
            return v
        try:
            return datetime.fromisoformat(v)
        except:
            return datetime.strptime(v, "%H:%M %d/%m").replace(year=datetime.utcnow().year)

class SendMessage(Message):
    contactId: Optional[int] = None

class UploadMessages(BaseModel):
    messages: List[Message]
    contactId: Optional[int] = None

# AI Virtual Girlfriend Models
class MoodType(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    TIRED = "tired"
    ROMANTIC = "romantic"
    PLAYFUL = "playful"

class UserPreference(BaseModel):
    topics: Dict[str, float] = Field(default_factory=dict)  # Topic: Interest score (0-1)
    activities: Dict[str, int] = Field(default_factory=dict)  # Activity: Interaction count
    mood_patterns: Dict[str, List[str]] = Field(default_factory=dict)  # Mood: List of patterns
    
    def update_preferences(self, message: str, mood: MoodType):
        # Update topics based on message content
        words = self._preprocess_text(message)
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                self.topics[word] = min(1.0, self.topics.get(word, 0) + 0.05)
        
        # Update mood patterns
        if mood not in self.mood_patterns:
            self.mood_patterns[mood] = []
        self.mood_patterns[mood].extend(words)
        
        # Keep only top 20 topics
        self.topics = dict(sorted(self.topics.items(), key=lambda x: x[1], reverse=True)[:20])
        
    def _preprocess_text(self, text: str) -> List[str]:
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english') + stopwords.words('vietnamese'))
        return [word for word in tokens if word.isalpha() and word not in stop_words]

class MemoryItem(BaseModel):
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mood: Optional[MoodType] = None
    importance: float = Field(ge=0, le=1, default=0.5)  # 0 = forgettable, 1 = very important
    tags: List[str] = Field(default_factory=list)

class UserProfile(BaseModel):
    user_id: str
    name: str = ""
    age: Optional[int] = None
    gender: Optional[str] = None
    preferences: UserPreference = Field(default_factory=UserPreference)
    memories: List[MemoryItem] = Field(default_factory=list)
    mood_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)
    
    def add_memory(self, content: str, mood: MoodType = None, importance: float = 0.5, tags: List[str] = None):
        self.memories.append(MemoryItem(
            content=content,
            mood=mood,
            importance=importance,
            tags=tags or []
        ))
        # Keep only recent 100 memories
        self.memories = sorted(self.memories, key=lambda x: x.timestamp, reverse=True)[:100]
    
    def update_mood(self, mood: MoodType, confidence: float = 0.7):
        self.mood_history.append({
            "mood": mood,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        })
        # Keep mood history for last 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.mood_history = [m for m in self.mood_history if m["timestamp"] > cutoff]
    
    def get_current_mood(self) -> Optional[MoodType]:
        if not self.mood_history:
            return None
        # Get most recent mood with high confidence
        recent_mood = self.mood_history[-1]
        if recent_mood["confidence"] > 0.6:
            return recent_mood["mood"]
        return None

class KaraokeSong(BaseModel):
    videoId: str
    title: str
    artist: str
    thumbnail: Optional[str] = None
    mood: Optional[str] = None  # Thêm trường mood

    @validator("thumbnail", always=True)
    def default_thumb(cls, v, values):
        return v or f"https://i.ytimg.com/vi/{values.get('videoId')}/hqdefault.jpg"

# AI Virtual Girlfriend Core
class VirtualGirlfriend:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.user_profiles: Dict[str, UserProfile] = {}
        self._load_profiles()
    
    async def _load_profiles(self):
        """Load all user profiles from database into memory"""
        try:
            cursor = mongo.user_profiles.find({})
            async for doc in cursor:
                # Convert MongoDB document to UserProfile object
                profile_data = json_util.loads(json_util.dumps(doc))
                
                # Convert memory items
                memories_cursor = mongo.memories.find({"user_id": profile_data["user_id"]})\
                    .sort("timestamp", -1).limit(100)
                memory_items = []
                async for mem in memories_cursor:
                    mem_data = json_util.loads(json_util.dumps(mem))
                    memory_items.append(MemoryItem(
                        content=mem_data["content"],
                        timestamp=mem_data["timestamp"],
                        mood=MoodType(mem_data["mood"]) if mem_data.get("mood") else None,
                        importance=mem_data.get("importance", 0.5),
                        tags=mem_data.get("tags", [])
                    ))
                
                # Convert mood history
                mood_cursor = mongo.mood_history.find({"user_id": profile_data["user_id"]})\
                    .sort("timestamp", -1).limit(100)
                mood_history = []
                async for mood in mood_cursor:
                    mood_data = json_util.loads(json_util.dumps(mood))
                    mood_history.append({
                        "mood": MoodType(mood_data["mood"]),
                        "confidence": mood_data["confidence"],
                        "timestamp": mood_data["timestamp"]
                    })
                
                # Create UserProfile object
                profile = UserProfile(
                    user_id=profile_data["user_id"],
                    name=profile_data.get("name", ""),
                    age=profile_data.get("age"),
                    gender=profile_data.get("gender"),
                    preferences=UserPreference(
                        topics=profile_data.get("preferences", {}).get("topics", {}),
                        activities=profile_data.get("preferences", {}).get("activities", {}),
                        mood_patterns=profile_data.get("preferences", {}).get("mood_patterns", {})
                    ),
                    memories=memory_items,
                    mood_history=mood_history,
                    last_interaction=profile_data.get("last_interaction", datetime.utcnow())
                )
                
                self.user_profiles[profile_data["user_id"]] = profile
                
            logger.info(f"Loaded {len(self.user_profiles)} user profiles from database")
                
        except Exception as e:
            logger.error(f"Error loading user profiles: {str(e)}")
            # Start with empty profiles if there's an error
            self.user_profiles = {}
    
    async def _save_profiles(self):
        """Save all user profiles to database"""
        try:
            for user_id, profile in self.user_profiles.items():
                # Convert UserProfile to dict
                profile_dict = {
                    "user_id": profile.user_id,
                    "name": profile.name,
                    "age": profile.age,
                    "gender": profile.gender,
                    "preferences": {
                        "topics": profile.preferences.topics,
                        "activities": profile.preferences.activities,
                        "mood_patterns": profile.preferences.mood_patterns
                    },
                    "last_interaction": profile.last_interaction
                }
                
                # Save profile
                await mongo.user_profiles.update_one(
                    {"user_id": user_id},
                    {"$set": profile_dict},
                    upsert=True
                )
                
                # Save memories
                for memory in profile.memories:
                    memory_dict = {
                        "user_id": user_id,
                        "content": memory.content,
                        "timestamp": memory.timestamp,
                        "mood": memory.mood.value if memory.mood else None,
                        "importance": memory.importance,
                        "tags": memory.tags
                    }
                    await mongo.memories.update_one(
                        {"user_id": user_id, "content": memory.content[:200]},
                        {"$set": memory_dict},
                        upsert=True
                    )
                
                # Save mood history
                for mood in profile.mood_history:
                    mood_dict = {
                        "user_id": user_id,
                        "mood": mood["mood"].value,
                        "confidence": mood["confidence"],
                        "timestamp": mood["timestamp"]
                    }
                    await mongo.mood_history.insert_one(mood_dict)
            
            logger.info(f"Saved {len(self.user_profiles)} user profiles to database")
            
        except Exception as e:
            logger.error(f"Error saving user profiles: {str(e)}")
    
    def analyze_mood(self, text: str) -> MoodType:
        # Simple mood analysis based on keywords
        text = text.lower()
        positive_words = ['happy', 'joy', 'love', 'great', 'amazing', 'wonderful', 'excited']
        negative_words = ['sad', 'angry', 'hate', 'bad', 'terrible', 'awful', 'tired']
        
        pos_score = sum(1 for word in positive_words if word in text)
        neg_score = sum(1 for word in negative_words if word in text)
        
        if pos_score > neg_score:
            return MoodType.HAPPY
        elif neg_score > pos_score:
            return MoodType.SAD
        elif 'angry' in text or 'hate' in text:
            return MoodType.ANGRY
        elif 'tired' in text or 'exhausted' in text:
            return MoodType.TIRED
        elif 'love' in text or 'romantic' in text:
            return MoodType.ROMANTIC
        elif 'play' in text or 'fun' in text or 'game' in text:
            return MoodType.PLAYFUL
        else:
            return MoodType.CALM
    
    async def process_message(self, user_id: str, message: str) -> str:
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update last interaction time
        profile.last_interaction = datetime.utcnow()
        
        # Analyze mood
        mood = self.analyze_mood(message)
        profile.update_mood(mood)
        
        # Update preferences
        profile.preferences.update_preferences(message, mood)
        
        # Add to memory
        memory_content = f"User said: {message}"
        profile.add_memory(memory_content, mood=mood, importance=0.7)
        
        # Save chat history
        try:
            await mongo.chat_history.insert_one({
                "user_id": user_id,
                "role": "user",
                "content": message,
                "mood": mood.value,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
        
        # Generate response based on context
        response = self._generate_response(profile, message, mood)
        
        # Add AI response to chat history
        try:
            await mongo.chat_history.insert_one({
                "user_id": user_id,
                "role": "assistant",
                "content": response,
                "mood": mood.value,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Error saving AI response to chat history: {str(e)}")
        
        # Save updated profile (async)
        asyncio.create_task(self._save_profiles())
        
        return response
    
    def _generate_response(self, profile: UserProfile, message: str, mood: MoodType) -> str:
        # Simple response generation based on mood and context
        responses = {
            MoodType.HAPPY: [
                "I'm so happy to hear that! 😊",
                "That's wonderful news! What else makes you happy?",
                "Your happiness is contagious! 😄"
            ],
            MoodType.SAD: [
                "I'm sorry to hear you're feeling down. Do you want to talk about it?",
                "Sending you virtual hugs. What's on your mind?",
                "I'm here for you. What's making you feel this way?"
            ],
            MoodType.ANGRY: [
                "I can see you're upset. Take a deep breath. What happened?",
                "I'm here to listen. Do you want to talk about what's making you angry?"
            ],
            MoodType.TIRED: [
                "You sound exhausted. Have you taken a break recently?",
                "Sometimes a short break can help. Would you like to talk about something relaxing?"
            ],
            MoodType.ROMANTIC: [
                "That's so sweet of you to say! 💕",
                "You always know how to make me smile. 😊"
            ],
            MoodType.PLAYFUL: [
                "Hehe, you're so funny! 😄",
                "You always know how to make me laugh! 😂"
            ]
        }
        
        # Default responses if mood not found
        default_responses = [
            "Tell me more about that.",
            "That's interesting. Go on.",
            "I see. What else is on your mind?",
            "I understand. How does that make you feel?"
        ]
        
        mood_responses = responses.get(mood, default_responses)
        return random.choice(mood_responses)

# Initialize Virtual Girlfriend
girlfriend = VirtualGirlfriend()

# API Endpoints for Virtual Girlfriend
class ChatRequest(BaseModel):
    user_id: str
    message: str
    context: Optional[Dict[str, Any]] = None

class MemoryRequest(BaseModel):
    user_id: str
    content: str
    mood: Optional[MoodType] = None
    importance: float = 0.5
    tags: Optional[List[str]] = None

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

@app.post("/api/gf/chat", response_model=Dict[str, Any])
async def chat_with_gf(chat_request: ChatRequest):
    """
    Chat with your virtual girlfriend
    """
    try:
        response = await girlfriend.process_message(
            user_id=chat_request.user_id,
            message=chat_request.message
        )
        
        # Get user profile for additional context
        profile = girlfriend.user_profiles.get(chat_request.user_id)
        current_mood = profile.get_current_mood() if profile else None
        
        return {
            "response": response,
            "mood": current_mood.value if current_mood else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in chat_with_gf: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gf/memories")
async def add_memory(memory_request: MemoryRequest):
    """
    Add a memory for the user
    """
    try:
        if memory_request.user_id not in girlfriend.user_profiles:
            girlfriend.user_profiles[memory_request.user_id] = UserProfile(user_id=memory_request.user_id)
        
        profile = girlfriend.user_profiles[memory_request.user_id]
        profile.add_memory(
            content=memory_request.content,
            mood=memory_request.mood,
            importance=memory_request.importance,
            tags=memory_request.tags or []
        )
        
        await girlfriend._save_profiles()
        return {"status": "success", "message": "Memory added successfully"}
    except Exception as e:
        logger.error(f"Error adding memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gf/memories/{user_id}")
async def get_memories(user_id: str, limit: int = 10, offset: int = 0):
    """
    Get user's memories
    """
    try:
        if user_id not in girlfriend.user_profiles:
            return []
            
        profile = girlfriend.user_profiles[user_id]
        memories = profile.memories[offset:offset+limit]
        return [
            {
                "content": m.content,
                "mood": m.mood.value if m.mood else None,
                "timestamp": m.timestamp.isoformat(),
                "tags": m.tags
            }
            for m in memories
        ]
    except Exception as e:
        logger.error(f"Error getting memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gf/profile/{user_id}")
async def get_profile(user_id: str):
    """
    Get user profile
    """
    try:
        if user_id not in girlfriend.user_profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
            
        profile = girlfriend.user_profiles[user_id]
        current_mood = profile.get_current_mood()
        
        return {
            "user_id": profile.user_id,
            "name": profile.name,
            "age": profile.age,
            "gender": profile.gender,
            "current_mood": current_mood.value if current_mood else None,
            "top_topics": dict(sorted(profile.preferences.topics.items(), key=lambda x: x[1], reverse=True)[:5]),
            "last_interaction": profile.last_interaction.isoformat(),
            "memory_count": len(profile.memories)
        }
    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/gf/profile/{user_id}")
async def update_profile(user_id: str, profile_update: ProfileUpdate):
    """
    Update user profile
    """
    try:
        if user_id not in girlfriend.user_profiles:
            girlfriend.user_profiles[user_id] = UserProfile(user_id=user_id)
            
        profile = girlfriend.user_profiles[user_id]
        
        if profile_update.name is not None:
            profile.name = profile_update.name
        if profile_update.age is not None:
            profile.age = profile_update.age
        if profile_update.gender is not None:
            profile.gender = profile_update.gender
            
        profile.last_interaction = datetime.utcnow()
        await girlfriend._save_profiles()
        
        return {"status": "success", "message": "Profile updated successfully"}
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gf/mood-history/{user_id}")
async def get_mood_history(user_id: str, days: int = 7):
    """
    Get mood history for the user
    """
    try:
        if user_id not in girlfriend.user_profiles:
            return []
            
        profile = girlfriend.user_profiles[user_id]
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        return [
            {
                "mood": m["mood"].value,
                "confidence": m["confidence"],
                "timestamp": m["timestamp"].isoformat()
            }
            for m in profile.mood_history
            if m["timestamp"] > cutoff
        ]
    except Exception as e:
        logger.error(f"Error getting mood history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gf/suggest-topics/{user_id}")
async def suggest_topics(user_id: str, limit: int = 5):
    """
    Get suggested topics based on user preferences
    """
    try:
        if user_id not in girlfriend.user_profiles:
            return []
            
        profile = girlfriend.user_profiles[user_id]
        topics = sorted(profile.preferences.topics.items(), key=lambda x: x[1], reverse=True)
        
        # If not enough topics, add some defaults
        default_topics = ["music", "movies", "books", "travel", "food", "sports", "technology"]
        topic_set = {t[0] for t in topics}
        
        for topic in default_topics:
            if topic not in topic_set and len(topics) < limit:
                topics.append((topic, 0.5))  # Default medium interest
        
        return [{"topic": t[0], "score": t[1]} for t in topics[:limit]]
    except Exception as e:
        logger.error(f"Error suggesting topics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup / Shutdown
@app.on_event("startup")
async def startup():
    global redis, mongo, girlfriend
    
    # Initialize Redis with better configuration
    try:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.warning("REDIS_URL not set, Redis will be disabled")
        else:
            redis = aioredis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            await redis.ping()
            await FastAPILimiter.init(redis)
            logger.info("✅ Redis initialized")
    except Exception as e:
        logger.warning(f"Redis init failed: {e}")
        redis = None
    
    # Initialize MongoDB
    try:
        mongo = MongoDBConnection()
        await mongo.init_indexes()
        logger.info("✅ MongoDB initialized")
    except Exception as e:
        logger.error(f"MongoDB init failed: {e}")
        raise
    
    # Initialize Virtual Girlfriend
    try:
        await girlfriend._load_profiles()
        logger.info("✅ Virtual Girlfriend initialized")
    except Exception as e:
        logger.error(f"Virtual Girlfriend init failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    # Save all profiles before shutting down
    if 'girlfriend' in globals():
        await girlfriend._save_profiles()
    
    # Close database connections
    if 'mongo' in globals() and mongo:
        mongo.client.close()
    
    # Close Redis connection
    if 'redis' in globals() and redis:
        await redis.close()
    
    logger.info("Application shutdown complete")

@app.get("/")
async def root():
    return {"message": "✅ API Ready"}

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint that verifies all critical services
    """
    status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "mongodb": {
                "status": "ok",
                "error": None
            },
            "redis": {
                "status": "ok" if redis else "not configured",
                "error": None
            },
            "websockets": {
                "active_connections": sum(len(conns) for conns in active_websockets.values()),
                "active_rooms": len(active_websockets)
            }
        }
    }
    
    # Check MongoDB connection
    try:
        await mongo.client.admin.command('ping')
    except Exception as e:
        status["status"] = "degraded"
        status["services"]["mongodb"]["status"] = "error"
        status["services"]["mongodb"]["error"] = str(e)
    
    # Check Redis connection if configured
    if redis:
        try:
            await redis.ping()
        except Exception as e:
            status["status"] = "degraded"
            status["services"]["redis"]["status"] = "error"
            status["services"]["redis"]["error"] = str(e)
    
    return status

@app.get("/test-db")
async def test_db():
    await mongo.client.admin.command("ping")
    return {"db": "ok"}

@app.get("/test-redis")
async def test_redis():
    if redis:
        return {"redis": await redis.ping()}
    return {"redis": "not configured"}

@app.get("/get-contacts")
@redis_cached(60, "get_contacts")
async def get_contacts():
    return await mongo.contacts.find({}, {"_id": 0}).to_list(None)

def normalize_ts(ts):
    return ts.isoformat() if isinstance(ts, datetime) else str(ts)

@app.get("/get-messages")
@redis_cached(60, "get_messages")
async def get_messages(contactId: Optional[int] = Query(None), page: int = 1, limit: int = 100):
    query = {"contact_id": contactId} if contactId else {"contact_id": {"$exists": False}}
    docs = await mongo.messages.find(query).to_list(None)
    messages = [msg for doc in docs for msg in doc.get("messages", [])]
    messages.sort(key=lambda x: normalize_ts(x.get("timestamp", "")))
    return messages[(page - 1) * limit: page * limit]

@app.post("/send-message", dependencies=[Depends(RateLimiter(times=5, seconds=10))])
async def send_message(msg: SendMessage):
    query = {"contact_id": msg.contactId} if msg.contactId else {"contact_id": {"$exists": False}}
    await mongo.messages.update_one(query, {"$push": {"messages": msg.dict()}}, upsert=True)
    if msg.contactId:
        await mongo.contacts.update_one({"id": msg.contactId}, {
            "lastMessage": msg.content,
            "timestamp": msg.timestamp
        }, upsert=True)
    if redis:
        await redis.delete("get_messages")
    return {"message": "sent"}

@app.post("/upload-messages")
async def upload_messages(data: UploadMessages):
    query = {"contact_id": data.contactId} if data.contactId else {"contact_id": {"$exists": False}}
    await mongo.messages.update_one(query, {
        "$push": {"messages": {"$each": [m.dict() for m in data.messages]}}
    }, upsert=True)
    if redis:
        await redis.delete("get_messages")
    return {"message": f"Uploaded {len(data.messages)} messages"}

@app.delete("/clear-messages/{doc_id}")
async def clear_messages(doc_id: str):
    oid = ObjectId(doc_id)
    res = await mongo.messages.update_one({"_id": oid}, {"$set": {"messages": []}})
    if not res.matched_count:
        raise HTTPException(status_code=404, detail="Not found")
    if redis:
        await redis.delete("get_messages")
    return {"message": "cleared"}

@app.websocket("/ws/{contact_id}")
async def websocket_endpoint(ws: WebSocket, contact_id: int):
    logger.info(f"New WebSocket connection for contact {contact_id}")
    
    # Check max connections
    if len(active_websockets[contact_id]) >= MAX_WEBSOCKETS:
        logger.warning(f"Max connections reached for contact {contact_id}")
        await ws.close(code=1008, reason="Too many connections")
        return
    
    await ws.accept()
    active_websockets[contact_id].append(ws)
    logger.info(f"Active WebSockets for contact {contact_id}: {len(active_websockets[contact_id])}")
    
    try:
        while True:
            try:
                text = await ws.receive_text()
                logger.debug(f"Received message from contact {contact_id}: {text[:100]}...")
                
                msg = {
                    "role": "user",
                    "content": text,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Save to database
                await mongo.messages.update_one(
                    {"contact_id": contact_id}, 
                    {"$push": {"messages": msg}}, 
                    upsert=True
                )
                
                # Broadcast to all connections for this contact
                for conn in active_websockets[contact_id]:
                    try:
                        await conn.send_json({"contact_id": contact_id, "message": msg})
                    except Exception as e:
                        logger.error(f"Error sending to WebSocket: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in WebSocket handler: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for contact {contact_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket: {e}")
    finally:
        if contact_id in active_websockets and ws in active_websockets[contact_id]:
            active_websockets[contact_id].remove(ws)
            logger.info(f"Cleaned up WebSocket for contact {contact_id}")

# ========== Karaoke APIs ==========
@app.get("/api/songs", response_model=List[KaraokeSong])
async def get_karaoke_songs():
    return await mongo.karaoke_songs.find({}, {"_id": 0}).to_list(None)

@app.post("/api/songs")
async def add_karaoke_song(song: KaraokeSong):
    if await mongo.karaoke_songs.find_one({"videoId": song.videoId}):
        raise HTTPException(status_code=400, detail="Bài hát đã tồn tại")
    await mongo.karaoke_songs.insert_one(song.dict())
    return {"message": "Đã thêm bài hát"}

@app.post("/api/lyrics/upload/{video_id}")
async def upload_lyrics_file(video_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".srt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .srt")
    content = await file.read()
    lyrics_text = content.decode("utf-8")
    await mongo.karaoke_lyrics.update_one(
        {"videoId": video_id},
        {"$set": {"lyrics": lyrics_text}},
        upsert=True
    )
    return {"message": f"Đã lưu lời bài hát cho {video_id}"}

@app.get("/api/lyrics/{video_id}", response_class=PlainTextResponse)
async def get_lyrics(video_id: str):
    doc = await mongo.karaoke_lyrics.find_one({"videoId": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Không có lời bài hát")
    return doc["lyrics"]

@app.get("/chat")
async def chat_ui():
    """Simple HTML chat interface for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Virtual Girlfriend Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chatbox { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px; }
            #message { width: 80%; padding: 8px; }
            button { padding: 8px 15px; }
            .user { color: blue; margin: 5px 0; }
            .assistant { color: green; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>Virtual Girlfriend Chat</h1>
        <div id="chatbox"></div>
        <div>
            <input type="text" id="message" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
        <script>
            const userId = 'user_' + Math.random().toString(36).substr(2, 9);
            const chatbox = document.getElementById('chatbox');
            const messageInput = document.getElementById('message');
            
            function addMessage(role, content) {
                const div = document.createElement('div');
                div.className = role;
                div.textContent = (role === 'user' ? 'You: ' : 'AI: ') + content;
                chatbox.appendChild(div);
                chatbox.scrollTop = chatbox.scrollHeight;
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message to chat
                addMessage('user', message);
                messageInput.value = '';
                
                try {
                    const response = await fetch('/api/gf/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: userId,
                            message: message
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const data = await response.json();
                    addMessage('assistant', data.response);
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                }
            }
            
            // Allow sending message with Enter key
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Initial greeting
            addMessage('assistant', 'Hello! I\'m your virtual girlfriend. How can I make your day better?');
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    import argparse
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chat and Karaoke API Server')
    parser.add_argument('--host', type=str, default=os.getenv('HOST', '0.0.0.0'), 
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8000')), 
                      help='Port to run the server on')
    parser.add_argument('--reload', action='store_true', 
                      default=os.getenv('RELOAD', 'false').lower() == 'true',
                      help='Enable auto-reload')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    
    # Determine if we're in development mode
    is_dev = os.getenv('ENV', 'development') == 'development'
    
    # Configure and run the server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload and is_dev,  # Only reload in development
        log_level="debug" if is_dev else "info",
        workers=1 if is_dev else os.cpu_count(),  # Scale workers in production
        access_log=True,
        timeout_keep_alive=30  # Keep-alive timeout in seconds
    )
    )
