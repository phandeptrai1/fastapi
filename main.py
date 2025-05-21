import os
import logging
import json
from typing import List, Optional
from datetime import datetime, timezone
from functools import wraps
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, validator
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from redis import asyncio as aioredis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Pro Chat + Karaoke API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI not set")
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client.chat_app
        self.contacts = self.db.contacts
        self.messages = self.db.messages
        self.karaoke_songs = self.db.karaoke_songs
        self.karaoke_lyrics = self.db.karaoke_lyrics

    async def init_indexes(self):
        await self.contacts.create_index("id")
        await self.messages.create_index("contact_id")
        await self.karaoke_songs.create_index("videoId", unique=True)
        await self.karaoke_lyrics.create_index("videoId", unique=True)

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

class KaraokeSong(BaseModel):
    videoId: str
    title: str
    artist: str
    thumbnail: Optional[str] = None
    mood: Optional[str] = None  # Thêm trường mood

    @validator("thumbnail", always=True)
    def default_thumb(cls, v, values):
        return v or f"https://i.ytimg.com/vi/{values.get('videoId')}/hqdefault.jpg"

# Startup / Shutdown
@app.on_event("startup")
async def startup():
    global redis, mongo
    try:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            redis = aioredis.from_url(redis_url, decode_responses=True)
            await redis.ping()
            await FastAPILimiter.init(redis)
            logger.info("✅ Redis initialized")
    except Exception as e:
        logger.warning(f"Redis init failed: {e}")
        redis = None

    mongo = MongoDBConnection()
    await mongo.init_indexes()

@app.on_event("shutdown")
async def shutdown():
    mongo.client.close()
    if redis:
        await redis.close()

@app.get("/")
async def root():
    return {"message": "✅ API Ready"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

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
    if len(active_websockets[contact_id]) >= MAX_WEBSOCKETS:
        await ws.close(code=1008, reason="Too many connections")
        return
    await ws.accept()
    active_websockets[contact_id].append(ws)
    try:
        while True:
            text = await ws.receive_text()
            msg = {
                "role": "user",
                "content": text,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await mongo.messages.update_one({"contact_id": contact_id}, {"$push": {"messages": msg}}, upsert=True)
            for conn in active_websockets[contact_id]:
                await conn.send_json({"contact_id": contact_id, "message": msg})
    except WebSocketDisconnect:
        active_websockets[contact_id].remove(ws)

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

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
