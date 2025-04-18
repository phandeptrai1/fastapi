import os
import logging
import socket
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer
from redis import asyncio as aioredis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from collections import defaultdict
from urllib.parse import urlparse

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("chat_api")

# FastAPI app init
app = FastAPI(title="Pro Chat API")

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis setup
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    logger.error("REDIS_URL không được set trong biến môi trường")
    raise ValueError("REDIS_URL không được set")

logger.info(f"Đang kết nối tới Redis tại: {redis_url.split('@')[-1]}")  # Ẩn password trong log
parsed_url = urlparse(redis_url)

# Kiểm tra DNS resolution
try:
    socket.gethostbyname(parsed_url.hostname)
    logger.info(f"DNS resolution thành công cho {parsed_url.hostname}")
except socket.gaierror as e:
    logger.error(f"DNS resolution thất bại cho {parsed_url.hostname}: {e}")
    raise

# Khởi tạo Redis
try:
    redis = aioredis.from_url(redis_url, decode_responses=True, ssl=True)
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo aioredis: {e}")
    raise

# Cache setup
try:
    cache = Cache(
        cache_class=Cache.REDIS,
        endpoint=parsed_url.hostname,
        port=parsed_url.port,
        password=parsed_url.password,
        username=parsed_url.username,
        ssl=True,
        serializer=JsonSerializer()
    )
    logger.info("Khởi tạo aiocache thành công")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo aiocache: {e}")
    raise

# WebSocket tracking
MAX_WEBSOCKETS = 50
active_websockets = defaultdict(list)

# MongoDB Singleton
class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            uri = os.getenv("MONGODB_URI")
            if not uri:
                raise ValueError("MONGODB_URI không được set")
            cls._client = AsyncIOMotorClient(uri)
            cls._db = cls._client["chat_app"]
            cls._contacts = cls._db["contacts"]
            cls._messages = cls._db["messages"]
        return cls._instance

    async def init_indexes(self):
        await self._contacts.create_index("id")
        await self._messages.create_index("contact_id")

    @property
    def client(self): return self._client
    @property
    def db(self): return self._db
    @property
    def contacts_collection(self): return self._contacts
    @property
    def messages_collection(self): return self._messages

mongo = MongoDBConnection()

# Pydantic Models
class Message(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str
    timestamp: datetime

    @validator("timestamp", pre=True)
    def parse_custom_timestamp(cls, value):
        if isinstance(value, datetime):
            return value
        try:
            dt = datetime.strptime(value, "%H:%M %d/%m")
            return dt.replace(year=datetime.utcnow().year)
        except ValueError:
            raise ValueError("timestamp phải có định dạng 'HH:mm dd/MM'")

class SendMessage(Message):
    contactId: Optional[int] = None

class UploadMessages(BaseModel):
    messages: List[Message]
    contactId: Optional[int] = None

# App startup
@app.on_event("startup")
async def startup_event():
    try:
        await mongo.init_indexes()
        # Test Redis connection
        pong = await redis.ping()
        logger.info(f"Kết nối Redis thành công: {pong}")
        redis_for_limiter = aioredis.from_url(redis_url, decode_responses=True, ssl=True)
        await FastAPILimiter.init(redis_for_limiter)
        logger.info("Khởi tạo FastAPILimiter thành công")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo startup event: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    mongo.client.close()
    await redis.close()
    logger.info("Đóng kết nối MongoDB và Redis")

@app.get("/")
async def home():
    return {"message": "🚀 Pro Chat API is running!"}

@app.get("/test-db")
async def test_db():
    try:
        await mongo.client.admin.command("ping")
        collections = await mongo.db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-redis")
async def test_redis():
    try:
        pong = await redis.ping()
        return {"message": "Redis connection OK", "response": pong}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

@app.get("/get-contacts")
@cached(ttl=60, cache=cache, namespace="get_contacts")
async def get_contacts():
    return await mongo.contacts_collection.find({}, {"_id": 0}).to_list(None)

@app.get("/get-messages")
@cached(ttl=60, cache=cache, namespace="get_messages")
async def get_messages(contactId: Optional[int] = Query(None), page: int = 1, limit: int = 100):
    query = {"contact_id": {"$exists": False}} if contactId is None else {"contact_id": contactId}
    docs = await mongo.messages_collection.find(query).to_list(None)
    messages = [msg for doc in docs for msg in doc.get("messages", [])]
    messages.sort(key=lambda x: x.get("timestamp", ""))
    start, end = (page - 1) * limit, page * limit
    return messages[start:end]

@app.get("/count-messages")
async def count_messages(contactId: Optional[int] = Query(None)):
    query = {"contact_id": {"$exists": False}} if contactId is None else {"contact_id": contactId}
    docs = await mongo.messages_collection.find(query).to_list(None)
    return {"contactId": contactId, "totalMessages": sum(len(doc.get("messages", [])) for doc in docs)}

@app.post("/send-message", dependencies=[Depends(RateLimiter(times=5, seconds=10))])
async def send_message(message: SendMessage):
    query = {"contact_id": message.contactId} if message.contactId else {"contact_id": {"$exists": False}}
    await mongo.messages_collection.update_one(query, {"$push": {"messages": message.dict()}}, upsert=True)
    if message.contactId:
        await mongo.contacts_collection.update_one({"id": message.contactId}, {"$set": {"lastMessage": message.content, "timestamp": message.timestamp}})
    await redis.delete("get_messages")
    return {"message": "Message sent"}

@app.post("/upload-messages")
async def upload_messages(data: UploadMessages):
    query = {"contact_id": data.contactId} if data.contactId else {"contact_id": {"$exists": False}}
    await mongo.messages_collection.update_one(query, {"$push": {"messages": {"$each": [msg.dict() for msg in data.messages]}}}, upsert=True)
    await redis.delete("get_messages")
    return {"message": f"Uploaded {len(data.messages)} messages"}

@app.delete("/clear-messages/{document_id}")
async def clear_messages(document_id: str):
    try:
        object_id = ObjectId(document_id)
        result = await mongo.messages_collection.update_one({"_id": object_id}, {"$set": {"messages": []}})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        await redis.delete("get_messages")
        return {"message": f"Cleared messages in document {document_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{contact_id}")
async def websocket_endpoint(websocket: WebSocket, contact_id: int):
    if len(active_websockets[contact_id]) >= MAX_WEBSOCKETS:
        await websocket.close(code=1008, reason="Too many connections")
        return
    await websocket.accept()
    active_websockets[contact_id].append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = {
                "role": "user",
                "content": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await mongo.messages_collection.update_one(
                {"contact_id": contact_id},
                {"$push": {"messages": message}},
                upsert=True
            )
            for ws in active_websockets[contact_id]:
                await ws.send_json({"contact_id": contact_id, "message": message})
    except WebSocketDisconnect:
        active_websockets[contact_id].remove(websocket)
    except Exception as e:
        active_websockets[contact_id].remove(websocket)
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
