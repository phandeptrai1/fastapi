import os
import logging
import socket
import asyncio
import json
import time
import ssl
from typing import List, Optional
from datetime import datetime
from functools import wraps
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
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
dns_resolved = False
try:
    socket.gethostbyname(parsed_url.hostname)
    logger.info(f"DNS resolution thành công cho {parsed_url.hostname}")
    dns_resolved = True
except socket.gaierror as e:
    logger.warning(f"DNS resolution thất bại cho {parsed_url.hostname}: {e}. Vẫn thử kết nối Redis...")

# Biến global cho Redis
redis = None

# Custom Redis cache decorator
def redis_cached(ttl: int, namespace: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Tạo key từ namespace và kwargs
            key = f"{namespace}:{json.dumps(kwargs, sort_keys=True)}"
            # Kiểm tra cache
            cached_result = await redis.get(key)
            if cached_result:
                logger.info(f"Cache hit cho key: {key}")
                return json.loads(cached_result)
            # Gọi hàm nếu cache miss
            result = await func(*args, **kwargs)
            # Lưu vào cache
            await redis.setex(key, ttl, json.dumps(result))
            logger.info(f"Cache set cho key: {key} với TTL {ttl}s")
            return result
        return wrapper
    return decorator

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
    global redis
    try:
        await mongo.init_indexes()
        # Khởi tạo Redis với retry
        async def init_redis_with_retry(max_attempts=3, delay=2):
            # Tạo ssl_context để bỏ qua kiểm tra hostname và chứng chỉ
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2  # Đảm bảo TLS 1.2+
            for attempt in range(1, max_attempts + 1):
                try:
                    redis_client = aioredis.from_url(
                        redis_url,
                        decode_responses=True,
                        ssl_context=ssl_context
                    )
                    pong = await redis_client.ping()
                    logger.info(f"Kết nối Redis thành công: {pong}")
                    return redis_client
                except Exception as e:
                    logger.error(f"Thử kết nối Redis lần {attempt}/{max_attempts} thất bại: {e}")
                    if attempt < max_attempts:
                        logger.info(f"Thử lại sau {delay} giây...")
                        await asyncio.sleep(delay)
                    else:
                        raise Exception(f"Không thể kết nối Redis sau {max_attempts} lần thử: {e}")
        
        redis = await init_redis_with_retry()
        # Test TCP connection
        try:
            reader, writer = await asyncio.open_connection(parsed_url.hostname, parsed_url.port, ssl=True)
            writer.close()
            await writer.wait_closed()
            logger.info("Kết nối TCP tới Redis thành công")
        except Exception as e:
            logger.warning(f"Kết nối TCP tới Redis thất bại: {e}")
        # Khởi tạo FastAPILimiter
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        redis_for_limiter = aioredis.from_url(
            redis_url,
            decode_responses=True,
            ssl_context=ssl_context
        )
        await FastAPILimiter.init(redis_for_limiter)
        logger.info("Khởi tạo FastAPILimiter thành công")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo startup event: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    mongo.client.close()
    if redis:
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
@redis_cached(ttl=60, namespace="get_contacts")
async def get_contacts():
    return await mongo.contacts_collection.find({}, {"_id": 0}).to_list(None)

@app.get("/get-messages")
@redis_cached(ttl=60, namespace="get_messages")
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
