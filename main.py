import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

MAX_WEBSOCKETS = 50
active_websockets = defaultdict(list)

class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            cls._client = AsyncIOMotorClient(uri, maxPoolSize=10, minPoolSize=1)
            cls._db = cls._client["chat_app"]
            cls._contacts = cls._db["contacts"]
            cls._messages = cls._db["messages"]
        return cls._instance

    async def init_indexes(self):
        await self._contacts.create_index([("id", 1)])
        await self._messages.create_index([("contact_id", 1)])
        logger.info("MongoDB indexes initialized.")

    @property
    def client(self): return self._client
    @property
    def db(self): return self._db
    @property
    def contacts_collection(self): return self._contacts
    @property
    def messages_collection(self): return self._messages

mongo = MongoDBConnection()

class Message(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str
    timestamp: str

class SendMessage(Message):
    contactId: Optional[int] = None

class UploadMessages(BaseModel):
    messages: List[Message]
    contactId: Optional[int] = None

@app.on_event("startup")
async def startup_event():
    await mongo.init_indexes()

@app.get("/")
async def home():
    return {"message": "🚀 FastAPI is running!"}

@app.get("/test-db")
async def test_db():
    try:
        await mongo.client.admin.command("ping")
        collections = await mongo.db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"DB test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-contacts")
@cached(ttl=60, cache=Cache.MEMORY)
async def get_contacts():
    try:
        contacts = await mongo.contacts_collection.find(
            {}, {"_id": 0, "id": 1, "name": 1, "lastMessage": 1, "timestamp": 1, "avatar": 1}
        ).to_list(None)
        return contacts
    except Exception as e:
        logger.error(f"Error fetching contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-messages")
@cached(ttl=60, cache=Cache.MEMORY, key_builder=lambda *args, **kwargs: f"get_messages_{kwargs['contact_id']}_{kwargs['page']}_{kwargs['limit']}")
async def get_messages(
    contact_id: Optional[int] = Query(None, alias="contactId"),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1)
):
    try:
        query = {"contact_id": {"$exists": False}} if contact_id is None else {"contact_id": contact_id}
        cursor = mongo.messages_collection.find(query, {"_id": 0, "messages": 1})
        docs = await cursor.to_list(length=None)

        all_messages = []
        for doc in docs:
            all_messages.extend(doc.get("messages", []))

        all_messages.sort(key=lambda x: x.get("timestamp", ""))
        start = (page - 1) * limit
        end = start + limit
        return all_messages[start:end]
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/count-messages")
async def count_messages(contact_id: Optional[int] = Query(None, alias="contactId")):
    try:
        query = {"contact_id": {"$exists": False}} if contact_id is None else {"contact_id": contact_id}
        cursor = mongo.messages_collection.find(query, {"_id": 0, "messages": 1})
        docs = await cursor.to_list(length=None)
        total = sum(len(doc.get("messages", [])) for doc in docs)
        return {"contactId": contact_id, "totalMessages": total}
    except Exception as e:
        logger.error(f"Error counting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send-message")
async def send_message(message: SendMessage):
    try:
        query = {"contact_id": message.contactId} if message.contactId else {"contact_id": {"$exists": False}}
        await mongo.messages_collection.update_one(query, {"$push": {"messages": message.dict()}}, upsert=True)

        if message.contactId:
            await mongo.contacts_collection.update_one(
                {"id": message.contactId},
                {"$set": {"lastMessage": message.content, "timestamp": message.timestamp}}
            )

        await cache.delete("get_contacts")
        if message.contactId:
            await cache.delete(f"get_messages_{message.contactId}_{1}_{100}")
        else:
            await cache.delete("get_messages_none_1_100")

        return {"message": "Message sent successfully"}
    except Exception as e:
        logger.error(f"Send message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-messages")
async def upload_messages(data: UploadMessages):
    try:
        messages = [msg.dict() for msg in data.messages]
        query = {"contact_id": data.contactId} if data.contactId else {"contact_id": {"$exists": False}}
        await mongo.messages_collection.update_one(query, {"$push": {"messages": {"$each": messages}}}, upsert=True)

        if data.contactId and messages:
            last_message = messages[-1]
            await mongo.contacts_collection.update_one(
                {"id": data.contactId},
                {"$set": {"lastMessage": last_message["content"], "timestamp": last_message["timestamp"]}}
            )

        await cache.delete("get_contacts")
        if data.contactId:
            await cache.delete(f"get_messages_{data.contactId}_{1}_{100}")
        else:
            await cache.delete("get_messages_none_1_100")

        return {"message": f"Uploaded {len(messages)} messages"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
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
            message = {"role": "user", "content": data, "timestamp": "now"}
            await mongo.messages_collection.update_one(
                {"contact_id": contact_id},
                {"$push": {"messages": message}},
                upsert=True
            )
            for ws in active_websockets[contact_id]:
                await ws.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        active_websockets[contact_id].remove(websocket)
        logger.info(f"WebSocket disconnected: contact_id={contact_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_websockets[contact_id]:
            active_websockets[contact_id].remove(websocket)

@app.on_event("shutdown")
async def shutdown_event():
    mongo.client.close()
    logger.info("MongoDB connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
