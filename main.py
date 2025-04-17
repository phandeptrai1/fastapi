import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer
from collections import defaultdict

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(title="Chat API")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo cache
cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

# Giới hạn WebSocket
MAX_WEBSOCKETS = 50
active_websockets = defaultdict(list)

# Kết nối MongoDB
class MongoDBConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            uri = os.getenv("MONGODB_URI")
            if not uri:
                logger.error("Biến môi trường MONGODB_URI chưa được thiết lập.")
                raise ValueError("Biến môi trường MONGODB_URI chưa được thiết lập.")
            try:
                logger.info("Đang kết nối đến MongoDB Atlas...")
                cls._client = AsyncIOMotorClient(uri, maxPoolSize=10, minPoolSize=1)
                cls._db = cls._client["chat_app"]
                cls._contacts = cls._db["contacts"]
                cls._messages = cls._db["messages"]
                logger.info("Kết nối MongoDB Atlas thành công.")
            except Exception as e:
                logger.error(f"Lỗi kết nối MongoDB: {e}")
                raise
        return cls._instance

    async def init_indexes(self):
        await self._contacts.create_index([("id", 1)])
        await self._messages.create_index([("contact_id", 1)])
        logger.info("Khởi tạo indexes MongoDB thành công.")

    @property
    def client(self): return self._client

    @property
    def db(self): return self._db

    @property
    def contacts_collection(self): return self._contacts

    @property
    def messages_collection(self): return self._messages

mongo = MongoDBConnection()

# Pydantic models
class Message(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str
    timestamp: str

class SendMessage(Message):
    contactId: Optional[int] = None

class UploadMessages(BaseModel):
    messages: List[Message]
    contactId: Optional[int] = None

# Sự kiện khởi động
@app.on_event("startup")
async def startup_event():
    await mongo.init_indexes()

# Trang chủ
@app.get("/")
async def home():
    return {"message": "🚀 FastAPI đang chạy!"}

# Kiểm tra kết nối DB
@app.get("/test-db")
async def test_db():
    try:
        await mongo.client.admin.command("ping")
        collections = await mongo.db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Kiểm tra DB thất bại: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Lấy danh sách contacts
@app.get("/get-contacts")
@cached(ttl=60, cache=Cache.MEMORY)
async def get_contacts():
    try:
        contacts = await mongo.contacts_collection.find(
            {}, {"_id": 0, "id": 1, "name": 1, "lastMessage": 1, "timestamp": 1, "avatar": 1}
        ).to_list(None)
        return contacts
    except Exception as e:
        logger.error(f"Lỗi khi lấy contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Lấy tin nhắn
@app.get("/get-messages")
@cached(
    ttl=60,
    cache=Cache.MEMORY,
    key_builder=lambda *args, **kwargs: f"get_messages_{kwargs['contact_id']}_{kwargs['page']}_{kwargs['limit']}"
)
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
        logger.error(f"Lỗi khi lấy tin nhắn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Đếm tin nhắn
@app.get("/count-messages")
async def count_messages(contact_id: Optional[int] = Query(None, alias="contactId")):
    try:
        query = {"contact_id": {"$exists": False}} if contact_id is None else {"contact_id": contact_id}
        cursor = mongo.messages_collection.find(query, {"_id": 0, "messages": 1})
        docs = await cursor.to_list(length=None)
        total = sum(len(doc.get("messages", [])) for doc in docs)
        return {"contactId": contact_id, "totalMessages": total}
    except Exception as e:
        logger.error(f"Lỗi khi đếm tin nhắn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Gửi tin nhắn
@app.post("/send-message")
async def send_message(message: SendMessage):
    try:
        query = {"contact_id": message.contactId} if message.contactId else {"contact_id": {"$exists": False}}
        await mongo.messages_collection.update_one(
            query,
            {"$push": {"messages": message.dict()}},
            upsert=True
        )

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

        return {"message": "Gửi tin nhắn thành công"}
    except Exception as e:
        logger.error(f"Lỗi khi gửi tin nhắn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Upload nhiều tin nhắn
@app.post("/upload-messages")
async def upload_messages(data: UploadMessages):
    try:
        messages = [msg.dict() for msg in data.messages]
        query = {"contact_id": data.contactId} if data.contactId else {"contact_id": {"$exists": False}}
        await mongo.messages_collection.update_one(
            query,
            {"$push": {"messages": {"$each": messages}}},
            upsert=True
        )

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

        return {"message": f"Đã upload {len(messages)} tin nhắn"}
    except Exception as e:
        logger.error(f"Lỗi khi upload tin nhắn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Xóa messages cho document _id cụ thể
@app.delete("/clear-messages/{document_id}")
async def clear_messages(document_id: str):
    try:
        if document_id != "67e53df52ea06ce3f3ed901d":
            logger.error(f"ID document không hợp lệ: chỉ chấp nhận 67e53df52ea06ce3f3ed901d, nhận được {document_id}")
            raise HTTPException(
                status_code=400,
                detail="ID document không hợp lệ: chỉ chấp nhận 67e53df52ea06ce3f3ed901d"
            )

        object_id = ObjectId(document_id)
        document = await mongo.messages_collection.find_one({"_id": object_id})
        if not document:
            logger.warning(f"Không tìm thấy document với _id: {document_id}")
            raise HTTPException(status_code=404, detail=f"Không tìm thấy document với _id: {document_id}")

        result = await mongo.messages_collection.update_one(
            {"_id": object_id},
            {"$set": {"messages": []}}
        )
        if result.modified_count > 0:
            logger.info(f"Đã xóa tất cả tin nhắn cho document _id: {document_id}")
            await cache.delete("get_contacts")
            await cache.delete("get_messages_none_1_100")
            return {"message": f"Đã xóa tất cả tin nhắn cho document _id: {document_id}"}
        else:
            logger.warning(f"Không có thay đổi nào cho document _id: {document_id} (messages có thể đã rỗng).")
            return {"message": f"Không có thay đổi nào cho document _id: {document_id} (messages có thể đã rỗng)."}
    except ValueError:
        logger.error(f"ID document không hợp lệ: {document_id}")
        raise HTTPException(status_code=400, detail="ID document không hợp lệ: phải là ObjectId hợp lệ")
    except Exception as e:
        logger.error(f"Lỗi khi xóa tin nhắn: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{contact_id}")
async def websocket_endpoint(websocket: WebSocket, contact_id: int):
    if len(active_websockets[contact_id]) >= MAX_WEBSOCKETS:
        await websocket.close(code=1008, reason="Quá nhiều kết nối")
        return

    await websocket.accept()
    active_websockets[contact_id].append(websocket)

    try:
        WHILE True:
            data = await websocket.receive_text()
            message = {"role": "user", "content": data, "timestamp": "now"}
            await mongo.messages_collection.update_one(
                {"contact_id": contact_id},
                {"$push": {"messages": message}},
                upsert=True
            )
            for ws in active_websockets[contact_id]:
                await ws.send_text(f"Tin nhắn nhận được: {data}")
    except WebSocketDisconnect:
        active_websockets[contact_id].remove(websocket)
        logger.info(f"WebSocket ngắt kết nối: contact_id={contact_id}")
    except Exception as e:
        logger.error(f"Lỗi WebSocket: {e}")
    finally:
        if websocket in active_websockets[contact_id]:
            active_websockets[contact_id].remove(websocket)

# Sự kiện tắt
@app.on_event("shutdown")
async def shutdown_event():
    mongo.client.close()
    logger.info("Đã đóng kết nối MongoDB")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
