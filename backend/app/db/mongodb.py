from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.core.config import settings

# These get set on app startup (see app/main.py lifespan) and reused
# everywhere else in the app - never create a second client.
client: AsyncIOMotorClient | None = None
db = None
gridfs_bucket: AsyncIOMotorGridFSBucket | None = None


async def connect_to_mongo():
    global client, db, gridfs_bucket
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.DB_NAME]
    gridfs_bucket = AsyncIOMotorGridFSBucket(db)

    # Fail fast if the URI/credentials are wrong, instead of failing later
    # on the first real request.
    await client.admin.command("ping")

    # Helpful indexes - safe to call every startup, no-ops if they exist
    await db.documents.create_index("session_id")
    await db.chunks.create_index("session_id")
    await db.chunks.create_index("doc_id")
    await db.conversations.create_index("session_id")


async def close_mongo_connection():
    global client
    if client is not None:
        client.close()


def get_db():
    if db is None:
        raise RuntimeError("MongoDB not connected yet - did startup run?")
    return db


def get_gridfs_bucket() -> AsyncIOMotorGridFSBucket:
    if gridfs_bucket is None:
        raise RuntimeError("MongoDB not connected yet - did startup run?")
    return gridfs_bucket