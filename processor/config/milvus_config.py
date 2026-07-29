# 导入核心依赖
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# 定义milvus向量数据库配置类
@dataclass
class MilvusConfig:
    milvus_url: str  # Milvus服务端连接地址
    milvus_db_name: str # Milvus数据库名称
    chunks_collection: str # 存储切片的集合名称
    entity_name_collection: str # 预留-实体名称集合
    item_name_collection: str # 存储文档对应实体类的集合名称
    milvus_user: str
    milvus_password: str

# 实例化milvus配置对象
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    milvus_db_name=os.getenv("MILVUS_DB_NAME") or "default",
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION"),
    milvus_user = os.getenv("MILVUS_USER"),
    milvus_password = os.getenv("MILVUS_PASSWORD")
)
