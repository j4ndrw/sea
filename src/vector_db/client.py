from src.constants import EVOLVED_KNOWLEDGE_BASE_DIR
import chromadb

knowledge_base_client = chromadb.PersistentClient(path=EVOLVED_KNOWLEDGE_BASE_DIR)
