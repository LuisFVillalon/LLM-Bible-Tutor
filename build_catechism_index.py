import os
import csv
import uuid
from dotenv import load_dotenv
from chromadb import PersistentClient
from openai import OpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to .env")

# Config
CSV_PATH = "data/catechism_compendium.csv"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ccc_qna"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 200  # smaller to avoid API/rate issues

def main():
    client = OpenAI(api_key=api_key)
    chroma = PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

    to_docs, to_ids, to_metas, to_embeds = [], [], [], []
    total = 0

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use Question + Answer as embedding text
            text = f"Q: {row['Question']}\nA: {row['Answer']}".strip()
            if not text:
                continue

            meta = {
                "question_num": row["QuestionNumber"],
                "paragraph_refs": row["ParagraphRefs"],
                "section": row["Section"],
                "chapter": row["Chapter"]
            }
            doc_id = f"{row['QuestionNumber']}-{uuid.uuid4().hex[:8]}"

            emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

            to_docs.append(text)
            to_ids.append(doc_id)
            to_metas.append(meta)
            to_embeds.append(emb)

            if len(to_docs) >= BATCH_SIZE:
                collection.upsert(documents=to_docs, embeddings=to_embeds, metadatas=to_metas, ids=to_ids)
                total += len(to_docs)
                print(f"Upserted {total} rows...")
                to_docs, to_ids, to_metas, to_embeds = [], [], [], []

    if to_docs:
        collection.upsert(documents=to_docs, embeddings=to_embeds, metadatas=to_metas, ids=to_ids)
        total += len(to_docs)

    print(f"✅ Finished indexing {total} Q&A entries into Chroma collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
