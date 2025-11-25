# File used to inspect the contents of a persistent Chroma DB

# inspect_chroma.py
import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from openai import OpenAI

# Load environment (needed for OpenAI embeddings)
load_dotenv()

# Path to your persistent Chroma DB
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "drb_verses"
EMBED_MODEL = "text-embedding-3-small"

def main():
    # Connect to Chroma
    client = PersistentClient(path=CHROMA_DIR)

    # Get the collection by name
    collection = client.get_collection(COLLECTION_NAME)

    # Count how many documents are stored
    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' contains {count} verses.\n")

    # Peek at the first few documents
    sample = collection.peek(limit=5)  # grab 5 random docs
    print("Sample verses from the collection:\n")
    for doc, meta in zip(sample["documents"], sample["metadatas"]):
        ref = f'{meta.get("book")} {meta.get("chapter")}:{meta.get("verse")}'
        print(f"- {ref}: {doc[:80]}{'...' if len(doc) > 80 else ''}")

    # --- Direct semantic search test (no GPT) ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file")

    openai_client = OpenAI(api_key=api_key)

    # Example query
    QUESTION = "How to be a good husband?"

    # Create embedding for the question
    q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=QUESTION).data[0].embedding

    # Search Chroma for the 5 closest matches
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    print("\nTop 5 verses Chroma found for your test question:\n")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        ref = f'{meta["book"]} {meta["chapter"]}:{meta["verse"]}'
        print(f"- {ref} (score {dist:.4f}) -> {doc[:80]}{'...' if len(doc) > 80 else ''}")


if __name__ == "__main__":
    main()
