# create embeddings + index
# file used to build the Chroma vector DB from a CSV Bible file

# Import standard library modules used for file paths, CSV reading, IDs.
import os   # lets us read environment variables and handle paths
import csv  # to read structured Bible CSV file
import uuid # to generate unique IDs for each verse/document

# Import for loading environment variables from .env file
from dotenv import load_dotenv # reads .env and sets environment variables

# Import the Chroma persistent client (saves data to disk)
from chromadb import PersistentClient # provides a database that lives on disk

# Import OpenAI client for creating embeddings via the OpenAI API
from openai import OpenAI # official OpenAI Python SDK

# Call load_dotenv() so variables in .env become available
load_dotenv()

# Set the path to the CSV Bible file 
CSV_PATH = "data/DRC.csv"   # the structured Bible data: Book, Chapter, Verse, Text
# Set where Chroma will store its database files on disk
CHROMA_DIR = "chroma_db"    # folder where vector database data will be saved
# Choose a cost-effective and good quality embedding model
EMBED_MODEL = "text-embedding-3-small" # OpenAI embeddings model name
# Define a safe batch size (Chroma limit is ~5461, so we pick smaller)
BATCH_SIZE = 1000

def main():
    # Pull the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY") # read the OpenAI key from .env
    # If the key is missing, stop and tell the user to set it
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in your .env file.")
    
    # Create an OpenAI client using the API key
    client = OpenAI(api_key=api_key) # used to call embeddings API

    # Create a Chroma persistent client pointing at CHROMA_DIR on disk
    chroma = PersistentClient(path=CHROMA_DIR) # open/creates the DB directory
    # Create or load a collection where we'll store verses
    collection = chroma.get_or_create_collection(name="drb_verses") # a named "table"

    # Prepare lists for batching
    to_docs, to_ids, to_metas, to_embeds = [], [], [], []
    total_count = 0  # track how many verses have been indexed so far

    # Open the CSV file for reading
    with open(CSV_PATH, encoding="utf-8") as f:     # open the CSV file
        reader = csv.DictReader(f)                  # read rows as dicts
        for row in reader:                          # loop over every verse row
            text = (row["Text"] or "").strip()      # get the verse text safely
            if not text:                            # skip empty rows
                continue

            # Build a small metadata dictionary to keep references with each verse
            meta = {
                "book": row["Book"].strip(),        # book name
                "chapter": row["Chapter"].strip(),  # chapter number as string
                "verse": row["Verse"].strip(),      # verse number as string
            }

            # Create a mostly human-readable ID plus a short random suffix
            doc_id = f'{meta["book"]}-{meta["chapter"]}-{meta["verse"]}-{uuid.uuid4().hex[:8]}'

            # Ask OpenAI to convert the verse text into an embedding (a list of floats)
            emb = client.embeddings.create(
                model=EMBED_MODEL, # which embedding model to use
                input=text         # the actual string to embed
            ).data[0].embedding    # pull the vector out of the response

            # Add this verse’s info to the current batch
            to_docs.append(text)
            to_ids.append(doc_id)
            to_metas.append(meta)
            to_embeds.append(emb)

            # If batch reaches BATCH_SIZE, upsert and reset
            if len(to_docs) >= BATCH_SIZE:
                collection.upsert(
                    documents=to_docs,
                    embeddings=to_embeds,
                    metadatas=to_metas,
                    ids=to_ids
                )
                total_count += len(to_docs)
                print(f"Upserted {total_count} verses so far...")

                # Reset lists for the next batch
                to_docs, to_ids, to_metas, to_embeds = [], [], [], []

        # After the loop, flush any leftover verses
        if to_docs:
            collection.upsert(
                documents=to_docs,
                embeddings=to_embeds,
                metadatas=to_metas,
                ids=to_ids
            )
            total_count += len(to_docs)

    # Print a final confirmation
    print(f"Finished indexing {total_count} verses into {CHROMA_DIR}/ (collection 'drb_verses').")


# Standard Python entry point pattern
if __name__ == "__main__":
    main()
