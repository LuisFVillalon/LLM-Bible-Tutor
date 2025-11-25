# sanity-check retrieval only

# Import modules for environment variables and Chroma access
import os                               # to read environment variables
from dotenv import load_dotenv          # to load .env values into the environment
from chromadb import PersistentClient   # to open our on-disk Chroma DB
from openai import OpenAI               # to get embeddings for the question

# Load variables from .env so we can use OPENAI_API_KEY
load_dotenv()

# Set a test question; change this string to try different queries
QUESTION = "Where does Jesus teach about forgiveness?"  # example user question

def main():
    # Read your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")   # get the key from environment
    # Initialize an OpenAI client for embeddings
    client = OpenAI(api_key=api_key)        # we only need embeddings here

    # Connect to the Chroma database directory we created earlier
    chroma = PersistentClient(path="chroma_db")     # open the same DB folder
    # Get the same collection name we used in build_index.py
    col = chroma.get_collection("drb_verses")       # open the verses collection

    # Create an embedding for the QUESTION text so we can search by meaning
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",   # same embedding model used for verses
        input=QUESTION                     # the query string to embed
    ).data[0].embedding                    # extract the vector

    # Query the vector database for the top 5 closest verses by cosine distance
    res = col.query(
        query_embeddings=[q_emb],          # list of one embedding to search with
        n_results=5,                       # how many nearest neighbors to return
        include=["documents", "metadatas", "distances"]  # return full info
    )

    # Loop through the results and print reference + text + similarity score
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        ref = f'{meta["book"]} {meta["chapter"]}:{meta["verse"]}'  # Book Chap:Verse
        print(f"- {ref}  (score: {dist:.4f})")                     # show distance
        print(f"  {doc}\n")                                       # show verse text

# Run main() if the script is executed directly
if __name__ == "__main__":
    main()
