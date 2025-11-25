# full RAG Q&A with GPT-4o-mini
# file used to run a retrieval-augmented generation (RAG) Q&A loop

# Import to read environment variables (API key) from .env
import os                    # standard library to access environment variables
from dotenv import load_dotenv  # loads .env file into environment

# Import LangChain components for OpenAI chat models and embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI wrappers for LangChain

# Import Chroma integration for LangChain (VectorStore wrapper)
from langchain_community.vectorstores import Chroma  # lets LangChain talk to Chroma

# Import prompt templating to structure the LLM instructions
from langchain.prompts import ChatPromptTemplate  # helps create a reusable prompt

# Import a tool to pass raw input directly through a chain
from langchain.schema.runnable import RunnablePassthrough  # used to pass question as-is

# Load .env so OPENAI_API_KEY is available
load_dotenv()

# Create a ChatOpenAI LLM instance that will call GPT-4o-mini for answers
llm = ChatOpenAI(
    model="gpt-4o-mini",                      # the chat model to use
    temperature=0.2,                          # low randomness for factual tone
    api_key=os.getenv("OPENAI_API_KEY"),      # read API key from environment
)

# Create an embeddings object to let LangChain compute embeddings when needed
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",           # embedding model name
    api_key=os.getenv("OPENAI_API_KEY"),      # same API key as above
)

# Create a LangChain VectorStore pointing to our on-disk Chroma collection
vectordb = Chroma(
    collection_name="drb_verses",             # must match the collection we built
    persist_directory="chroma_db",            # folder with Chroma data
    embedding_function=emb,                   # how to embed queries if needed
)

# Turn the VectorStore into a retriever; tells it how many results to return
retriever = vectordb.as_retriever(
    search_kwargs={"k": 5}                    # fetch top 5 relevant verses
)

# Build a structured prompt that instructs the model how to answer
prompt = ChatPromptTemplate.from_template(
    # The triple-quoted string below is the actual template
    """You are a Catholic Bible tutor using the Douay-Rheims Bible.
Use ONLY the provided passages to answer. If the answer is not present, say you don't know.
Always cite book, chapter, and verse.

Passages:
{context}

Question: {question}

Return:
- 2–5 sentence answer.
- Bullet list of citations (Book Chapter:Verse).
"""
)

# Helper function that converts retrieved documents into a readable string
def format_docs(docs):
    # We'll put each verse on its own line with a reference and the verse text
    lines = []                                # collect each formatted line here
    for d in docs:                            # loop over Document objects
        m = d.metadata                        # metadata contains book/chapter/verse
        ref = f'{m.get("book")} {m.get("chapter")}:{m.get("verse")}'  # make "Book C:V"
        lines.append(f"- {ref} — {d.page_content}")    # add formatted line
    return "\n".join(lines)                   # join lines into a single string

# Build the LangChain "graph":
# 1) Use the retriever on the question to get relevant context, format it
# 2) Fill the prompt template with {context} and {question}
# 3) Send to LLM to get an answer
chain = (
    {"context": retriever | format_docs,       # pipe: retrieve then format
     "question": RunnablePassthrough()}        # pass the raw question straight through
    | prompt                                   # fill the prompt template
    | llm                                      # call the model to produce an answer
)

# Allow interactive Q&A from the terminal
if __name__ == "__main__":
    print("Bible Tutor ready. Type a question, or 'quit' to exit.")  # greeting text
    while True:                                                      # loop until user exits
        q = input("\nQ: ").strip()                                  # read user input
        if q.lower() in {"quit", "exit"}:                            # if user wants to stop
            break                                                    # break the loop
        ans = chain.invoke(q)                                        # run the chain with question
        print("\n" + ans.content)                                    # print the model's reply
