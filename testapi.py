# file used to test the OpenAI API key 

from openai import OpenAI
import os
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

# now you can use it
api_key = os.getenv("OPENAI_API_KEY")
print("API Key loaded?", bool(api_key))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
resp = client.models.list()
print([m.id for m in resp.data[:5]])
