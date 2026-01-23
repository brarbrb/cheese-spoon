import os
from dotenv import load_dotenv
from pinecone import Pinecone

# 1. Load variables from the .env file
load_dotenv() 

# 2. Get the key securely
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please check your .env file.")

# 3. Initialize Pinecone
pc = Pinecone(api_key=api_key)
