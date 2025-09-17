import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {api_key[:10] if api_key else 'Not loaded'}...")
print(f"API Key length: {len(api_key) if api_key else 0}")
print(f"Current directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")

# Read .env file directly
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        print("\n.env file contents:")
        for line in f:
            if 'OPENAI_API_KEY' in line:
                print(f"Found: {line[:30]}...")