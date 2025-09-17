from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"Testing API key: {api_key[:20]}...{api_key[-4:]}")

try:
    client = OpenAI(api_key=api_key)
    # Simple test call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'API key is working!'"}],
        max_tokens=10
    )
    print("✅ API key is valid!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API key error: {e}")
    print("\nPlease check:")
    print("1. Is this a real OpenAI API key?")
    print("2. Does the key have available credits?")
    print("3. Is the key active (not revoked)?")
    print("\nGet a new key at: https://platform.openai.com/api-keys")