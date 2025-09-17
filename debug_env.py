#!/usr/bin/env python3
import os
from dotenv import load_dotenv

print("Before load_dotenv:")
print(f"  OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

load_dotenv()

print("\nAfter load_dotenv:")
api_key = os.getenv('OPENAI_API_KEY')
print(f"  OPENAI_API_KEY: {api_key[:20]}..." if api_key else "  OPENAI_API_KEY: None")

print(f"\n.env file exists: {os.path.exists('.env')}")
print(f"Current directory: {os.getcwd()}")

from openai import OpenAI
client = OpenAI()
print(f"\nOpenAI client API key: {client.api_key[:20]}..." if client.api_key else "OpenAI client API key: None")