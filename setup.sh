#!/bin/bash

echo "Setting up English Level Test Speech Evaluation System..."
echo "==========================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Set up environment variables
echo ""
echo "Please set the following environment variables:"
echo "1. OPENAI_API_KEY - Your OpenAI API key for GPT-4 evaluation"
echo "2. HUGGINGFACE_TOKEN - Your Hugging Face token for speaker diarization (optional)"
echo ""
echo "You can set them in your shell profile or export them:"
echo "export OPENAI_API_KEY='your-api-key-here'"
echo "export HUGGINGFACE_TOKEN='your-hf-token-here'"
echo ""

# Download Whisper model
echo "The Whisper model will be downloaded on first run..."
echo ""

echo "Setup complete! You can now run:"
echo "python speech_test2.py"