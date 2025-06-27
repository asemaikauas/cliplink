#!/usr/bin/env python3
"""
Test script to verify environment variable loading
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_loading():
    """Test if environment variables are loaded correctly"""
    print("🔍 Testing environment variable loading...")
    
    # Check HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        if hf_token == 'your_huggingface_token_here':
            print("⚠️ HF_TOKEN is set to placeholder value")
            print("   Please update your .env file with your actual HuggingFace token")
        else:
            print(f"✅ HF_TOKEN loaded: {hf_token[:10]}...")
    else:
        print("❌ HF_TOKEN not found in environment")
        print("   Please create a .env file with your HuggingFace token")
    
    # Check other important variables
    other_vars = [
        'ENVIRONMENT',
        'MAX_WORKERS',
        'HOST',
        'PORT'
    ]
    
    for var in other_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️ {var}: not set (using defaults)")
    
    print("\n📝 Instructions:")
    print("1. Copy env.example to .env")
    print("2. Get your HuggingFace token from: https://huggingface.co/settings/tokens")
    print("3. Accept license agreements for pyannote models:")
    print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("   - https://huggingface.co/pyannote/segmentation-3.0")
    print("4. Replace 'your_huggingface_token_here' with your actual token")

if __name__ == "__main__":
    test_env_loading() 