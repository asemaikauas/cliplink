import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api():
    # Test the root endpoint
    print("Testing root endpoint...")
    response = requests.get("http://localhost:8000/")
    print(f"Root endpoint: {response.status_code} - {response.json()}")
    
    # Test the health endpoint
    print("\nTesting health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Health endpoint: {response.status_code} - {response.json()}")
    
    # Test the transcript endpoint
    print("\nTesting transcript endpoint...")
    test_url = "https://youtu.be/XLjx9wGOZhI?si=-KwJ1G7NdBnWK_zA"  # User provided link
    payload = {"youtube_url": test_url}
    
    response = requests.post(
        "http://localhost:8000/transcript",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Transcript endpoint: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Video title: {data.get('title', 'N/A')}")
        print(f"Category: {data.get('category', 'N/A')}")
        print(f"Description: {data.get('description', 'N/A')[:100]}...")
        print(f"Transcript length: {len(data.get('transcript', ''))}")
        print(f"Timecodes count: {len(data.get('timecodes', []))}")
        if data.get('transcript'):
            print(f"First 200 chars: {data.get('transcript', '')[:200]}...")
        else:
            print("Transcript is empty")
    else:
        print(f"Error: {response.text}")
    
    # Check environment variable
    print(f"\nAPI Key loaded: {'Yes' if os.getenv('YOUTUBE_TRANSCRIPT_API') else 'No'}")
    print(f"API Key: {os.getenv('YOUTUBE_TRANSCRIPT_API')[:10]}..." if os.getenv('YOUTUBE_TRANSCRIPT_API') else "No API Key")

if __name__ == "__main__":
    test_api() 