#!/usr/bin/env python3
"""
HuggingFace Authentication and Model Setup Script
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False, e.stderr

def test_pyannote_models():
    """Test pyannote model access"""
    print("\n🧪 Testing pyannote model access...")
    
    try:
        # Test segmentation model
        print("📥 Testing pyannote/segmentation-3.0...")
        from pyannote.audio import Model, Inference
        
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        inference = Inference(model)
        print("✅ Segmentation model loaded successfully")
        
        # Test speaker diarization model
        print("📥 Testing pyannote/speaker-diarization-3.1...")
        from pyannote.audio import Pipeline
        
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("✅ Speaker diarization pipeline loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        print(f"   This is likely due to missing authentication or license acceptance")
        return False

def check_hf_cli():
    """Check if huggingface-cli is available"""
    success, output = run_command("huggingface-cli --version", "Checking HuggingFace CLI")
    if not success:
        print("⚠️ HuggingFace CLI not found. Installing...")
        success, _ = run_command("pip install huggingface_hub[cli]", "Installing HuggingFace CLI")
        if success:
            print("✅ HuggingFace CLI installed successfully")
        return success
    return True

def check_hf_auth():
    """Check if user is authenticated with HuggingFace"""
    success, output = run_command("huggingface-cli whoami", "Checking HuggingFace authentication")
    if success:
        print(f"✅ Authenticated with HuggingFace")
        return True
    else:
        print("❌ Not authenticated with HuggingFace")
        return False

def main():
    """Main setup function"""
    print("🚀 HuggingFace Authentication and Model Setup")
    print("=" * 50)
    
    # Check if HF CLI is available
    if not check_hf_cli():
        print("❌ Failed to install HuggingFace CLI")
        return False
    
    # Check authentication status
    is_authenticated = check_hf_auth()
    
    if not is_authenticated:
        print("\n🔑 Authentication Required:")
        print("Run the following command to authenticate:")
        print("   huggingface-cli login")
        print("\nAfter authentication, run this script again to test model access.")
        return False
    
    # Test model access
    print("\n🧪 Testing model access...")
    models_work = test_pyannote_models()
    
    if models_work:
        print("\n✅ Setup completed successfully!")
        print("All models are accessible and working correctly.")
    else:
        print("\n⚠️ Model access failed!")
        print("Please ensure you have:")
        print("1. Accepted license agreements:")
        print("   - https://huggingface.co/pyannote/segmentation-3.0")
        print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. Your HuggingFace account has access to these models")
        print("3. Try logging out and back in:")
        print("   huggingface-cli logout")
        print("   huggingface-cli login")
    
    print("\n📝 Next steps:")
    print("1. Start the server: uvicorn app.main:app --reload")
    print("2. Test the advanced endpoints")
    print("3. Check README_ADVANCED_SETUP.md for detailed usage")
    
    return models_work

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        print("To complete setup manually:")
        print("1. Run: huggingface-cli login")
        print("2. Accept model licenses")
        print("3. Run this script again") 