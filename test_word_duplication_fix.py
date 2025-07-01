#!/usr/bin/env python3
"""Test to verify word duplication bug is fixed."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.subs import SubtitleProcessor

def test_word_duplication_fix():
    """Test that words are not duplicated when text is split."""
    
    print("🧪 Testing Word Duplication Fix...")
    print("="*50)
    
    processor = SubtitleProcessor(max_chars_per_line=25, max_lines=2)
    
    # Test case that should trigger the bug
    test_text = "you know this is how you can learn coding. This is kind of how you can fix your bugs."
    
    print(f"📝 Input text: '{test_text}'")
    print(f"📏 Max chars per line: {processor.max_chars_per_line}")
    print(f"📄 Max lines: {processor.max_lines}")
    print()
    
    # Test the wrapping function directly
    wrapped_lines, remaining_text = processor._wrap_text(test_text)
    
    print(f"✅ First wrap result:")
    print(f"   Lines: {wrapped_lines}")
    print(f"   Remaining: '{remaining_text}'")
    print()
    
    # Check for word duplication in remaining text
    if remaining_text:
        words_in_lines = " ".join(wrapped_lines).split()
        words_in_remaining = remaining_text.split()
        
        print(f"🔍 Checking for duplications...")
        print(f"   Words in lines: {words_in_lines}")
        print(f"   Words in remaining: {words_in_remaining}")
        
        # Check if the first word of remaining appears at the end of lines
        if words_in_lines and words_in_remaining:
            last_word_in_lines = words_in_lines[-1]
            first_word_in_remaining = words_in_remaining[0]
            
            if last_word_in_lines == first_word_in_remaining:
                print(f"❌ DUPLICATION FOUND: '{last_word_in_lines}' appears in both!")
                return False
            else:
                print(f"✅ No duplication: '{last_word_in_lines}' != '{first_word_in_remaining}'")
        
        # Test second wrap of remaining text
        print(f"\n🔄 Testing second wrap of remaining text...")
        wrapped_lines2, remaining_text2 = processor._wrap_text(remaining_text)
        
        print(f"   Lines: {wrapped_lines2}")
        print(f"   Remaining: '{remaining_text2}'")
        
        # Check for duplication in second wrap
        if remaining_text2:
            words_in_lines2 = " ".join(wrapped_lines2).split()
            words_in_remaining2 = remaining_text2.split()
            
            if words_in_lines2 and words_in_remaining2:
                last_word_in_lines2 = words_in_lines2[-1]
                first_word_in_remaining2 = words_in_remaining2[0]
                
                if last_word_in_lines2 == first_word_in_remaining2:
                    print(f"❌ DUPLICATION FOUND in second wrap: '{last_word_in_lines2}' appears in both!")
                    return False
                else:
                    print(f"✅ No duplication in second wrap: '{last_word_in_lines2}' != '{first_word_in_remaining2}'")
    
    print(f"\n🎉 Word duplication test PASSED!")
    return True

if __name__ == "__main__":
    success = test_word_duplication_fix()
    if success:
        print("\n✅ Fix verified! Words should no longer be duplicated.")
        sys.exit(0)
    else:
        print("\n❌ Fix failed! Word duplication still occurs.")
        sys.exit(1) 