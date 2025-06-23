import os
import json
import httpx
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def extract_subtitles_for_segments(viral_segments: List[Dict], original_timecodes: List[Dict]) -> List[Dict]:
    """
    Extract subtitle text for each viral segment based on timecodes
    """
    enhanced_segments = []
    
    print(f"\n🎬 Извлекаю субтитры для {len(viral_segments)} сегментов...")
    
    for i, segment in enumerate(viral_segments, 1):
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        title = segment.get('title', 'Без названия')
        
        print(f"\n--- Сегмент {i}: {title} ---")
        print(f"⏰ Время: {start_time} - {end_time} сек ({end_time - start_time} сек)")
        
        # Find all subtitle segments that fall within this time range
        subtitle_segments = []
        subtitle_text_parts = []
        
        for timecode in original_timecodes:
            tc_start = timecode.get('start', 0)
            tc_duration = timecode.get('duration', 0)
            tc_end = tc_start + tc_duration
            tc_text = timecode.get('text', '').strip()
            
            # Check if this timecode overlaps with the viral segment
            if (tc_start >= start_time and tc_start <= end_time) or \
               (tc_end >= start_time and tc_end <= end_time) or \
               (tc_start <= start_time and tc_end >= end_time):
                
                subtitle_segments.append({
                    "start": tc_start,
                    "end": tc_end,
                    "duration": tc_duration,
                    "text": tc_text
                })
                
                if tc_text:  # Only add non-empty text
                    subtitle_text_parts.append(tc_text)
        
        # Combine all subtitle text for this segment
        full_subtitle_text = ' '.join(subtitle_text_parts)
        
        print(f"📝 Найдено субтитров: {len(subtitle_segments)} сегментов")
        print(f"📄 Полный текст ({len(full_subtitle_text)} символов):")
        print(f"   {full_subtitle_text[:150]}{'...' if len(full_subtitle_text) > 150 else ''}")
        
        # Create enhanced segment with subtitles
        enhanced_segment = {
            **segment,
            "duration": end_time - start_time,
            "subtitles": {
                "full_text": full_subtitle_text
            }
        }
        
        enhanced_segments.append(enhanced_segment)
    
    return enhanced_segments

async def analyze_transcript_with_gemini(transcript_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze transcript using Gemini AI to generate viral video segments
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Extract data for prompt
    title = transcript_data.get('title', '')
    description = transcript_data.get('description', '')
    category = transcript_data.get('category', '')
    transcript = transcript_data.get('transcript', '')
    timecodes = transcript_data.get('timecodes', [])

    print(f"\n🤖 Анализ видео с Gemini AI:")
    print(f"📺 Название: {title}")
    print(f"📂 Категория: {category}")
    print(f"📄 Длина транскрипта: {len(transcript)} символов")
    print(f"⏱️ Количество таймкодов: {len(timecodes)}")

    prompt = f"""

     You are a senior expert in video editing and screenwriting and 
     story producer for social networks (TikTok, YouTube Shorts, Reels) with 10 
     years of experience.

     Your task is to analyze the full text transcription, title, description and category of the Youtube Video and construct up to 10 viral short videos 
     with a clear storyline, selecting the catchy and sensational moments to drive people watch the full video. Video should start with a "hook" in the 
     first 3-7 seconds that contains provocative, catching attention information. Each video should have one unique idea, story or theme. Provide 
     duration of each selected video segment in seconds with start and end timecodes. 

     VIDEO INFORMATION:
     Title: {title}
     Description: {description}
     Category: {category}
     
     TRANSCRIPT:
     {transcript}

     TIMECODES: 
     {timecodes}
     
     INSTRUCTIONS:
     Analyze the transcript for viral moments, hooks, and engaging content, create 3-10 short video segments (40 seconds to 2 minutes each), 
     focus on controversial, surprising, educational, or entertaining moments, use the timecodes to select precise start/end times in seconds. Make sure that 
     selected video segments are not overlappping and tell one story. 

     1. Use the transcript + timecodes to pick exact start & end in seconds (floats not allowed).  
     2. Duration for every segment: 40 ≤ (end − start) ≤ 120.  

     
     Output Format: 
     {{
         "viral_segments": [
             {{
                 "title": "Catchy title for the segment",
                 "start": 0,
                 "end": 45
             }}
         ]
     }}
     
     Remember: Return ONLY the JSON object, no additional text or explanations."""

    # Use the correct Gemini 2.5 Pro preview model name from the API list
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-06-05:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 1.2,
            "topP": 0.9,
            "topK": 40,
            "maxOutputTokens": 65535
        }
    }
    
    print(f"\n🚀 Отправляю запрос к Gemini API...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=600.0)
            response.raise_for_status()
            
            gemini_response = response.json()
            
            if 'candidates' in gemini_response and len(gemini_response['candidates']) > 0:
                candidate = gemini_response['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    generated_text = candidate['content']['parts'][0]['text']
                    
                    print(f"✅ Получен ответ от Gemini ({len(generated_text)} символов)")
                    
                    # Clean up the response (remove markdown code blocks if present)
                    generated_text = generated_text.strip()
                    if generated_text.startswith('```json'):
                        generated_text = generated_text[7:]
                    if generated_text.startswith('```'):
                        generated_text = generated_text[3:]
                    if generated_text.endswith('```'):
                        generated_text = generated_text[:-3]
                    generated_text = generated_text.strip()
                      
                    try:
                        viral_analysis = json.loads(generated_text)
                        
                        print(f"\n🎯 Gemini нашел {len(viral_analysis.get('viral_segments', []))} вирусных сегментов")
                        
                        # Extract subtitles for each viral segment
                        if 'viral_segments' in viral_analysis and timecodes:
                            enhanced_segments = extract_subtitles_for_segments(
                                viral_analysis['viral_segments'], 
                                timecodes
                            )
                            viral_analysis['viral_segments'] = enhanced_segments
                            
                            print(f"\n📋 ИТОГОВЫЙ JSON С СУБТИТРАМИ:")
                            print("=" * 60)
                            print(json.dumps(viral_analysis, ensure_ascii=False, indent=2))
                            print("=" * 60)
                        else:
                            print(f"⚠️ Не найдены вирусные сегменты или таймкоды для обработки")
                        
                        # Combine original data with Gemini analysis
                        result = {
                            "gemini_analysis": viral_analysis
                        }
                        
                        return result
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                        print(f"❌ Generated text: {generated_text}")
                        return {
                            "gemini_analysis": {
                                "error": "Failed to parse Gemini response",
                                "raw_response": generated_text
                            }
                        }
            
            print(f"❌ Нет валидного ответа от Gemini")
            return {
                "gemini_analysis": {
                    "error": "No valid response from Gemini",
                    "raw_response": gemini_response
                }
            }
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_response = e.response.json()
                error_detail = json.dumps(error_response, indent=2)
            except:
                error_detail = e.response.text
            
            print(f"❌ Gemini API error {e.response.status_code}: {error_detail}")
            raise Exception(f"Gemini API error {e.response.status_code}: {error_detail}")
        except httpx.TimeoutException:
            print(f"⏱️ Gemini API request timed out after 600 seconds")
            raise Exception("Gemini API request timed out after 600 seconds")
        except Exception as e:
            print(f"❌ Unexpected error calling Gemini API: {str(e)}")
            raise Exception(f"Unexpected error calling Gemini API: {str(e)}") 