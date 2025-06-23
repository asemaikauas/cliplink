#!/usr/bin/env python3
"""
🎛️ ТЕСТ СГЛАЖИВАНИЯ ДВИЖЕНИЙ В ВЕРТИКАЛЬНОМ КРОПЕ

Этот скрипт поможет вам протестировать разные уровни сглаживания 
для устранения дёрганья головы в вертикальных видео.

Использование:
    python test_smoothing.py <путь_к_видео>
    
Пример:
    python test_smoothing.py downloads/video.mp4
"""

import sys
import os
from pathlib import Path
import time

# Добавляем путь к модулям
sys.path.append('app')

def test_smoothing_levels(video_path: str):
    """
    Тестирует все уровни сглаживания для одного видео
    """
    try:
        from app.services.vertical_crop import crop_video_to_vertical, get_available_resolutions
        
        input_path = Path(video_path)
        if not input_path.exists():
            print(f"❌ Видео не найдено: {video_path}")
            return
        
        print(f"🎬 Тестируем сглаживание для: {input_path.name}")
        
        # Создаём папку для тестов
        test_dir = Path("test_smoothing")
        test_dir.mkdir(exist_ok=True)
        
        base_name = input_path.stem
        
        smoothing_levels = {
            "low": "Минимальное сглаживание (быстрая реакция)",
            "medium": "Среднее сглаживание (баланс)",
            "high": "Максимальное сглаживание (очень плавно)"
        }
        
        results = []
        
        for level, description in smoothing_levels.items():
            print(f"\n🎛️ === ТЕСТ: {level.upper()} SMOOTHING ===")
            print(f"📝 {description}")
            
            output_path = test_dir / f"{base_name}_smoothing_{level}.mp4"
            
            start_time = time.time()
            
            success = crop_video_to_vertical(
                input_path=input_path,
                output_path=output_path,
                resolution="shorts_hd",
                use_speaker_detection=True,
                smoothing_strength=level
            )
            
            processing_time = time.time() - start_time
            
            if success and output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024*1024)
                print(f"✅ Создан: {output_path.name}")
                print(f"⏱️ Время: {processing_time:.1f} сек")
                print(f"📁 Размер: {file_size_mb:.1f} MB")
                
                results.append({
                    "level": level,
                    "description": description,
                    "file": output_path,
                    "size_mb": file_size_mb,
                    "time_sec": processing_time
                })
            else:
                print(f"❌ Ошибка создания: {level}")
        
        # Финальный отчёт
        print(f"\n🎉 === РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===")
        print(f"📂 Исходное видео: {input_path.name}")
        print(f"📁 Выходная папка: {test_dir}/")
        print(f"📊 Создано файлов: {len(results)}/3")
        
        if results:
            print(f"\n📋 Сравнение результатов:")
            for result in results:
                print(f"  🎛️ {result['level'].upper()}: {result['file'].name}")
                print(f"     └─ {result['description']}")
                print(f"     └─ Размер: {result['size_mb']:.1f} MB, Время: {result['time_sec']:.1f} сек")
            
            print(f"\n💡 РЕКОМЕНДАЦИИ:")
            print(f"   • LOW    = Для быстрых динамичных сцен")
            print(f"   • MEDIUM = Универсальный выбор (рекомендуется)")
            print(f"   • HIGH   = Для максимальной плавности (может быть медленным)")
            
            print(f"\n🎯 Откройте файлы и сравните:")
            print(f"   Какой уровень сглаживания выглядит лучше для вашего видео?")
        
    except ImportError:
        print("❌ Ошибка: Модули вертикального кропа недоступны")
        print("   Установите зависимости: pip install opencv-python pydub webrtcvad")
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")

def main():
    """Главная функция"""
    print("🎛️ ТЕСТЕР СГЛАЖИВАНИЯ ДВИЖЕНИЙ")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("❌ Неправильное использование!")
        print("   Использование: python test_smoothing.py <путь_к_видео>")
        print("   Пример: python test_smoothing.py downloads/video.mp4")
        return
    
    video_path = sys.argv[1]
    test_smoothing_levels(video_path)

if __name__ == "__main__":
    main() 