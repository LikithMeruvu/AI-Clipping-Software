import random
from config import TEMP_DIR

def cleanup_temp_files():
    """Removes all files from the temporary directory."""
    print("🧹 Cleaning up temporary files...")
    try:
        for item in TEMP_DIR.iterdir():
            item.unlink()
        print("✅ Cleanup complete.")
    except Exception as e:
        print(f"⚠️  Could not clean up all temporary files: {e}")


def generate_random_clips(duration, num_clips, min_duration, max_duration):
    """Generates random clip start and end times as a fallback."""
    clips = []
    for i in range(num_clips):
        clip_duration = random.uniform(min_duration, max_duration)
        if duration - clip_duration <= 0:
            continue
        start = random.uniform(0, duration - clip_duration)
        clips.append({
            'start': start,
            'end': start + clip_duration,
            'title': f'Random clip {i+1}',
            'virality_score': 30,
            'hook_type': 'general',
            'duration': clip_duration
        })
    return clips
