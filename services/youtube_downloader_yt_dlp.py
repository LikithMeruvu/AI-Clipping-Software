import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yt_dlp

from config import TEMP_DIR, YOUTUBE_COOKIES_CONTENT, YOUTUBE_USER_AGENT


class YouTubeDownloader:
    def __init__(self, temp_dir=TEMP_DIR):
        self.temp_dir = temp_dir

    @staticmethod
    def get_video_id(url):
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        if 'youtube.com' in url:
            return parse_qs(urlparse(url).query).get('v', [None])[0]
        return None

    def download(self, url):
        video_id = self.get_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL provided.")

        opts = {
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': str(self.temp_dir / f'{video_id}.%(ext)s'),
            'quiet': True,
            'merge_output_format': 'mp4',
            'geo_bypass': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'no_warnings': False,
            'retries': 10,
            'fragment_retries': 10,
            'extractor_retries': 10,
            'user_agent': YOUTUBE_USER_AGENT
        }

        cookie_file_path = None
        if YOUTUBE_COOKIES_CONTENT and "PASTE" not in YOUTUBE_COOKIES_CONTENT:
            print("Authentication cookies found. Applying them to the download request.")
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, dir=self.temp_dir, suffix='.txt') as cookie_file:
                cookie_file.write(YOUTUBE_COOKIES_CONTENT)
                cookie_file_path = cookie_file.name
            opts['cookiefile'] = cookie_file_path
        else:
            print("Warning: No cookies provided. The download may be blocked by YouTube for certain videos.")

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                try:
                    info = ydl.extract_info(url, download=True)
                except yt_dlp.utils.DownloadError as e:
                    print(f"First download attempt failed: {str(e)}")
                    print("Trying alternative download method...")
                    # Try with different format options
                    opts['format'] = 'best[ext=mp4]/best'
                    with yt_dlp.YoutubeDL(opts) as ydl2:
                        info = ydl2.extract_info(url, download=True)
        except Exception as e:
            raise Exception(f"Failed to download video: {str(e)}")
        finally:
            if cookie_file_path and os.path.exists(cookie_file_path):
                os.remove(cookie_file_path)

        video_path = next(self.temp_dir.glob(f'{video_id}.mp4'), None)
        if not video_path:
            # Try to find any video file with the video_id prefix
            video_path = next(self.temp_dir.glob(f'{video_id}.*'), None)
            if not video_path:
                raise FileNotFoundError("Failed to download the video file.")

        return video_path, info.get('title', 'N/A'), info.get('duration', 0)