import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# pytubefix is the maintained fork of pytube
from pytubefix import YouTube
from pytubefix.exceptions import PytubeFixError

from config import TEMP_DIR, YOUTUBE_USER_AGENT


class YouTubeDownloader:
    """
    Handles the downloading of YouTube videos.

    YouTube no longer offers muxed (audio+video) streams above 360p. Those videos
    are fetched as separate video and audio streams and muxed with ffmpeg. Without
    the audio track, transcription fails, so this matters for most videos.
    """
    def __init__(self, temp_dir=TEMP_DIR):
        self.temp_dir = temp_dir

    def _sanitize_filename(self, filename):
        if not filename:
            return "unknown_title"
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        if len(filename) > 100:
            filename = filename[:97] + '...'
        return filename

    @staticmethod
    def get_video_id(url):
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        if 'youtube.com' in url:
            return parse_qs(urlparse(url).query).get('v', [None])[0]
        return None

    @staticmethod
    def _ffmpeg_exe():
        """Use the ffmpeg binary bundled with moviepy's imageio-ffmpeg dependency."""
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()

    def _mux(self, video_file, audio_file, out_file):
        result = subprocess.run(
            [self._ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video_file, "-i", audio_file,
             "-c", "copy", out_file],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not os.path.exists(out_file):
            raise RuntimeError(f"ffmpeg could not mux audio and video: {result.stderr.strip()[-300:]}")
        return out_file

    def download(self, url):
        """
        Downloads a YouTube video and returns (path, title, duration).
        The returned mp4 always contains an audio track when the video has one.
        """
        print(f"🔗 Processing YouTube URL: {url}")
        video_id = self.get_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL provided.")
        print(f"✅ Extracted video ID: {video_id}")

        os.makedirs(self.temp_dir, exist_ok=True)
        output_path = os.path.abspath(str(self.temp_dir))
        output_filename = f"{video_id}.mp4"
        final_path = os.path.join(output_path, output_filename)

        try:
            print("⏳ Fetching video metadata...")
            yt = YouTube(url)
            title = self._sanitize_filename(yt.title)
            duration = yt.length
            print(f"Video title: {title}")
            print(f"Video duration: {duration} seconds ({duration//60}m {duration%60}s)")

            progressive = (
                yt.streams.filter(progressive=True, file_extension='mp4')
                .order_by('resolution').desc().first()
            )
            if progressive:
                print(f"Selected muxed stream: {progressive.resolution}")
                video_path = progressive.download(output_path=output_path, filename=output_filename)
            else:
                video_stream = (
                    yt.streams.filter(adaptive=True, only_video=True, file_extension='mp4')
                    .order_by('resolution').desc().first()
                )
                audio_stream = (
                    yt.streams.filter(adaptive=True, only_audio=True)
                    .order_by('abr').desc().first()
                )
                if not video_stream or not audio_stream:
                    raise ValueError("No suitable video and audio streams found")
                print(f"Selected video stream: {video_stream.resolution}; audio: {audio_stream.abr}")
                v_file = video_stream.download(output_path=output_path, filename=f"v_{video_id}.mp4")
                a_file = audio_stream.download(output_path=output_path, filename=f"a_{video_id}.m4a")
                video_path = self._mux(v_file, a_file, final_path)
                for part in (v_file, a_file):
                    if os.path.exists(part):
                        os.remove(part)

            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise FileNotFoundError(f"Downloaded file is missing or empty: {video_path}")
            print(f"✅ Download complete! File size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")
            return Path(video_path), title, duration

        except PytubeFixError as e:
            raise Exception(f"Failed to download video: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during download: {str(e)}")
