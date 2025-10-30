import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Replace pytube with pytubefix
from pytubefix import YouTube
from pytubefix.exceptions import PytubeFixError

from config import TEMP_DIR, YOUTUBE_USER_AGENT


class YouTubeDownloader:
    """
    Handles the downloading of YouTube videos.

    This class uses the pytubefix library to download a YouTube video
    and prepares it for further processing.
    """
    def __init__(self, temp_dir=TEMP_DIR):
        """
        Initializes the YouTubeDownloader.

        Args:
            temp_dir (Path, optional): The directory to save temporary files.
                                       Defaults to TEMP_DIR from config.
        """
        self.temp_dir = temp_dir
        
    def _sanitize_filename(self, filename):
        """
        Sanitizes a filename by removing invalid characters.

        Args:
            filename (str): The filename to sanitize.

        Returns:
            str: The sanitized filename.
        """
        if not filename:
            return "unknown_title"
            
        # Replace characters that are problematic in filenames
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            filename = filename.replace(char, '_')
            
        # Limit filename length to avoid path too long errors
        if len(filename) > 100:
            filename = filename[:97] + '...'
            
        return filename

    @staticmethod
    def get_video_id(url):
        """
        Extracts the video ID from a YouTube URL.

        Args:
            url (str): The YouTube URL.

        Returns:
            str: The video ID, or None if not found.
        """
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        if 'youtube.com' in url:
            return parse_qs(urlparse(url).query).get('v', [None])[0]
        return None

    def download(self, url):
        """
        Downloads a YouTube video from the given URL.

        Args:
            url (str): The URL of the YouTube video.

        Returns:
            tuple: A tuple containing the path to the downloaded video,
                   the video title, and its duration.
        """
        print(f"🔗 Processing YouTube URL: {url}")
        video_id = self.get_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL provided.")
        print(f"✅ Extracted video ID: {video_id}")

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"✅ Temporary directory ready: {self.temp_dir}")
        
        # Set up output path - use absolute path to avoid any path issues
        output_path = os.path.abspath(str(self.temp_dir))
        output_filename = f"{video_id}.mp4"
        print(f"📂 Output will be saved to: {os.path.join(output_path, output_filename)}")
        
        try:
            # Create YouTube object with custom options
            # Note: pytube doesn't directly support setting user agent via class attributes
            # We'll just create the YouTube object normally
            
            print(f"Downloading YouTube video with ID: {video_id}")
            print(f"⏳ Fetching video metadata...")
            # Create YouTube object
            yt = YouTube(url)
            print(f"✅ Connected to YouTube API successfully")
            
            # Get video information and sanitize title
            title = self._sanitize_filename(yt.title)
            duration = yt.length
            
            print(f"Video title: {title}")
            print(f"Video duration: {duration} seconds ({duration//60}m {duration%60}s)")
            print(f"Video author: {yt.author}")
            print(f"⏳ Selecting best quality stream...")
            
            # First try to get progressive stream (combined audio and video)
            stream = (
                yt.streams
                .filter(progressive=True, file_extension='mp4')
                .order_by('resolution')
                .desc()
                .first()
            )
            
            # If no suitable progressive stream is found, try adaptive stream
            if not stream:
                print("⚠️ No progressive stream found, trying adaptive stream")
                stream = (
                    yt.streams
                    .filter(file_extension='mp4')
                    .order_by('resolution')
                    .desc()
                    .first()
                )
            
            if not stream:
                raise ValueError("No suitable video stream found")
            
            print(f"Selected stream: {stream.resolution}, {stream.mime_type}")
            print(f"Stream itag: {stream.itag}, File size: {stream.filesize/(1024*1024):.1f} MB")
            print(f"⏳ Starting download (this may take a while)...")
            
            # Download the video with error handling
            try:
                print(f"⏳ Downloading video to {output_path}...")
                video_path = stream.download(output_path=output_path, filename=output_filename)
                
                # Verify the file exists and has content
                if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                    raise FileNotFoundError(f"Downloaded file is missing or empty: {video_path}")
                
                file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"✅ Download complete! File size: {file_size_mb:.2f} MB")
                    
                # Convert to Path object for consistency with the rest of the code
                video_path = Path(video_path)
                print(f"Download complete: {video_path}")
            except OSError as e:
                # Handle specific OS errors like invalid characters in filename
                print(f"OS Error during download: {e}")
                # Create a temporary file with a safe name
                temp_file = os.path.join(output_path, f"youtube_video_{video_id}.mp4")
                video_path = stream.download(output_path=output_path, filename=f"youtube_video_{video_id}.mp4")
                video_path = Path(video_path)
                print(f"Download complete with safe filename: {video_path}")
            
            return video_path, title, duration
            
        except PytubeFixError as e:
            print(f"PytubeFixError: {str(e)}")
            raise Exception(f"Failed to download video: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise Exception(f"Unexpected error during download: {str(e)}")

        # This should never be reached, but just in case
        video_path = Path(output_path) / output_filename
        if not video_path.exists():
            raise FileNotFoundError("Failed to download the video file.")

        return video_path, "Unknown Title", 0
