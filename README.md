# YouTube Viral Clipper

**Transform any YouTube video into captivating, shareable short-form clips with the YouTube Viral Clipper.** This free and open-source software leverages AI to automate the entire video clipping process, making viral content creation accessible to everyone.

## Features

- **AI-Powered Clip Selection**: Automatically identifies the most engaging and viral-worthy moments in a video.
- **Smart Face Tracking**: Keeps the speaker perfectly framed, even in dynamic scenes.
- **Engaging Captions**: Adds word-by-word captions in various styles to boost viewer retention.
- **Easy Configuration**: Simple setup with a `.env` file for your API keys and preferences.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/youtube-viral-clipper.git
   cd youtube-viral-clipper
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your environment:**
   Copy the `.env.example` file to `.env` and add your Gemini API key.
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and add your API key:
   ```
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   ```

## Usage

Run the `main.py` script from your terminal:
```bash
python main.py
```
The script will prompt you for the YouTube URL and other options. The generated clips will be saved in the `clips` directory.

## Troubleshooting

If you encounter issues downloading YouTube videos, try the following:

- **Use a custom user agent**: Set a custom user agent in your `.env` file to improve download reliability.
- **Update pytube**: YouTube frequently changes its platform, so keeping `pytube` updated can help:
  ```bash
  pip install -U pytube
  ```
- **Check video availability**: Ensure the video is publicly available and not age-restricted or private.
