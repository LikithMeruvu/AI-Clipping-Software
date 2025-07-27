# YouTube Viral Clipper: AI-Powered Viral Video Creation

## Unleash the Power of AI to Create Viral Short-Form Video Content Effortlessly!

Transform any YouTube video into captivating, shareable short-form clips with the **YouTube Viral Clipper**. This revolutionary, free, and open-source software leverages cutting-edge Artificial Intelligence to automate the entire video clipping process, making viral content creation accessible to everyone.

Whether you're a content creator, marketer, or just looking to share engaging moments, our tool streamlines the workflow from video download to polished, captioned clips, ready to dominate social media feeds.

## Why Choose YouTube Viral Clipper?

- **🚀 AI-Driven Efficiency**: Say goodbye to manual editing! Our intelligent algorithms handle the heavy lifting.
- **💰 Completely Free & Open Source**: Empowering creators without hidden costs. Contribute to a growing community!
- **🌐 Universal Appeal**: Designed for anyone looking to maximize their video content's reach and impact.
- **⚡️ Speed & Simplicity**: Get viral clips in minutes, not hours.

## Features That Make a Difference

- **🎥 Advanced YouTube Video Downloader**: Securely fetches videos in the highest available quality, ensuring your clips look stunning.
- **🧠 Intelligent AI-Powered Clip Selection**: Utilizing state-of-the-art Gemini AI, the system analyzes video transcripts to pinpoint the most engaging, trending, and viral-worthy segments, guaranteeing maximum impact.
- **👁️ Dynamic Face Tracking & Cropping**: Our smart face-tracking technology automatically detects and keeps the speaker perfectly framed, even in dynamic scenes, ensuring professional-grade visual focus.
- **✍️ Engaging Word-by-Word Captions**: Boost viewer retention and accessibility with beautifully animated, synchronized captions available in diverse, customizable styles.
- **⚙️ Seamless Configuration**: Get started in minutes with a straightforward `.env` file setup for all your preferences and API keys.

## Get Started in 3 Simple Steps!

Ready to revolutionize your content creation? Follow these quick steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/youtube-viral-clipper.git
    cd youtube-viral-clipper
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Your Environment:**

    Copy the `.env.example` file to `.env` and add your Gemini API key. This is crucial for enabling the AI-powered features.

    ```bash
    cp .env.example .env
    ```

    Open the newly created `.env` file and insert your API key:

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    *(Optional: Enhance download reliability by setting a custom user agent for YouTube downloads in your `.env` file:)*

    ```
    # Set a custom user agent for YouTube downloads
    YOUTUBE_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ```

## How to Use

Execute the `main.py` script from your terminal:

```bash
python main.py
```

The script will intuitively guide you through the process, prompting for the YouTube URL and other customization options. Your newly generated, viral-ready clips will be conveniently saved in the `clips` directory.

## Troubleshooting & Tips

Encountering issues? Here are some common solutions:

### YouTube Download Challenges

If you face difficulties downloading YouTube videos:

1.  **Explore Different Videos**: Some videos may have regional restrictions or content policies that prevent direct downloading.

2.  **Utilize a Custom User Agent**: As detailed in the setup section, configuring a custom user agent in your `.env` file can often resolve download blocks.

3.  **Keep `pytube` Updated**: YouTube frequently updates its platform, which can affect download functionality. Ensure your `pytube` library is always up-to-date:

    ```bash
    pip install -U pytube
    ```

4.  **Verify Video Accessibility**: Confirm that the video is publicly available and not age-restricted or set to private.

> **Important Note**: This application now exclusively uses `pytube` for YouTube downloads, offering a more robust and reliable download experience compared to previous methods.

## Features

- **YouTube Video Downloader**: Downloads videos from YouTube in the best available quality.
- **AI-Powered Clip Selection**: Uses Gemini to analyze the transcript and select the most engaging and viral-worthy clips.
- **Intelligent Face Tracking**: Automatically detects and crops the video to keep the speaker in the frame.
- **Engaging Captions**: Adds word-by-word captions in various styles to make the clips more engaging.
- **Simple Configuration**: Easy to set up and configure with a `.env` file.

## Setup

Follow these three simple steps to get started:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/youtube-viral-clipper.git
    cd youtube-viral-clipper
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Add your API key and configure settings:**

    Copy the `.env.example` file to a new file named `.env` and add your Gemini API key.

    ```bash
    cp .env.example .env
    ```

    Now, open the `.env` file and add your API key:

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Optional: Configure a custom user agent for YouTube downloads:

    ```
    # Set a custom user agent for YouTube downloads
    YOUTUBE_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ```

## Usage

Run the `main.py` script from your terminal:

```bash
python main.py
```

The script will prompt you for the YouTube URL and other options. The generated clips will be saved in the `clips` directory.

## Troubleshooting

### YouTube Download Issues

If you encounter download issues with YouTube videos:

1. **Try Different Videos**: 
   Some videos may have restrictions that prevent downloading.

2. **Set a Custom User Agent**: 
   Configure a custom user agent in your `.env` file as shown in the setup section.

3. **Update pytube**: 
   YouTube frequently changes their systems, so keeping pytube updated helps:
   ```bash
   pip install -U pytube
   ```

4. **Check Video Availability**: 
   Make sure the video is publicly available and not age-restricted or private.

> Note: This application now uses pytube instead of yt-dlp for YouTube downloads, which should provide a more reliable download experience.

<a href="https://www.buymeacoffee.com/likith_codebook"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=likith_codebook&button_colour=BD5FFF&font_colour=ffffff&font_family=Cookie&outline_colour=000000&coffee_colour=FFDD00" /></a>
