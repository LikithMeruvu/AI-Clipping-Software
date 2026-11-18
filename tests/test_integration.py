import asyncio
import shutil
import subprocess

import pytest

from clipper.captions import STYLES, Word, group_words
from clipper.config import Settings
from clipper.media import probe
from clipper.render import RenderJob, render_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@needs_ffmpeg
@pytest.mark.slow
def test_render_produces_vertical_video_with_audio_and_jump_cut(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(src)], check=True)
    style = STYLES["hormozi"]
    words = [Word(0.0, 0.4, "hello"), Word(0.4, 0.9, "world")]
    intervals = [(1.0, 5.0), (8.0, 10.0)]  # 4 s + 2 s kept, a 3 s pause removed
    job = RenderJob(source=src, output=tmp_path / "out.mp4", intervals=intervals,
                    captions=group_words(words, style), style=style, hook="Test hook",
                    progress_bar=True, track_faces=False, encoder="libx264", models_dir=tmp_path)
    result = render_clip(job)
    info = probe(result.path)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio
    assert abs(info.duration - 6.0) < 0.2  # duration equals the sum of kept intervals


def test_mcp_server_registers_expected_tools(tmp_path):
    from clipper.mcp_server import build_server
    server = build_server(Settings(_env_file=None, data_dir=tmp_path, output_dir=tmp_path))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"clip_video", "transcribe_video", "render_clips", "get_job", "list_jobs", "list_styles"} <= names
