"""Unit tests for cogno_vox.video_sampler."""

from cogno_vox.video_sampler import extract_keyframes


def test_extract_keyframes_empty_input():
    frames = extract_keyframes(b"")
    assert frames == []


def test_extract_keyframes_invalid_video_bytes():
    frames = extract_keyframes(b"NOT_A_VIDEO_FILE")
    assert frames == []
