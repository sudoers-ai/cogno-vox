"""
cogno_vox.video_sampler — Intelligent keyframe extraction from video streams.

Filters redundant frames using structural difference thresholds (Tier 0 filtering)
to produce a compact sequence of 1..N keyframes for VLLM consumption.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False



def extract_keyframes(
    video_bytes: bytes,
    *,
    max_frames: int = 16,
    scene_threshold: float = 0.3,
    output_format: str = ".jpg"
) -> list[bytes]:
    """Extract distinct keyframes from raw video bytes.

    Uses frame-to-frame histogram difference to detect scene changes.
    If cv2 is unavailable or video parsing fails, returns an empty list.

    :param video_bytes: Raw bytes of the video file (.mp4, .webm, etc.)
    :param max_frames: Maximum number of keyframes to extract.
    :param scene_threshold: Difference threshold (0.0 to 1.0) to declare a scene change.
    :param output_format: Image format for extracted frames (".jpg", ".png").
    :return: List of encoded image bytes for each extracted keyframe.
    """
    if not _HAS_CV2 or not video_bytes:
        return []

    # Write bytes to temporary file for cv2.VideoCapture
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    keyframes: list[bytes] = []
    cap = None
    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            logger.warning("video_sampler: cv2 failed to open video stream")
            return []

        prev_hist: Optional[np.ndarray] = None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 100  # Fallback estimate

        # Step size to avoid reading every single frame on long videos
        step = max(1, total_frames // (max_frames * 4))
        frame_idx = 0

        while cap.isOpened() and len(keyframes) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                # Convert to HSV and calculate color histogram
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                is_keyframe = False
                if prev_hist is None:
                    is_keyframe = True
                else:
                    # Compare histogram correlation (1.0 = identical, < threshold = different)
                    score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    diff = 1.0 - max(0.0, score)
                    if diff >= scene_threshold:
                        is_keyframe = True

                if is_keyframe:
                    prev_hist = hist
                    success, buffer = cv2.imencode(output_format, frame)
                    if success:
                        keyframes.append(buffer.tobytes())

            frame_idx += 1

    except Exception as exc:
        logger.warning("video_sampler: keyframe extraction failed: %s", exc)
        return []
    finally:
        if cap is not None:
            cap.release()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return keyframes
