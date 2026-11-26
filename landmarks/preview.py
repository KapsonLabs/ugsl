"""Optional OpenCV visualization of the streaming RGB output."""

import os
import sys
from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np

from landmarks.exceptions import FrameProcessingError

WINDOW = "UGSL Sampled Frames"
RIGHT = {65363, 2555904, 63235, 3}


def _caption(frame, summary):
    preview = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
    source = f"{summary.source_fps:g}" if summary.source_fps is not None else "unknown"
    target = f"{summary.target_fps:g}" if summary.target_fps is not None else "all frames"
    entries = [
        f"Sample {frame.sample_id}",
        f"Frame {frame.frame_index} | PTS {frame.timestamp_ms // 1} ms | Relative {frame.relative_timestamp_ms // 1} ms",
        f"Source FPS {source} | Target {target}",
    ]
    scale = max(0.26, min(0.7, preview.shape[1] / 950))
    spacing = max(12, round(31 * scale))
    for index, text in enumerate(entries, start=1):
        point = (5, index * spacing)
        cv2.putText(preview, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def preview_frames(run) -> None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise FrameProcessingError("The OpenCV preview requires a graphical desktop session.")
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        for frame in run:
            cv2.imshow(WINDOW, _caption(frame, run.summary))
            while True:
                key = cv2.waitKeyEx(50)
                with suppress(cv2.error):
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                        return
                if key in (ord("q"), ord("Q"), 27):
                    return
                if key in RIGHT or key in (ord("n"), ord("N"), ord(" ")):
                    break
    except cv2.error as exc:
        raise FrameProcessingError("Could not display the OpenCV preview. Use a graphical desktop session.") from exc
    finally:
        with suppress(cv2.error):
            cv2.destroyWindow(WINDOW)


def save_contact_sheet(run, destination: Path, limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 12:
        raise FrameProcessingError("Choose a contact sheet limit from 1 to 12 frames.")
    if destination.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise FrameProcessingError("Choose a PNG or JPEG output path for the contact sheet.")
    tiles = []
    tile_width, tile_height = 320, 210
    try:
        for frame in run:
            if len(tiles) == limit:
                continue
            scale = min(tile_width / frame.width, (tile_height - 24) / frame.height, 1)
            image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            x = (tile_width - image.shape[1]) // 2
            tile[:image.shape[0], x:x + image.shape[1]] = image
            label = f"#{frame.frame_index} | PTS {frame.timestamp_ms // 1} ms | +{frame.relative_timestamp_ms // 1} ms"
            cv2.putText(tile, label, (8, tile_height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        columns = min(3, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            y, x = divmod(index, columns)
            sheet[y * tile_height:(y + 1) * tile_height, x * tile_width:(x + 1) * tile_width] = tile
        ok, encoded = cv2.imencode(destination.suffix.lower(), sheet)
        if not ok:
            raise FrameProcessingError("Could not encode the contact sheet.")
        with destination.open("xb") as output:
            output.write(encoded.tobytes())
    except cv2.error as exc:
        raise FrameProcessingError("Could not create the OpenCV contact sheet.") from exc
    except OSError as exc:
        raise FrameProcessingError("Could not save the contact sheet. Use a new path in a writable directory.") from exc
