# UGSL Recognition Engine · Phase 1.1

Collect labelled video samples for a Ugandan Sign Language training dataset. An operator frames the signer, starts a timed take, reviews it, and decides whether to add it to the local dataset. The CLI returns after the requested number of samples has been accepted or the operator quits.

## Installation

Use Python 3.11 or newer on a macOS or Linux desktop with an attached camera. Run the commands from the project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The preview uses the GUI-enabled `opencv-python` package and requires a graphical desktop session.

### macOS camera permissions

Allow camera access when macOS prompts. If access was previously denied, open **System Settings → Privacy & Security → Camera**, enable access for the terminal or other app launching Python if it appears in the list, and rerun the command. [Apple’s camera permission guide](https://support.apple.com/en-lk/guide/mac-help/mchlf6d108da/mac)

## Usage

```bash
python cli.py preview --camera 0
python cli.py capture --sign WATER --signer SIGNER001 --duration 3
python cli.py capture --sign WATER --signer SIGNER001 --duration 3 --repetitions 10
python cli.py capture --sign WATER --signer SIGNER001 --camera 0 \
  --width 1280 --height 720 --fps 30 --duration 5 --countdown 3 --repetitions 10
```

Defaults are camera index 0, 1280 × 720, 30 requested FPS, a three-second recording, a three-second countdown, and one accepted repetition. Set `--countdown 0` for an immediate start. Recordings use MP4 with the MPEG-4 Part 2 (`mp4v`) codec. Sign and signer labels are used as entered. Each label must start with an ASCII letter or digit and contain at most 64 ASCII letters, digits, underscores, or hyphens.

Click the preview window before using its keys. Frame the signer’s head and upper body, and check that their hands stay in view throughout the sign.

| Key | Where it works | Action |
| --- | --- | --- |
| Space | Ready | Start a take |
| Q | Preview or any active capture state | End the session and discard any unaccepted take |
| Esc | Countdown, recording, or review | Cancel the take and return to Ready |
| Esc | Playback | Return to review |
| A | Review | Save the take and advance the sample count |
| R | Review | Discard the take and return to Ready at the same sample number |
| P | Review | Replay the take and return to review |

Press Space again after rejecting a take or when Ready for the next repetition. Terminal Ctrl+C also exits the application and runs session cleanup.

## Dataset and metadata

Each accepted take has a UUID shared by the video and its JSON sidecar. Repetition numbering starts at 1 each time a capture session starts.

```text
datasets/raw/
└── WATER/
    └── SIGNER001/
        ├── <uuid>.mp4
        └── <uuid>.json
```

| Fields | Meaning |
| --- | --- |
| `sample_id`, `gloss`, `signer_id`, `repetition` | UUID, operator-provided labels and the accepted sample number in this session |
| `camera_index`, `width`, `height` | Selected device index and actual video dimensions in pixels |
| `requested_fps` | Frame rate requested from the camera |
| `measured_fps` | Successful frames divided by the elapsed capture time, including transient read failures and capture-loop overhead |
| `encoded_fps` | Playback rate reported when reopening the encoded MP4 |
| `frame_count`, `failed_frame_reads` | Successful captures and unsuccessful camera reads during the take |
| `requested_duration_seconds`, `actual_duration_seconds` | Target take length and elapsed monotonic capture time, excluding countdown, encoding, and review |
| `file_size_bytes`, `file_path` | MP4 byte size and path. CLI paths are relative to the directory used to launch the command |
| `recorded_at` | UTC timestamp at the start of the take in ISO 8601 format |

## How capture works

- `config.py` defines settings and validates labels. `cli.py` exposes the two operator commands and reports device and storage errors.
- `CameraCapture` in `ingestion/capture.py` requests a camera mode, checks that frames arrive, and uses the delivered dimensions. The session moves from Ready through Countdown and Recording to Review. Playback returns to Review. Accepting the final take completes the session.
- The capture loop uses a monotonic clock. Frame retrieval can extend a take slightly beyond its target. Captured frames keep their original dimensions and orientation. The framing guide and labels are drawn on a separate preview image.
- OpenCV’s writer uses a constant frame rate. The session checks that a writer can open before capture, stages the raw frames on disk, then encodes them at the measured average rate. This keeps total playback time close to elapsed capture time. Irregular inter-frame timing and gaps from failed reads are spread across the clip at uniform intervals, and the codec can round the requested encoding rate. MP4V compression is lossy.
- Before offering review, the session reopens the MP4 and checks that every expected frame decodes at the correct dimensions. On acceptance it writes the metadata dataclass from `ingestion/schemas.py` to JSON and publishes the pair without replacing existing files. `ingestion/exceptions.py` defines errors that the CLI can report directly.
- Temporary files live in a hidden `datasets/.capture-*` directory. Raw staging uses about 237 MiB for three seconds at 720p and 30 FPS, plus space for the encoded MP4. Allow time for encoding and validation when the window says **Preparing review**.
- Cleanup runs on normal exit, exceptions, Ctrl+C, and CLI SIGTERM. A force kill or power loss can leave a temporary directory, or a JSON sidecar if interrupted between publishing the sidecar and the video. After confirming the session is no longer running, remove its stale temporary directory and any sidecar with a missing same-name MP4.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Camera unavailable or black preview | Check camera permissions and connection, close other camera apps, and try a different `--camera` index. On Linux, check access to the corresponding `/dev/video*` device. |
| Requested resolution was ignored | The camera backend can select another mode. Check the preview and the sidecar dimensions. Request a supported even-sized mode if recording reports an odd resolution. |
| Preview window fails to open | Run from a graphical desktop with the GUI-enabled OpenCV package. |
| Keyboard shortcuts appear unresponsive | Focus the preview window. During **Preparing review**, wait for the review controls to appear. |
| Clip has stutter or low measured FPS | Compare requested, measured, and encoded FPS and check the failed read count. Try a lower supported resolution or improve lighting. Repeated frame-read failures end the session and discard the pending take. |
| Writer or storage error | Check codec support, available disk space and write permission for `datasets/` and `datasets/raw/`. Staging and final directories need to be on the same filesystem with hard-link support. |

## Automated tests

```bash
python -m pytest -q
```

The suite mocks the camera, writer, decoder, keyboard, window, and clock. It checks state transitions, frame timing, untouched frame delivery to the writer, metadata, rejection and acceptance, repetition progress, and cleanup on failures.

## Manual physical-camera checklist

Use a fresh label such as `CAMERA_TEST` for the checklist so that existing samples do not affect file counts.

1. [ ] Run `python cli.py preview --camera 0`. Confirm a live image and the expected dimensions. Adjust the signer’s position, press Q, then rerun preview to verify the camera was released.
2. [ ] Run `python cli.py capture --sign CAMERA_TEST --signer SIGNER001 --duration 3 --repetitions 2`. Confirm the sign, signer, framing guide, and **Sample 1 of 2**.
3. [ ] Press Space and check the countdown and recording progress. Press Esc during the countdown or take. Verify that the session returns to **Sample 1 of 2** and creates no video or sidecar.
4. [ ] Press Space and complete a take using a clearly visible movement. Press P in Review, then R. Check that the replay returns to Review, rejection returns to Ready, and the sample stays at **1 of 2**.
5. [ ] Record and accept two takes. Confirm the sample advances after the first accept and the session exits after the second. Check that there are two MP4/JSON pairs under `datasets/raw/CAMERA_TEST/SIGNER001/`.
6. [ ] Play the MP4s in an external player. Check that the full frame and original orientation are preserved, that there are no overlays, and that motion and duration look close to the live take. Verify sidecar labels, UUIDs, repetition numbers, dimensions, and byte sizes. Compare `frame_count / encoded_fps` with `actual_duration_seconds`.
7. [ ] Rerun the capture command with the same labels, start a take, then press Q during recording or review. Confirm the accepted count on disk is still two and that the camera can reopen. For a removable camera, also check that disconnecting it early during a longer take reports an error and leaves no new pair.
