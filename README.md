# UGSL Recognition Engine · Phase 1.2

Collect labelled Ugandan Sign Language video samples and manage the dataset in a local SQLite registry. Operators record and review each take in OpenCV. The registry links accepted video files to vocabulary entries and recording sessions for registered signers.

## Install

Use Python 3.11 or newer on a macOS or Linux desktop. Run these commands from the project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The preview requires the GUI-enabled OpenCV package and a graphical desktop. On macOS, grant camera access when prompted. If permission was denied, open **System Settings → Privacy & Security → Camera** and enable access for the app launching Python. [Apple’s camera permission guide](https://support.apple.com/en-lk/guide/mac-help/mchlf6d108da/mac)

The CLI initializes `datasets/registry.db` when a registry operation first needs it. Override the database and raw dataset paths using global options before the command or using environment variables.

```bash
python cli.py --db /path/to/registry.db --dataset-root /path/to/raw stats
export UGSL_REGISTRY_DB=/path/to/registry.db
export UGSL_DATASET_ROOT=/path/to/raw
```

Use the same database and dataset root together. Registry paths are relative to that root.

## Register signs and signers

```bash
python cli.py sign add --gloss WATER --category "Food and drinks" --type dynamic --target-samples 100
python cli.py sign add --gloss WATER --variant CENTRAL --type dynamic
python cli.py sign list --category "Food and drinks" --type dynamic --active
python cli.py sign show UGSL0001
python cli.py sign update UGSL0001 --target-samples 150
python cli.py sign deactivate UGSL0002

python cli.py signer add --code SIGNER001 --dominant-hand right --proficiency fluent
python cli.py signer consent SIGNER001 --status granted --reference CONSENT001
python cli.py signer list
python cli.py signer show SIGNER001
python cli.py signer update SIGNER001 --region Central
```

Have UGSL experts establish vocabulary entries, meanings, and variants. The registry normalizes glosses for lookup and gives every entry its own permanent ID. A shared English gloss can identify multiple entries with distinct variant labels. Categories are free text. Renaming a gloss leaves its ID, sample links, and file paths intact.

Use a pseudonymous signer code and an opaque consent reference. Store consent documents separately. Signers start with `UNKNOWN` consent, and only active signers with granted consent can contribute training recordings. To deactivate a signer, run `python cli.py signer deactivate SIGNER001`.

### Consent withdrawal

```bash
python cli.py signer withdraw SIGNER001 --reference WITHDRAWAL001
python cli.py sample list --status excluded
```

Withdrawal records the consent change, marks the signer’s registered samples `EXCLUDED`, and reports the affected MP4 and JSON paths for retention decisions. Files stay on disk. A subsequent grant permits fresh recordings while previously excluded samples stay excluded. Synchronization also blocks previously unregistered samples that predate that fresh grant. Check any saved-but-unregistered files separately when handling withdrawal.

## Preview and record

```bash
python cli.py preview --camera 0
python cli.py session create --signer SIGNER001 --camera-type local --angle front --lighting natural
python cli.py session list
python cli.py session show SESSION_ID
python cli.py capture --sign UGSL0001 --session SESSION_ID --camera 0 --duration 3 --repetitions 10
python cli.py sample list --sign UGSL0001
python cli.py sample show SAMPLE_ID
python cli.py session complete SESSION_ID
```

Replace `SESSION_ID` with the ID printed by `session create` and `SAMPLE_ID` with a registered sample UUID. Record different signs by running `capture` again with the same active session ID and another sign ID. After checking the registered count, use `session complete` to close a finished session or `session cancel` to close an abandoned one. Both close the session to new samples and retain its existing records.

Local capture requests 1280 × 720 at 30 FPS by default. Camera mode, recording duration, countdown duration, and repetitions can be changed with `--width`, `--height`, `--fps`, `--duration`, `--countdown`, and `--repetitions`. Defaults are three seconds for recording, three seconds for countdown, and one accepted take. A zero countdown starts capture immediately.

### Remote camera

Use an `HTTP(S)` or `RTSP(S)` stream readable by the OpenCV FFmpeg backend. The remote stream supplies the frame dimensions and cadence. On a camera host, the bundled MJPEG sender can serve one client at a time. It binds to loopback by default and serves an unauthenticated stream, so use a trusted network and set the camera host’s LAN address explicitly if another machine needs to connect.

```bash
# On the camera host. Replace the address with that host's LAN address.
python -m tools.camera_server --bind 192.168.1.105 --port 8080 --camera 0

# On the recording workstation.
python cli.py preview --source http://192.168.1.105:8080/stream
python cli.py session create --signer SIGNER001 --camera-type remote --angle front --lighting natural
python cli.py capture --sign UGSL0001 --session SESSION_ID \
  --source http://192.168.1.105:8080/stream --duration 3 --repetitions 10
```

Close preview before capturing when using the bundled sender. Keep stream credentials out of camera URLs. The registry and sidecars contain no stream URL.

### Standalone camera check

The existing `--sign` and `--signer` syntax still works for standalone test recordings. They go to `datasets/standalone/` with default paths, outside the registry’s raw dataset scan.

```bash
python cli.py capture --sign CAMERA_TEST --signer TEST001 --duration 3 --repetitions 2
```

### Keyboard controls

Focus the camera window before using the keys. Frame the signer’s head and upper body, keeping their hands visible throughout the sign.

| Key | State | Action |
| --- | --- | --- |
| Space | Ready | Start countdown and capture |
| Q | Preview or active capture | Quit and discard the unaccepted take |
| Esc | Countdown, recording, or review | Cancel the take and return to Ready |
| Esc | Playback | Return to Review |
| A | Review | Accept and save the take. Registered capture also writes its registry record |
| R | Review | Reject the take and return to the same repetition |
| P | Review | Replay the take |

## Storage and schema

```text
datasets/
├── registry.db
├── raw/
│   └── UGSL0001/SIGNER001/
│       ├── <uuid>.mp4
│       └── <uuid>.json
└── standalone/
    └── CAMERA_TEST/TEST001/<uuid>.mp4
```

| Table | Purpose |
| --- | --- |
| `signs` | Permanent sign ID, gloss, variant, category, description, type, target count, active flag, and timestamps |
| `signers` | UUID, pseudonymous code, optional background attributes, current consent and reference, active flag, and timestamps |
| `signer_consent_events` | Timestamped consent changes and optional references |
| `recording_sessions` | UUID linked to one signer, operator, camera and environment details, notes, status, and start and end times |
| `video_samples` | UUID shared with the files, sign and session foreign keys, sample type, repetition, relative paths, metrics, status, and timestamps |
| `sign_sequence` | Counter used to allocate permanent vocabulary IDs |

SQLite enforces foreign keys on every connection. Versioned SQL in `registry/migrations/` runs transactionally and tracks its version with `PRAGMA user_version`. A unique constraint covers normalized gloss and variant. The sample UUID and video path also have unique constraints. The session determines the signer for each registered sample. Videos stay in the filesystem.

The JSON sidecar records the camera index for local takes, the gloss at capture, the signer code, and the repetition within that invocation. Registered captures add stable sign and session IDs. It also records frame counts, failed reads, dimensions, MP4 byte size, requested and measured durations, and the UTC capture timestamp. `requested_fps` is the requested camera rate, `measured_fps` is successful frames divided by elapsed capture time, and `encoded_fps` is read back from the saved video.

The camera workflow in `ingestion/` saves the accepted pair before invoking the callback from `services/recording_service.py`. The service rechecks eligibility before each take and inside the SQLite registration transaction. The repetition advances only after registration succeeds. Operator acceptance sets `CAPTURED`. Linguistic review will use `APPROVED` and `REJECTED` in a later phase.

OpenCV’s constant-rate MP4V writer encodes at the measured capture rate so playback is close to the real duration. Irregular frame arrival times are spread across the clip. The full frame is preserved and overlays appear only in preview. Three seconds at 720p and 30 FPS requires about 237 MiB of temporary frame storage in addition to the MP4. Staging and accepted files need a filesystem with hard-link support. Use Q or terminal Ctrl+C for normal cleanup. After a force kill, check for stale `.capture-*` directories and orphaned sidecars once the session has stopped.

## Synchronization and recovery

```bash
python cli.py registry sync --dry-run
python cli.py registry sync
python cli.py registry sync --session SESSION_ID
python cli.py registry sync --session SESSION_ID --sign UGSL0001
```

Sync recursively scans the raw dataset directory for JSON sidecars and validates each same-name MP4 and its metadata. It preserves sample UUIDs and recording times and skips matching registrations on subsequent runs. A dry run reports proposed registrations without adding records, migrating the database, or creating it on disk.

New sidecars identify their sign and session. For legacy sidecars, register the signer and sign first, record the signer’s consent separately, and use an active session for that signer. Sync infers a session when exactly one is active. Provide `--session` to resolve multiple active sessions and `--sign` for an ambiguous legacy gloss with several variants. The chosen sign must match the legacy gloss. Explicit selections that conflict with an unregistered sidecar are reported. Each sidecar’s recorded path must agree with its actual video location. The scan reports invalid metadata, missing media, unsafe paths, and unresolved references. Valid pairs are processed even if others fail, and the command exits with a nonzero status when there are issues.

If registration fails after a take is accepted, the CLI reports its UUID and preserves the pair. Resolve the error, run sync, and inspect the sample before completing the session. Sync can register a saved pair once and report it as skipped on retries. Keep the session active until all recoverable files are registered.

## Statistics

```bash
python cli.py stats
python cli.py stats --by sign
python cli.py stats --by signer
python cli.py stats --by session
python cli.py stats --by status
python cli.py stats --coverage --min-signer-samples 5
```

Summary totals include all registered records, recorded duration in seconds, and recorded MP4 size in bytes. Coverage counts `CAPTURED` and `APPROVED` samples from signers with current granted consent. It shows collected samples and unique signers per vocabulary entry, entries with zero registrations, and active signs below their targets. Signer underrepresentation uses the supplied minimum for active signers with granted consent. The default minimum is one.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Local camera unavailable | Check camera permissions and index, close other camera apps, and on Linux check access to the corresponding `/dev/video*` device. |
| Remote camera unavailable | Confirm the stream URL, host reachability, and that no other client is using the bundled sender. Check the server’s camera permissions. |
| Requested local dimensions were ignored | The backend can choose another mode. Check the actual dimensions in preview or metadata. Request an even-sized supported mode for MP4 output. |
| Preview keys appear unresponsive | Focus the OpenCV window. While it says **Preparing review**, wait for encoding and validation. |
| Writer or storage error | Check write access and available space for the dataset and its staging directory. |
| Registry error after saving a take | Keep the session active and reconcile the saved pair with `registry sync --dry-run`, then `registry sync`. |

## Automated tests

```bash
python -m pytest -q
```

The tests use temporary SQLite databases and mocked camera, video, window, and keyboard interfaces. They cover the original capture workflow, migrations and foreign keys, consent, sessions, registration, recovery, synchronization, and coverage.

## Manual testing checklist

Use a fresh test database and raw dataset directory. Set `UGSL_REGISTRY_DB` and `UGSL_DATASET_ROOT` as shown above so each command uses the same test locations.

1. [ ] Run local preview, focus the window, and verify that Q releases the device. Run a standalone capture. Confirm that files land outside raw and `sample list` is empty.
2. [ ] Add a sign and a variant, and register a signer. Confirm that session creation fails with unknown consent. Record granted consent and create a local session.
3. [ ] Request two registered repetitions. Cancel one take with Esc, replay and reject one with P and R, then accept two. Verify that the repetition advances only after acceptance and both sample statuses are `CAPTURED`.
4. [ ] Play accepted MP4s in an external player. Check orientation, duration, full framing, and absence of overlays. Check that UUIDs, dimensions, sign and session IDs, and stored relative paths agree with the registry and sidecars.
5. [ ] Capture another sign in the same session. Check `stats --by session` and `stats --coverage`. Rename that sign and confirm the sample remains associated with the same ID and file.
6. [ ] Run sync dry-run and sync. Confirm that already registered samples are skipped. If you have a legacy pair, place it in the test raw directory with a valid path, map its sign and signer to an active session, and verify that dry-run proposes it while sync registers it once.
7. [ ] Complete the local session and confirm it rejects new submissions. Open a remote session. Preview a remote stream, close preview, then record and accept a take with `--source`. Verify the remote session linkage and playback.
8. [ ] Withdraw the test signer’s consent. Check the reported paths, confirm the files remain and the sample statuses are `EXCLUDED`, and verify that new sessions and contributions are blocked. Confirm that coverage drops while the total registered sample count stays the same.
