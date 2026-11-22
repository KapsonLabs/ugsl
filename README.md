# UGSL Recognition Engine · Phase 1.3

Collect labelled Ugandan Sign Language video samples and manage the dataset in a local SQLite registry. Operators record takes in OpenCV or import existing local video files. Both paths run through the same technical checks before storing the original video bytes and registering the sample. The registry links each sample to a vocabulary entry and a recording session for an eligible signer.

## Install

Use Python 3.11 or newer on a macOS or Linux desktop. Run these commands from the project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install FFmpeg and FFprobe separately and confirm that both are on PATH before ingesting or validating video. The FFmpeg build must support `-fps_mode passthrough`.

```bash
ffmpeg -version
ffprobe -version
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
| A | Review | Submit the take for ingestion. For standalone capture, save the take directly |
| R | Review | Reject the take and return to the same repetition |
| P | Review | Replay the take |

## Import local video

Create or select an active recording session for a signer whose consent covers the footage. The session determines the signer for each import. Use the IDs printed by the vocabulary and session commands.

```bash
python cli.py video inspect ./water.mp4
python cli.py video validate ./water.mp4
python cli.py ingest file ./water.mp4 --sign UGSL0001 --session SESSION_ID \
  --recorded-at 2026-11-22T09:00:00Z --source-description "Field camera" --notes "Front view"
python cli.py ingest directory ./water_samples --sign UGSL0001 --session SESSION_ID --recursive
python cli.py ingest manifest ./manifest.csv --dry-run
python cli.py ingest manifest ./manifest.csv --json > import-report.json
```

In a CSV manifest, `file_path`, `sign_id`, and `session_id` are required. Optional columns are `recorded_at`, `notes`, and `source_reference`. Relative paths are resolved from the manifest's directory.

```csv
file_path,sign_id,session_id,recorded_at,notes
water_001.mp4,UGSL0001,SESSION_ID,2026-11-22T09:00:00Z,Front view
eat_001.mov,UGSL0002,SESSION_ID,2026-11-22T09:05:00Z,Front view
```

Directory discovery looks for MP4, MOV, AVI, MKV and WebM and skips symlinked files and directories. The importer verifies the actual container and stores the original bytes using a suffix that matches the detected format. The source stays in place. Batch imports process each entry independently unless `--fail-fast` is supplied. Reports include outcomes, elapsed time and stored video bytes. Dry runs report projected video bytes and simulate duplicates within the batch without persisting files or database changes. Use `--json` for a machine-readable report on stdout.

Repeated content returns the existing sample UUID. The same content with a different sign or signer is reported as a metadata conflict. A duplicate by itself is a successful no-op. Rejections, conflicts and operational errors produce a nonzero exit status after the batch report.

## Technical validation

The isolated-sign profile requires a minimum of 640 × 480, 20 FPS, 20 frames and one second. Maximums are 10 seconds and 100 MiB. Resolution compares the long and short edges so portrait videos can qualify. Required metadata that remains unknown after probing and decoding fails the profile. Optional fields such as rotation, bitrate and the container creation timestamp can remain empty.

FFprobe reads container and stream metadata. FFmpeg decodes the full video stream and checks frame count and duration where they can be compared reliably. A successful decode establishes technical readability. A signer may still have started early or finished after the clip ended. OpenCV samples at most eight timestamps by default and reports brightness, exposure, sharpness and motion. Visual issues produce warnings for review. Static signs may have little motion, and dynamic signs may contain blur.

Defaults live in `config.py`. Set an environment variable named `UGSL_QUALITY_<FIELD>` to override a field for the CLI. For example, `UGSL_QUALITY_MIN_FPS=24` raises the frame-rate requirement and `UGSL_QUALITY_VISUAL_SAMPLES=4` changes the number of visual samples. Probe and decode timeouts and quarantine retention follow the same naming rule.

| Technical status | Meaning |
| --- | --- |
| `PENDING` | Technical checks have yet to run, such as for a legacy registry row or a duplicate skipped before validation |
| `PASSED` | Integrity and technical thresholds passed |
| `PASSED_WITH_WARNINGS` | Technical thresholds passed and visual assessment flagged issues |
| `FAILED` | Technical rules or integrity checks failed |
| `ERROR` | Validation could not complete, for example because a media tool timed out |

```bash
python cli.py sample validate SAMPLE_ID
python cli.py sample validate-all
python cli.py sample list --validation-status failed
```

Each registered revalidation adds to the report history and sets the sample's latest technical status. The linguistic status is independent. Revalidation skips reading footage whose signer has no current consent or whose sample is excluded. Use `sample validate-all --dry-run --json` to inspect results without saving reports.

## Storage and schema

```text
datasets/
├── registry.db
├── staging/
├── quarantine/
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
| `video_samples` | Sample UUID, sign/session links, relative paths, source, unique SHA-256, stream metadata, capture metrics, and separate technical and linguistic statuses |
| `video_validations` | Versioned technical and visual reports with a timestamp, profile, metrics, errors, and warnings |
| `ingestion_attempts` | Batch ID, source reference, optional sample ID and SHA-256, outcome, error details, and attempt timestamps. Includes failed attempts without a sample |
| `sign_sequence` | Counter used to allocate permanent vocabulary IDs |

SQLite enforces foreign keys on every connection. Migration 002 copies existing sample rows into the extended schema inside a transaction and preserves IDs, file paths and statuses. Existing rows start with technical validation `PENDING` and an empty checksum until backfilled. Video files stay in the filesystem.

Sidecars hold the original capture or import provenance, video stream metadata, SHA-256, and the initial validation report. `average_fps` and `nominal_fps` describe the stored video. Camera timing fields such as `requested_fps`, `measured_fps` and `actual_duration_seconds` describe the capture. Imports leave those fields empty. A supplied import recording timestamp is normalized to UTC. Without one, `recorded_at` remains empty and the original container creation timestamp is recorded separately when available. Following consent withdrawal and a fresh grant, ingestion requires a verified recording timestamp after that grant.

Registered camera capture preserves the review and replay controls. A submits the staged take to the shared ingestion service. A technical rejection or duplicate returns to Ready at the same repetition. The repetition advances after registration. New samples enter `CAPTURED`, including those with visual warnings. Linguistic review is a later phase.

OpenCV’s constant-rate MP4V writer encodes at the measured capture rate so playback is close to the real duration. Irregular frame arrival times are spread across the clip. The full frame is preserved and overlays appear only in preview. Three seconds at 720p and 30 FPS requires about 237 MiB of temporary frame storage in addition to the MP4.

## Hash backfill and recovery

```bash
python cli.py registry backfill-hashes --dry-run
python cli.py registry backfill-hashes
python cli.py registry reconcile --dry-run
python cli.py registry reconcile
```

Backfill fills missing hashes for unique registered videos. It reports historical duplicate groups and leaves their files and registry records in place. If multiple legacy records share bytes and lack a stored hash, the command leaves that group for manual resolution. It also reports checksums for unregistered legacy pairs. New ingestion can match older rows by hashing likely matches until backfill is complete.

Each in-progress ingestion has a private staging directory with the video and its request journal. The service verifies the checksum again before publication, links the sidecar and video to unused raw paths, and records the sample, validation and completed attempt in one SQLite transaction. Keep raw and staging on the same filesystem with hard-link support.

If storage or SQLite fails, a prepared journal remains for `registry reconcile`. Recovery checks the staged checksum and current consent before retrying registration with the same UUID. Keep the session active until recovery finishes. Reconciliation also reports unknown raw orphans, missing registered files and incomplete journals without inferring consent or deleting files. After a force kill, inspect leftover camera scratch directories named `.capture-*` after the session finishes.

Legacy Phase 1.1 sidecars still use `registry sync`. Register the sign and signer, record consent, and select an active session. With a unique sign and session, sync can match the labels itself. Use `--sign` for legacy gloss variants or `--session` when the signer has several active sessions. Sync preserves UUIDs and timestamps, checks for duplicate bytes, and leaves technical status `PENDING` until you run `sample validate` or `sample validate-all`.

```bash
python cli.py registry sync --dry-run
python cli.py registry sync --session SESSION_ID --sign UGSL0001
```

Quarantine is opt-in for external imports via `--quarantine`. It keeps a restricted copy of a staged technical rejection only when consent permits it. The default retention window is seven days. Reconciliation reports expired copies and copies that lack current consent. Preview with `registry reconcile --dry-run --purge-quarantine` before explicitly removing them with `registry reconcile --purge-quarantine`. External source files remain in place.

## Statistics

```bash
python cli.py stats
python cli.py stats --by sign
python cli.py stats --by signer
python cli.py stats --by session
python cli.py stats --by status
python cli.py stats --coverage --min-signer-samples 5
```

Summary totals include all registered records, recorded duration in seconds, and recorded video size in bytes. Coverage counts `CAPTURED` and `APPROVED` samples from signers with current granted consent. It shows collected samples and unique signers per vocabulary entry, entries with zero registrations, and active signs below their targets. Signer underrepresentation uses the supplied minimum for active signers with granted consent. The default minimum is one.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Local camera unavailable | Check camera permissions and index, close other camera apps, and on Linux check access to the corresponding `/dev/video*` device. |
| Remote camera unavailable | Confirm the stream URL, host reachability, and that no other client is using the bundled sender. Check the server’s camera permissions. |
| Requested local dimensions were ignored | The backend can choose another mode. Check the actual dimensions in preview or metadata. Request an even-sized supported mode for MP4 output. |
| Preview keys appear unresponsive | Focus the OpenCV window. While it says **Preparing review**, wait for encoding and validation. |
| Writer or storage error | Check write access and available space for the dataset and its staging directory. |
| Registry error after accepting a take | Keep the session active and run `registry reconcile --dry-run`, then `registry reconcile`. |
| FFmpeg or FFprobe unavailable | Install both executables and check PATH. Resolve the issue before retrying staged ingestion. |
| Technical rejection | Run `video validate` for the check results. Check framing, capture length, FPS, and resolution for a new take. |

## Automated tests

```bash
python -m pytest -q
```

The suite runs without a physical camera using mocked capture controls and temporary SQLite databases. Synthetic media exercises real FFprobe extraction, FFmpeg decoding, OpenCV visual sampling, and imports. The suite also covers migration from a Phase 1.2 registry, concurrent duplicate detection, dry runs, consent, and interrupted registration.

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

### Phase 1.3 ingestion checks

1. [ ] Use the isolated test database and register a signer, an active session and a sign. Confirm that `video inspect` and `video validate` report the dimensions, FPS and technical checks for a known good local file.
2. [ ] Import that file and compare its SHA-256 with the saved raw video. Confirm the original source remains in place, the JSON sidecar includes validation results, and the registry shows `CAPTURED`.
3. [ ] Rename a copy and import it again. Confirm the original sample UUID is returned. Try the same bytes with a different sign and confirm a metadata conflict.
4. [ ] Dry-run a manifest containing valid, duplicate and corrupt files. Confirm the result summary and that dry-run creates no files or records. Run the real batch and confirm valid rows continue past invalid entries.
5. [ ] Record an otherwise valid take under the default profile with a duration below one second. Accept it and confirm the technical rejection returns you to the same repetition. Accept a valid longer take and check that it advances. Repeat on a remote camera when available.
6. [ ] Run `registry backfill-hashes --dry-run`, `registry reconcile --dry-run`, and `sample validate-all --dry-run`. Review any duplicate groups or orphans. Check `sample list --validation-status failed` after a persisted revalidation of a failed test sample.
