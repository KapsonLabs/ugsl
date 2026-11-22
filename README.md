# UGSL Recognition Engine · Phase 1.5

Collect labelled Ugandan Sign Language video samples and manage the dataset in a local SQLite registry. Operators record takes in OpenCV or import existing local video files. Both paths run through the same technical checks before storing the original video bytes and registering the sample. The registry links each sample to a vocabulary entry and a recording session for an eligible signer. Human annotators mark the sign interval, and linguistic reviewers verify the annotation before the sample becomes training-eligible. Versioned exports freeze the approved clips, annotation intervals, and class mapping for isolated-sign training.

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
├── exports/
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
| `reviewers` | Local reviewer code, declared proficiency and role, optional qualification notes, and active status |
| `annotations` | Immutable sign and interval versions linked to the sample, annotator, and prior version |
| `review_decisions` | Append-only reviewer decisions scoped to an annotation version and review round, with reasons and adjudication flags |
| `review_events` | Review start, reopen, and variant-request history |
| `dataset_versions` | Named semantic versions, artifact schema, lifecycle, manifest checksum, content fingerprint, and the effective local export configuration |
| `dataset_members` | Snapshot of each included sample, exact annotation and approval round, verified sign, signer, and entry and source checksums |
| `dataset_events` | Dataset lifecycle transitions, timestamps, and revocation or failure reasons |
| `sign_sequence` | Counter used to allocate permanent vocabulary IDs |

SQLite enforces foreign keys on every connection. Migration 002 copies existing sample rows into the extended schema inside a transaction and preserves IDs, file paths and statuses. Rows migrated from schema 1 start with technical validation `PENDING` and can be backfilled with checksums. Migration 003 adds review state, current annotation, and review round to `video_samples`. Existing samples start in review state `PENDING`, retain their assigned sign, and gain no approval through migration. Migration 004 adds export snapshots and lifecycle auditing. Video files stay in the filesystem.

Sidecars hold the original capture or import provenance, video stream metadata, SHA-256, and the initial validation report. `average_fps` and `nominal_fps` describe the stored video. Camera timing fields such as `requested_fps`, `measured_fps` and `actual_duration_seconds` describe the capture. Imports leave those fields empty. A supplied import recording timestamp is normalized to UTC. Without one, `recorded_at` remains empty and the original container creation timestamp is recorded separately when available. Following consent withdrawal and a fresh grant, ingestion requires a verified recording timestamp after that grant.

Registered camera capture preserves the review and replay controls. A submits the staged take to the shared ingestion service. A technical rejection or duplicate returns to Ready at the same repetition. The repetition advances after registration. New samples enter `CAPTURED`, including those with visual warnings. Linguistic review is a separate step after ingestion.

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

## Annotation and linguistic review

Register the people who will annotate and verify isolated signs. Reviewer codes provide local accountability. The operator is responsible for checking qualifications before assigning a role or proficiency.

```bash
python cli.py reviewer add --code REVIEWER001 --proficiency fluent --role reviewer
python cli.py reviewer list
python cli.py reviewer show REVIEWER001
python cli.py annotation queue --sign UGSL0001 --status pending
python cli.py annotation open SAMPLE_ID --reviewer REVIEWER001
python cli.py annotation run --reviewer REVIEWER001 --sign UGSL0001 --priority
```

Replace `SAMPLE_ID` with the actual UUID shown in the queue. The queue includes technically passed, accessible recordings with current granted signer consent. `--sign` matches either the assigned sign or the current annotated sign. Other filters include signer code or UUID, session, technical status, review state, current annotation author or review participant, and source (`CAMERA`, `IMPORT`, `LOCAL`, or `REMOTE`). `--since` and `--until` use UTC registration dates. Select `--order oldest`, `newest`, or `sign`. `--priority` puts second review and correction first.

### Annotation window

Focus the OpenCV window to use the keyboard. When it directs you to the terminal, playback pauses while you enter a sign ID, rejection code, or other details.

| Key | Action |
| --- | --- |
| Space | Pause or resume |
| Left / Right | Move one decoded frame backward or forward |
| J / L | Seek backward or forward by about 500 milliseconds |
| I / O | Mark the sign start or end. O uses the end of the clip when the final frame is displayed |
| P | Replay the marked interval |
| + / − | Change playback speed through 0.25×, 0.5×, 1×, and 2× |
| S | Save the marked interval, with an optional label change |
| A | Approve the saved current annotation |
| R | Reject with a standard reason code |
| U | Request second review |
| N / Q | Move to the next eligible sample or exit. The tool asks before discarding an unsaved draft |

The overlay shows the original gloss, current annotation or variant, signer code, duration, position, frame number, markers, speed, review state, and technical warnings. FFprobe supplies source presentation timestamps relative to the first displayed frame. OpenCV decodes in sequence and restarts from the beginning to step backward, which can take longer for clips with many frames. On variable-rate clips, seeks land on an available presentation timestamp. Timing marks use integer milliseconds and represent one interval in the original video. The tool reports media whose backend cannot supply a usable presentation timeline or decode the requested frame. Saved annotations survive interruption. Overlays and boundaries stay separate from the original video.

### Manual annotation, correction, and review

```bash
python cli.py annotation create SAMPLE_ID --reviewer REVIEWER001 --sign UGSL0001 \
  --start-ms 500 --end-ms 2400 --hand-usage two-handed --movement-type dynamic
python cli.py review start SAMPLE_ID --reviewer REVIEWER001
python cli.py review approve SAMPLE_ID --reviewer REVIEWER001
python cli.py review reject SAMPLE_ID --reviewer REVIEWER001 --reason incomplete-sign
python cli.py review request-correction SAMPLE_ID --reviewer REVIEWER001 --reason invalid-temporal-boundaries
python cli.py review request-second SAMPLE_ID --reviewer REVIEWER001 --notes "Verify regional variation"
python cli.py review reopen SAMPLE_ID --reviewer REVIEWER001 --reason "Check the end boundary"
python cli.py annotation correct SAMPLE_ID --reviewer REVIEWER001 --sign UGSL0002 --reason "Verified regional variant"
python cli.py annotation request-variant SAMPLE_ID --reviewer REVIEWER001 --notes "Ask the vocabulary team to review this variant"
python cli.py annotation history SAMPLE_ID
python cli.py review history SAMPLE_ID
python cli.py sample eligibility SAMPLE_ID
```

The review commands shown are alternatives for the current state. An annotation advances from `PENDING` to `ANNOTATED` on save. A reviewer can start review explicitly, approve or reject, request a corrected version, or ask for independent review. Corrections create a new version and require a reason. If the initial annotation changes the assigned sign, provide `--reason` when creating it. Select a registered variant by sign ID. Variant requests preserve notes for the vocabulary team. The original sign assignment and video path stay intact.

Review decisions require an active reviewer or adjudicator with declared proficiency of `FLUENT`, `NATIVE`, or `QUALIFIED_INTERPRETER`. The default `SINGLE` policy allows the same qualified reviewer to annotate and approve. Set `UGSL_REVIEW_POLICY=DUAL` to require a different reviewer for verification. Under dual review, an initial rejection calls for independent confirmation. If the independent verdicts disagree, a third reviewer with the `ADJUDICATOR` role resolves the conflict or requests a correction. The original annotator and the conflicting voters cannot adjudicate that round. Explicit requests for second review also route to a different reviewer. Use one policy consistently across terminals.

Confidence is a human score. The default range is 0–1 and can be configured with `UGSL_ANNOTATION_CONFIDENCE_MIN` and `UGSL_ANNOTATION_CONFIDENCE_MAX`. A CLI decision can include `--annotation-id` and `--round` to guard against a stale version or review round. The annotation window supplies both automatically.

Reopening an approval starts a fresh round on the same annotation. Correcting an annotation starts a fresh round on a new version. Either action returns the sample to `CAPTURED` pending approval. Training eligibility checks current consent, readable media and integrity, passed technical validation, and a final approval for the current version and round under the applicable policy. The approved annotation supplies the verified sign ID. A withdrawn signer’s samples remain `EXCLUDED`, with their history retained.

## Statistics

```bash
python cli.py stats
python cli.py stats --by sign
python cli.py stats --by signer
python cli.py stats --by session
python cli.py stats --by status
python cli.py stats --coverage --min-signer-samples 5
python cli.py stats --annotations
python cli.py stats --reviews
python cli.py stats --approved-coverage
```

Summary totals include all registered records, recorded duration in seconds, and recorded video size in bytes. Coverage counts `CAPTURED` and `APPROVED` samples from signers with current granted consent. It shows collected samples and unique signers per vocabulary entry, entries with zero registrations, and active signs below their targets. Signer underrepresentation uses the supplied minimum for active signers with granted consent. The default minimum is one. Collection coverage counts original assignments. Approved coverage shows those assignments alongside verified labels and training-eligible counts. Review completion uses current finalized decisions divided by annotated samples. Approval and rejection rates use current finalized outcomes. Agreement compares only rounds with two independent verdicts on the same annotation and round, excluding adjudication. The agreement percentage is empty when there are no comparable rounds.

## Export versioned datasets

Export pulls from the current approved annotation. The original capture label and path stay in the registry. Eligible samples have granted training consent, a current approval under the applicable review policy, passed technical validation, an active verified sign, and a source video that matches its registered SHA-256. Run `registry backfill-hashes` for eligible legacy samples that lack a registered checksum. The export preview lists excluded samples and their reasons.

```bash
python cli.py dataset export --name ugsl-isolated-signs --version 1.0.0 --mode reference --dry-run
python cli.py dataset export --name ugsl-isolated-signs --version 1.0.0 --mode reference
python cli.py dataset export --name ugsl-isolated-signs --bump minor --mode portable --csv
python cli.py dataset export --name ugsl-basic --sign UGSL0001 --sign UGSL0002 \
  --min-samples-per-sign 10 --min-signers-per-sign 3
python cli.py dataset list
python cli.py dataset show ugsl-isolated-signs --version 1.0.0
python cli.py dataset stats ugsl-isolated-signs --version 1.0.0
```

Use `--version` for explicit SemVer or `--bump patch|minor|major` for an operator-selected increment. A new name starts at `1.0.0`. Omit both options after that to increment the patch. Failed and revoked version numbers remain reserved. The manifest schema version is independent of the dataset version.

`--sign`, `--category`, `--signer`, and `--session` can be repeated. Sign filters use verified IDs. A signer filter accepts a pseudonymous code or UUID. Coverage applies to explicitly requested signs and categories, including zero-sample classes. Without a sign or category filter it applies to the represented classes. An unmet threshold blocks export unless `--allow-incomplete-coverage` is given. Every export needs at least one eligible sample. Use `--exclude-warnings` to omit samples that passed with technical warnings. With `--max-samples-per-sign`, the lexically lowest sample UUIDs are selected within each verified sign. No other balancing is performed.

Dry-run does not reserve versions or write artifacts. Disabling `--verify-all-source-checksums` omits the extra checksum pass over selection candidates in a real export. Artifact verification and the final checks on selected source bytes remain mandatory.

### Reference and portable storage

The default destination is `exports/` beside the raw dataset. Use `--output-directory` to change it. Reference exports resolve relative video paths from the selected `--storage-root`, which defaults to the raw directory. They depend on access to that original storage. Portable exports check disk space and copy the original encoded video bytes and extensions. They can move between directories or machines with no change to the manifest.

```text
datasets/exports/ugsl-isolated-signs-v1.1.0/
├── dataset.json
├── manifest.jsonl
├── manifest.csv          # If requested
├── vocabulary.json
├── statistics.json
├── checksums.sha256
├── README.md
└── videos/               # Portable mode
    └── UGSL0001/
        └── <sample-uuid>.mp4
```

JSON Lines is authoritative. Below is one example record, formatted across several lines for readability. A real manifest stores each record on one line.

```json
{
  "sample_id": "ec8873a6-841d-45ca-a5e4-716998e819f0",
  "sign_id": "UGSL0001",
  "gloss": "WATER",
  "class_index": 0,
  "signer_id": "SIGNER001",
  "session_id": "d5e6b52e-65cf-4544-b62f-88f029494471",
  "video_path": "videos/UGSL0001/ec8873a6-841d-45ca-a5e4-716998e819f0.mp4",
  "sha256": "1dc5b41c675d7c020d665c2d96f5b66c91774c7e80a9c50a351d9ca5debfdd02",
  "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
  "video_codec": "mpeg4",
  "width": 1280,
  "height": 720,
  "average_fps": 30.0,
  "frame_count": 90,
  "duration_ms": 3000,
  "orientation": "LANDSCAPE",
  "rotation": null,
  "annotation": {
    "id": "8dab297a-e01d-46bf-b448-8ffe09d40633",
    "version": 2,
    "start_ms": 500,
    "end_ms": 2400,
    "hand_usage": "TWO_HANDED",
    "dominant_hand": "RIGHT",
    "movement_type": "DYNAMIC",
    "facial_expression": "UNKNOWN",
    "completeness": "COMPLETE",
    "confidence": 0.9
  },
  "technical_validation_status": "PASSED",
  "source_type": "CAMERA",
  "recorded_at": "2026-11-22T09:00:00+00:00",
  "file_size_bytes": 1500000
}
```

Classes are indexed from zero in permanent sign-ID order. Indices may change between exports. Keep the exact `vocabulary.json` from the training dataset with any model that uses its class indices. The content fingerprint includes the ordered records, annotation versions, class mapping, and effective selection options. Dataset UUIDs, timestamps, dataset version labels, and storage paths are outside that fingerprint. Reference and portable versions can therefore represent equivalent training content. Reports distinguish video bytes represented by a dataset from extra disk space needed for portable copying.

### Verify, compare, and read

```bash
python cli.py dataset verify ugsl-isolated-signs --version 1.0.0
python cli.py dataset verify --path ./datasets/exports/ugsl-isolated-signs-v1.1.0
python cli.py dataset compare ugsl-isolated-signs --from-version 1.0.0 --to-version 1.1.0 --json
```

For a reference path verification, pass `--storage-root` if the export uses a different root from `--dataset-root`. Add `--registry PATH` (or the global `--db` option) to consult a registry for a moved dataset. Integrity verification checks artifact and video checksums, schema, vocabulary, annotation bounds, totals, and registered membership when available. Permission is reported separately. An intact dataset can be revoked. Standalone verification with no registry or revocation notice reports lifecycle permission as unknown.

```python
from dataset_export.reader import DatasetReader

dataset = DatasetReader(
    "datasets/exports/ugsl-isolated-signs-v1.1.0",
    registry_path="datasets/registry.db",
)
for sample in dataset.iter_samples(sign_id="UGSL0001"):
    print(sample.sample_id, sample.sign_id, sample.video_path)
    print(sample.annotation.start_ms, sample.annotation.end_ms)
```

For reference exports, supply `storage_root=` or set `UGSL_REFERENCE_ROOT`. The reader checks essential manifest fields and known lifecycle status. Run the verifier for full artifact and video integrity. It rejects revoked versions when the registry or a matching revocation notice is available. A portable reader can consume a copy offline without a registry when there is no known revocation. In that case obtain current permission before use or redistribution. The reader uses the standard library and yields paths and metadata.

### Lifecycle and recovery

```bash
python cli.py dataset supersede ugsl-isolated-signs --version 1.0.0
python cli.py dataset affected --signer SIGNER001
python cli.py dataset affected --sample SAMPLE_ID
python cli.py dataset revoke ugsl-isolated-signs --version 1.1.0 --reason "Signer consent withdrawn"
python cli.py dataset reconcile --dry-run
python cli.py dataset reconcile
```

Superseded versions remain available for historical reproducibility. Consent loss, an invalidated approval or technical result, a changed registered checksum, or deactivation of a verified vocabulary entry revokes affected versions in the same registry transaction. Reapproval or a fresh consent grant does not reactivate revoked versions. Export a new version for the replacement dataset.

The registry is authoritative. Explicit revocation and the CLI consent and correction commands write local notices when possible. Reconciliation writes missing notices to registered export folders. The separate `revocation.json` notice is outside the artifact checksums, so revocation preserves the manifest and its integrity. Send the notice to recipients when a dataset leaves the registry host. Use `dataset affected` to identify known portable copies. Stop using those copies and coordinate removal of exported media and any dependent artifacts with the custodian. Copies stored elsewhere and later training artifacts require a separate inventory.

A new export reserves its version and membership in `BUILDING`, writes and verifies a private staging folder, then rechecks approval and consent before renaming and activating it. A build marker blocks use until activation finishes. Reconciliation can finalize a complete eligible build or mark an incomplete build `FAILED`. Staged files from failed or revoked builds remain private for inspection and custodian-directed cleanup. Dry-run performs no cleanup or lifecycle changes.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Local camera unavailable | Check camera permissions and index, close other camera apps, and on Linux check access to the corresponding `/dev/video*` device. |
| Remote camera unavailable | Confirm the stream URL, host reachability, and that no other client is using the bundled sender. Check the server’s camera permissions. |
| Requested local dimensions were ignored | The backend can choose another mode. Check the actual dimensions in preview or metadata. Request an even-sized supported mode for MP4 output. |
| Preview keys appear unresponsive | Focus the OpenCV window. While it says **Preparing review**, wait for encoding and validation. |
| Writer or storage error | Check write access and available space for the dataset and its staging directory. |
| Interrupted dataset build | Run `dataset reconcile --dry-run`, inspect the report, then run `dataset reconcile`. |
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

### Phase 1.4 annotation checks

1. [ ] In your test dataset, validate or ingest a clip with a known sign interval. Register a qualified reviewer and confirm the clip appears in `annotation queue --status pending`. Confirm a technically failed clip stays out of the queue.
2. [ ] Open the clip. Use Left and Right to step through it, use J and L to seek, change speed, mark the boundaries with I and O, and replay the interval with P. Try leaving an unsaved draft, decline the discard prompt, and save it with S. Confirm the original video checksum is unchanged.
3. [ ] Approve under the default policy and check `sample eligibility`. Correct the annotation to a registered variant and confirm the original label and file path remain unchanged, both versions appear in history, and the sample needs fresh approval.
4. [ ] Approve the current version, then reopen it. Check that the saved annotation remains current and the older approval no longer counts.
5. [ ] Set `UGSL_REVIEW_POLICY=DUAL`. Confirm the author cannot approve their own annotation. Have a different reviewer reject, a second independent reviewer approve, and an adjudicator resolve the conflict. Check that the two opinions and the adjudication are all in review history.
6. [ ] Run the annotation, review, and approved-coverage statistics. Check the distinction between original and verified labels. Withdraw the test signer’s consent and confirm their video no longer appears in the queue or counts as training-eligible.

### Phase 1.5 export checks

1. [ ] Use a test registry with approved, pending, and excluded clips. Preview a reference export and check its exclusion reasons. Set an unreachable coverage threshold and confirm the export is blocked.
2. [ ] Export a reference dataset and verify it against the configured raw root. Export a portable version with CSV, move a test copy, and verify it without the registry. Compare its video SHA-256 with the original.
3. [ ] Export the same selection again and compare versions. Confirm matching fingerprints, class mappings, and annotation versions. Correct and reapprove a sample, then export a replacement and check that comparison reports the changed annotation.
4. [ ] Modify a video copy in a disposable export and confirm integrity verification fails. Confirm a missing reference video is also reported.
5. [ ] Withdraw the test signer's consent. Use `dataset affected` to find the versions, verify that an intact revoked version reports use blocked, and confirm the reader refuses it with the registry or revocation notice. Check that finalized manifest hashes are unchanged.
6. [ ] Run `dataset reconcile --dry-run` before reconciling any interrupted test export. Confirm incomplete builds stay unavailable. Confirm a complete recoverable build reports the proposed activation without changing the registry in dry-run mode.
