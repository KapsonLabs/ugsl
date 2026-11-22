"""Statistics calculated from the frozen export selection."""

from collections import Counter
from statistics import mean


def _buckets(values) -> dict:
    counts = { "under_2000_ms": 0, "2000_to_4999_ms": 0, "5000_ms_or_more": 0 }
    for value in values:
        key = "under_2000_ms" if value < 2000 else "2000_to_4999_ms" if value < 5000 else "5000_ms_or_more"
        counts[key] += 1
    return counts


def overall(records) -> dict:
    return {
        "total_samples": len(records), "total_signs": len({record.sign_id for record in records}),
        "total_signers": len({record.signer_id for record in records}),
        "total_recording_sessions": len({record.session_id for record in records}),
        "total_duration_ms": sum(record.duration_ms for record in records),
        "total_video_bytes": sum(record.file_size_bytes for record in records),
    }


def summarize(records, vocabulary, selection) -> dict:
    by_sign, by_signer = {}, {}
    for row in records:
        by_sign.setdefault(row.sign_id, []).append(row)
        by_signer.setdefault(row.signer_id, []).append(row)
    classes = []
    for sign in vocabulary["classes"]:
        group = by_sign[sign["sign_id"]]
        durations = [row.duration_ms for row in group]
        classes.append({
            "sign_id": sign["sign_id"], "gloss": sign["gloss"], "sign_type": sign["sign_type"],
            "samples": len(group), "unique_signers": len({row.signer_id for row in group}),
            "recording_sessions": len({row.session_id for row in group}), "min_duration_ms": min(durations),
            "max_duration_ms": max(durations), "average_duration_ms": round(mean(durations), 3),
        })
    signer_rows = [
        {"signer_id": signer_id, "samples": len(group), "distinct_signs": len({row.sign_id for row in group}),
         "recording_sessions": len({row.session_id for row in group})}
        for signer_id, group in sorted(by_signer.items())
    ]
    class_sizes = [entry["samples"] for entry in classes]
    ratio = round(max(class_sizes) / min(class_sizes), 3)
    versions = Counter(str(row.annotation.version) for row in records)
    return {
        "overall": overall(records), "classes": classes, "signers": signer_rows,
        "quality": {
            "resolutions": dict(Counter(f"{row.width}x{row.height}" for row in records)),
            "frame_rates": dict(Counter("UNKNOWN" if row.average_fps is None else str(row.average_fps) for row in records)),
            "durations": _buckets(row.duration_ms for row in records),
            "technical_validation": dict(Counter(row.technical_validation_status for row in records)),
            "recording_sources": dict(Counter(row.source_type for row in records)),
        },
        "annotations": {
            "durations": _buckets(row.annotation.end_ms - row.annotation.start_ms for row in records),
            "corrected_classifications": sum(item.sample.sign_id != item.sign.id for item in selection.selected),
            "distinct_version_numbers": len(versions), "versions": dict(versions),
            "completeness": dict(Counter(row.annotation.completeness for row in records)),
        },
        "coverage": {
            "threshold_failures": selection.threshold_failures,
            "underrepresented_classes": [row["sign_id"] for row in selection.threshold_failures
                                         if row["samples"] < row["min_samples"] or not row["samples"]],
            "insufficient_unique_signers": [row["sign_id"] for row in selection.threshold_failures
                                           if row["unique_signers"] < row["min_signers"]],
            "overrepresented_signers": [row["signer_id"] for row in signer_rows if row["samples"] * 2 > len(records)],
            "largest_to_smallest_class_ratio": ratio, "severe_class_imbalance": ratio >= 5,
        },
    }
