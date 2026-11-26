"""Validate MediaPipe geometry and assign anatomical hand slots."""

from math import isfinite

import numpy as np

from landmarks.schemas import HAND_SLOTS, HandFrame


def _landmarks(points, kind: str, detection: dict, frame: HandFrame) -> np.ndarray | None:
    if points is None or (kind == "world" and len(points) == 0):
        return None
    count = len(points)
    if count != 21:
        frame.warnings.append({
            "code": "UNEXPECTED_LANDMARK_COUNT", "detection_index": detection["index"],
            "landmarks": kind, "expected": 21, "actual": count,
        })
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray([(p.x, p.y, p.z) for p in points], dtype=np.float32)
        valid = values.shape == (count, 3) and np.isfinite(values).all() if count else True
    except (AttributeError, OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        frame.invalid_landmark_values = True
        frame.warnings.append({
            "code": "INVALID_LANDMARK_VALUES", "detection_index": detection["index"], "landmarks": kind,
        })
    return values if valid and count == 21 else None


def _handedness(categories) -> tuple[list[dict], str | None, float | None]:
    predictions = []
    scores = {}
    for category in categories:
        label = getattr(category, "category_name", None) or getattr(category, "display_name", None)
        if not isinstance(label, str):
            label = None
        try:
            score = float(category.score)
        except (AttributeError, TypeError, ValueError, OverflowError):
            score = None
        if score is not None and not isfinite(score):
            score = None
        predictions.append({"label": label, "score": score})
        normalized = label.strip().upper() if label else None
        if normalized in HAND_SLOTS and score is not None and 0 <= score <= 1:
            scores[normalized] = max(score, scores.get(normalized, -1))
    if scores:
        best = max(scores.values())
        labels = [label for label, score in scores.items() if score == best]
        if len(labels) == 1:
            return predictions, labels[0], best
    return predictions, None, None


def parse_hand_result(raw, frame_index: int, timestamp_ms: int, *, mirrored_input: bool = False) -> HandFrame:
    frame = HandFrame(frame_index, timestamp_ms)
    images = getattr(raw, "hand_landmarks", None)
    worlds = getattr(raw, "hand_world_landmarks", None) or []
    handedness = getattr(raw, "handedness", None)
    if images is None or handedness is None:
        frame.warnings.append({"code": "MISSING_RESULT_COMPONENT", "image": images is None, "handedness": handedness is None})
    images, handedness = images or [], handedness or []
    frame.raw_image_detections = len(images)
    count = max(len(images), len(worlds), len(handedness))
    can_pair = len(images) == len(handedness) and count <= 2
    can_pair_world = len(worlds) == len(images)
    if not can_pair or (worlds and not can_pair_world):
        frame.ambiguous_handedness = not can_pair
        frame.warnings.append({
            "code": "UNEXPECTED_DETECTION_COUNTS", "image": len(images),
            "world": len(worlds), "handedness": len(handedness), "maximum_hands": 2,
        })

    candidates = []
    for index in range(count):
        categories = handedness[index] if index < len(handedness) else None
        predictions, label, score = _handedness(categories or [])
        slot = HAND_SLOTS.index(label) if label is not None else None
        if slot is not None and mirrored_input:
            slot = 1 - slot
        detection = {
            "index": index, "predictions": predictions,
            "proposed_slot": HAND_SLOTS[slot] if slot is not None else None, "assigned_slot": None,
        }
        image = _landmarks(images[index] if index < len(images) else None, "image", detection, frame)
        world = _landmarks(worlds[index] if index < len(worlds) else None, "world", detection, frame)
        if slot is None and index < len(images):
            frame.ambiguous_handedness = True
            frame.warnings.append({"code": "AMBIGUOUS_HANDEDNESS", "detection_index": index})
        frame.detections.append(detection)
        candidates.append((detection, image, world, slot, score))

    for detection, image, world, slot, score in candidates:
        duplicate_slot = slot is not None and sum(row[3] == slot for row in candidates) > 1
        if duplicate_slot:
            frame.ambiguous_handedness = True
            frame.warnings.append({
                "code": "DUPLICATE_HANDEDNESS", "slot": HAND_SLOTS[slot], "detection_index": detection["index"],
            })
        if can_pair and not duplicate_slot and slot is not None and image is not None:
            frame.image_landmarks[slot] = image
            frame.presence_mask[slot] = True
            frame.handedness_score[slot] = score
            detection["assigned_slot"] = HAND_SLOTS[slot]
            if can_pair_world and world is not None:
                frame.world_landmarks[slot] = world
        elif image is not None:
            detection["unassigned_image_landmarks"] = image.tolist()
        if world is not None and (detection["assigned_slot"] is None or not can_pair_world):
            detection["unassigned_world_landmarks"] = world.tolist()
    return frame
