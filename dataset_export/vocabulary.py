"""Deterministic vocabulary snapshots keyed by permanent sign IDs."""

from dataset_export.models import SCHEMA_VERSION
from registry.database import RegistryError


def snapshot(signs) -> dict:
    classes = []
    for index, sign in enumerate(sorted({sign.id: sign for sign in signs}.values(), key=lambda item: item.id)):
        classes.append({
            "class_index": index, "sign_id": sign.id, "gloss": sign.gloss, "category": sign.category,
            "sign_type": sign.sign_type, "variant_label": sign.variant_label,
        })
    return {"language": "UGSL", "schema_version": SCHEMA_VERSION, "classes": classes}


def validate_vocabulary(payload: dict) -> dict[str, dict]:
    if not isinstance(payload, dict) or payload.get("language") != "UGSL" or payload.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError("Unsupported vocabulary schema or language.")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise RegistryError("The vocabulary requires a nonempty class list.")
    result = {}
    for index, entry in enumerate(classes):
        if (
            not isinstance(entry, dict) or type(entry.get("class_index")) is not int or entry["class_index"] != index
            or not isinstance(entry.get("sign_id"), str) or not entry["sign_id"] or entry["sign_id"] in result
            or not isinstance(entry.get("gloss"), str) or not entry["gloss"]
            or entry.get("sign_type") not in ("STATIC", "DYNAMIC", "UNKNOWN")
            or any(entry.get(field) is not None and not isinstance(entry[field], str) for field in ("category", "variant_label"))
        ):
            raise RegistryError("Invalid or duplicate vocabulary class.")
        result[entry["sign_id"]] = entry
    return result
