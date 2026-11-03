"""Caption Cleaning and Store Building"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'"})       
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")                  


def clean_caption(text: str) -> str:
    """Light normalization for retrieval"""
    if not isinstance(text, str):
        raise TypeError(f"caption must be str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKD", text).translate(_QUOTES)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))  # cafe' -> cafe
    return " ".join(_TOKEN.findall(text.lower()))              # drops punctuation, remove extra spaces


def build_caption_store(lookup: dict, expected: int = 5):
    store, empty_clean = {}, []
    words_raw = words_clean = n_caps = bad_count = 0

    for image_id in sorted(lookup, key=int):                
        meta = lookup[image_id]
        raw = list(meta.get("captions", []))                   
        clean = [clean_caption(c) for c in raw]                

        for i, c in enumerate(clean):
            if not c:
                empty_clean.append({"image_id": int(image_id), "caption_index": i, "raw": raw[i]})

        store[str(image_id)] = {
            "file_name": meta.get("file_name"),
            "caption_raw": raw,
            "caption_clean": clean,
        }
        n_caps += len(raw)
        bad_count += len(raw) != expected
        words_raw += sum(len(c.split()) for c in raw)
        words_clean += sum(len(c.split()) for c in clean)

    report = {
        "images": len(store),
        "captions": n_caps,
        "images_with_unexpected_caption_count": bad_count,
        "empty_after_cleaning": empty_clean,
        "avg_words_raw": round(words_raw / max(n_caps, 1), 3),
        "avg_words_clean": round(words_clean / max(n_caps, 1), 3),
    }
    return store, report


def flatten_store(store: dict) -> list:
    """One row per caption. Row position in this list = vector position in the FAISS text index."""
    rows = []
    for image_id in sorted(store, key=int):
        entry = store[image_id]
        for i, (raw, clean) in enumerate(zip(entry["caption_raw"], entry["caption_clean"])):
            rows.append({
                "caption_id": f"{image_id}_{i}",
                "image_id": int(image_id),
                "caption_index": i,
                "caption_raw": raw,
                "caption_clean": clean,
            })
    return rows


def self_test():
    cases = {
        "A man riding a wave on top of a surfboard.\n": "a man riding a wave on top of a surfboard",
        "  Two   dogs -- playing, in the SNOW!! ": "two dogs playing in the snow",
        "A black-and-white photo of a woman\u2019s hat": "a black and white photo of a woman's hat",
        "Caf\u00e9 table with 2 cups.": "cafe table with 2 cups",
        "!!!": "",
    }
    for raw, expected in cases.items():
        got = clean_caption(raw)
        assert got == expected, (raw, got, expected)
        assert clean_caption(got) == got, "cleaning must be idempotent"

    lookup = {
        "2": {"file_name": "b.jpg", "captions": ["Hello, World!\n", "A  cat."]},
        "1": {"file_name": "a.jpg", "captions": ["Dogs.", "Two dogs!", "x", "y", "z"]},
    }
    store, report = build_caption_store(lookup)
    assert store["2"]["caption_raw"] == lookup["2"]["captions"], "raw must stay untouched"
    assert list(store) == ["1", "2"], "must be sorted by image id"
    rows = flatten_store(store)
    assert len(rows) == 7 and rows[0]["caption_id"] == "1_0"
    assert report["images_with_unexpected_caption_count"] == 1
    print("caption self-test PASSED")

def normalize_lookup(data) -> dict:
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        data = data["images"]
    if isinstance(data, list):
        out = {}
        for rec in data:
            key = str(rec["id"])
            assert key not in out, f"duplicate image id {key}"
            out[key] = rec
        return out
    return data

def main():
    p = argparse.ArgumentParser(description="Build raw/clean caption store.")
    p.add_argument("--input", type=Path, default=Path("data/coco_rag_data/unified_lookup.json"))
    p.add_argument("--output-dir", type=Path, default=Path("data/coco_rag_data"))
    p.add_argument("--name", default="kb", help="split name used in file names: kb, test, ...")
    p.add_argument("--forbid-ids", type=Path, default=None,
                   help="JSON list of IDs that must NOT be present (use test_ids.json for the KB)")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        return self_test()

    lookup = normalize_lookup(json.loads(a.input.read_text(encoding="utf-8")))

    if a.forbid_ids:                                           # anti-leakage guard
        forbidden = {str(i) for i in json.loads(a.forbid_ids.read_text(encoding="utf-8"))}
        overlap = forbidden & set(lookup)
        assert not overlap, f"LEAKAGE: {len(overlap)} forbidden IDs found in {a.input}"

    store, report = build_caption_store(lookup)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / f"captions_{a.name}.json").write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    with (a.output_dir / f"captions_{a.name}_flat.jsonl").open("w", encoding="utf-8") as f:
        for row in flatten_store(store):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (a.output_dir / f"captions_{a.name}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print({k: v for k, v in report.items() if k != "empty_after_cleaning"},
          "| empty after cleaning:", len(report["empty_after_cleaning"]))


if __name__ == "__main__":
    main()