"""Fetch frozen public inputs that are intentionally not duplicated in this repository."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_COMMIT = "6015d82feea1b3d593dfa329f2f1cb7a16349436"
MERGED_AUTM_URL = (
    "https://raw.githubusercontent.com/krish533/policy-communication-tech-transfer/"
    f"{SOURCE_COMMIT}/data/merged_autm.csv"
)
MERGED_AUTM_SHA256 = "070f10179b34a0730d1f09ea3322d21e8f32e20d3036760df2adfef182aa2d4b"
# Paper 1 sentence-level text and scores, used only for the Table 5 keyword comparison.
P1_COMMIT = "25a9472b34334825b6d6c6a334f5b88eb00695b5"
P1_POLICY_URL = (
    "https://raw.githubusercontent.com/krish533/Tech-transfer-1/"
    f"{P1_COMMIT}/P1_replication_package/data/derived/policy_level_indices_institution_year.csv"
)
P1_POLICY_SHA256 = "694c21acc07d2a50ed27199d0e7ec01bb6974f08f843cbce2d7da4318f864198"
P1_SENTENCES_URL = (
    "https://raw.githubusercontent.com/krish533/Tech-transfer-1/"
    f"{P1_COMMIT}/P1_replication_package/data/derived/sentence_scores_canonical.csv"
)
P1_SENTENCES_SHA256 = "81b54dbb98c51c5c85ff5ffd449d1c122250b4b06c886ac1b07142bede7f7e49"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256(dest) == expected_sha256:
        print(f"Verified existing {dest.relative_to(ROOT)}")
        return
    tmp = dest.with_suffix(dest.suffix + ".download")
    urllib.request.urlretrieve(url, tmp)
    actual = sha256(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
        )
    tmp.replace(dest)
    print(f"Fetched and verified {dest.relative_to(ROOT)}")


def main() -> None:
    fetch(MERGED_AUTM_URL, DATA / "merged_autm.csv", MERGED_AUTM_SHA256)
    fetch(
        P1_POLICY_URL,
        DATA / "p1_policy_level_indices_institution_year.csv",
        P1_POLICY_SHA256,
    )
    fetch(P1_SENTENCES_URL, DATA / "p1_sentence_scores_canonical.csv", P1_SENTENCES_SHA256)


if __name__ == "__main__":
    main()
