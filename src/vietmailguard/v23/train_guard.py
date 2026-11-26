"""Bounded cross-boundary audit for a Train-only unlabeled expansion.

The existing verified group-aware Development benchmark is immutable. No new
independent Test is claimed. Sparse cross joins inspect full-text word trigrams,
without prefix truncation, approximate Jaccard, or dropped candidate buckets.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import sqlite3
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

from vietmailguard.v22.recovery import comparison_text
from .dataset import connect, digest


def features(text: str) -> set[int]:
    words = re.findall(r"\w+", comparison_text(text))
    if len(words) < 10:
        return set()
    return {((zlib.crc32(value.encode()) << 32) | zlib.crc32(value[::-1].encode()))
            for value in ("\x1f".join(words[i:i+3]) for i in range(len(words)-2))}


class ReferenceJoin:
    """The prefix-filter theorem applies to any fixed global token ordering."""
    def __init__(self, texts: list[str], threshold: float = .8125):
        self.threshold = threshold
        self.sets = [features(text) for text in texts]
        self.counts = Counter(token for items in self.sets for token in items)
        self.postings = defaultdict(list)
        for i, items in enumerate(self.sets):
            for token in self.prefix(items):
                self.postings[token].append(i)
        # Same bounded sparse-intersection algorithm used in the existing
        # V2.2 SSL audit; int32 counts prevent the uint8 >255 overlap overflow.
        self.vocabulary = {token: i for i, token in enumerate(self.counts)}
        indices, offsets = [], [0]
        for items in self.sets:
            indices.extend(sorted(self.vocabulary[token] for token in items))
            offsets.append(len(indices))
        matrix = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, offsets),
                            shape=(len(texts), len(self.vocabulary)))
        self.transpose = matrix.T.tocsr()
        self.sizes = np.array([len(items) for items in self.sets], dtype=np.int64)

    def matches_many(self, texts):
        indices, offsets, sizes = [], [0], []
        for text in texts:
            items = features(text)
            sizes.append(len(items))
            indices.extend(sorted(self.vocabulary[token] for token in items if token in self.vocabulary))
            offsets.append(len(indices))
        matrix = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, offsets),
                            shape=(len(texts), len(self.vocabulary)))
        found = (matrix @ self.transpose).tocoo()
        union = np.asarray(sizes, dtype=np.int64)[found.row] + self.sizes[found.col] - found.data
        valid = (union > 0) & (found.data >= self.threshold * union)
        return {int(i) for i in found.row[valid]}

    def prefix(self, items):
        return sorted(items, key=lambda token: (self.counts[token], token))[
            :len(items)-math.ceil(self.threshold*len(items))+1]

    def matches(self, text: str):
        items = features(text)
        if not items:
            return []
        candidates = {i for token in self.prefix(items) for i in self.postings.get(token, ())}
        result = []
        for i in candidates:
            right = self.sets[i]
            if not self.threshold*len(items) <= len(right) <= len(items)/self.threshold:
                continue
            score = len(items & right) / len(items | right)
            if score >= self.threshold:
                result.append((i, score))
        return result


def reference_rows(root: Path) -> dict:
    csv.field_size_limit(100_000_000)
    directory = root / "data/v22/group_aware_supervised_pilot_01/cohort"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.csv"
        if digest(path) != manifest["output_sha256"][split]:
            raise ValueError("Locked Development reference SHA-256 differs")
        # Hash all existing source files; the stored manifest schema is preserved.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            result[split] = list(csv.DictReader(stream))
    if [len(result[k]) for k in ("train", "validation")] != [7748, 1938]:
        raise ValueError("Locked Development benchmark differs")
    result["source_hashes"] = {s: digest(directory / f"{s}.csv") for s in ("train", "validation")}
    result["manifest_sha256"] = digest(directory / "manifest.json")
    return result


def local_tokenizer(root: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    config = json.loads((root / "config/v21/runtime_multilingual.json").read_text(encoding="utf-8"))
    snapshot = Path(snapshot_download(config["encoder"]["model_id"],
                                     revision=config["encoder"]["revision"], local_files_only=True))
    path = snapshot / "tokenizer.json"
    if digest(path) != config["encoder_files"]["tokenizer.json"]:
        raise ValueError("Local tokenizer hash differs")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.no_padding()
    return tokenizer


def token_hash(tokenizer, text):
    values = tokenizer.encode(text).ids
    return hashlib.sha256(struct.pack("<" + "I"*len(values), *values)).digest()


def legacy_lineage_guard(connection, root):
    """Use loader's saved all-V2-source linkage, never old Test membership."""
    if connection.execute("SELECT 1 FROM cau_hinh WHERE ma='original_enron_lineage_guard_complete'").fetchone():
        return
    csv.field_size_limit(100_000_000)
    for path in sorted((root / "data/v21/processed/enron_original_v1").glob("part-*.csv")):
        # Intake has already locked these part checksums. No text is emitted.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            ids = [("enron_original:" + row["id"],) for row in csv.DictReader(stream)
                   if row.get("v2_enron_csv_row", "").strip()]
        connection.executemany("""INSERT OR IGNORE INTO cach_ly
            SELECT id_thu,'legacy_original_enron_source_lineage' FROM ban_ghi
            WHERE id_nguon=? AND id_thu IS NOT NULL""", ids)
        connection.commit()
    connection.execute("INSERT INTO cau_hinh VALUES ('original_enron_lineage_guard_complete','1')")
    connection.commit()


def propagate_known_constraints(connection, root):
    """Do not split legacy near/provenance components or exact template families.

    These are conservative split constraints, not duplicate-identity/label claims.
    Raw records remain stored; none is silently deleted or relabeled.
    """
    connection.execute("ATTACH DATABASE ? AS cu", ((root / "data/v22/phase3/expansion.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1",))
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS nhom_ke_thua(id_thu INTEGER PRIMARY KEY,id_nhom TEXT);
        CREATE INDEX IF NOT EXISTS chi_muc_nhom_ke_thua ON nhom_ke_thua(id_nhom);
        INSERT OR IGNORE INTO nhom_ke_thua
        SELECT t.id_thu,n.id_nhom FROM thu t JOIN cu.van_ban v ON v.ma_bam=t.ma_bam_chuan
        JOIN cu.nhom n ON n.id_van_ban=v.id_van_ban;
    """)
    previous = -1
    while True:
        current = connection.execute("SELECT COUNT(*) FROM cach_ly").fetchone()[0]
        if current == previous:
            break
        previous = current
        connection.execute("""INSERT OR IGNORE INTO cach_ly
            SELECT n.id_thu,'legacy_group_constraint' FROM nhom_ke_thua n
            WHERE n.id_nhom IN (SELECT z.id_nhom FROM nhom_ke_thua z JOIN cach_ly q ON q.id_thu=z.id_thu)""")
        connection.execute("""INSERT OR IGNORE INTO cach_ly
            SELECT t.id_thu,'exact_template_split_constraint' FROM thu t
            WHERE t.ma_bam_mau!='' AND t.ma_bam_mau IN
            (SELECT z.ma_bam_mau FROM thu z JOIN cach_ly q ON q.id_thu=z.id_thu WHERE z.ma_bam_mau!='')""")
        connection.commit()
    connection.execute("DETACH DATABASE cu")


def audit_expansion(root: Path) -> dict:
    reference = reference_rows(root)
    validation = reference["validation"]
    bodies = [row["body"] for row in validation]
    full = [row["subject"] + "\n" + row["body"] for row in validation]
    exact_body = {comparison_text(text) for text in bodies}
    joins = [ReferenceJoin(bodies), ReferenceJoin(full)]
    tokenizer = local_tokenizer(root)
    validation_tokens = {token_hash(tokenizer, row["subject"][:4096] + "\n" + row["body"][:4096]) for row in validation}
    with connect(root / "data/v23/intake.sqlite3") as connection:
        lock = json.dumps({"reference": reference["source_hashes"], "word_trigram_jaccard": .8125,
                           "full_text": True, "tokenizer_max_tokens": 128}, sort_keys=True)
        previous = connection.execute("SELECT gia_tri FROM cau_hinh WHERE ma='guard_protocol'").fetchone()
        if previous and previous[0] != lock:
            raise ValueError("Development guard protocol changed")
        connection.execute("INSERT OR IGNORE INTO cau_hinh VALUES ('guard_protocol',?)", (lock,))
        last = int((connection.execute("SELECT gia_tri FROM cau_hinh WHERE ma='guard_last'").fetchone() or ['0'])[0])
        cursor = connection.execute("""SELECT t.id_thu,t.van_ban_nen FROM thu t WHERE t.id_thu>?
            AND NOT EXISTS(SELECT 1 FROM tham_chieu r WHERE r.id_thu=t.id_thu)
            AND NOT EXISTS(SELECT 1 FROM cach_ly q WHERE q.id_thu=t.id_thu) ORDER BY t.id_thu""", (last,))
        processed = 0
        while batch := cursor.fetchmany(256):
            values = [json.loads(zlib.decompress(blob)) for _, blob in batch]
            body_matches = joins[0].matches_many([v["body"] for v in values])
            full_matches = joins[1].matches_many([v["subject"] + "\n" + v["body"] for v in values])
            encoded = tokenizer.encode_batch([v["subject"][:4096] + "\n" + v["body"][:4096] for v in values])
            for i, ((mail_id, blob), value, tokenized) in enumerate(zip(batch, values, encoded)):
                body = value["body"]
                reason = ""
                if comparison_text(body) in exact_body:
                    reason = "development_validation_body_or_template_equal"
                elif hashlib.sha256(struct.pack("<" + "I"*len(tokenized.ids), *tokenized.ids)).digest() in validation_tokens:
                    reason = "development_validation_tokenizer_input_equal"
                elif i in body_matches or i in full_matches:
                    reason = "development_validation_full_word_trigram_jaccard"
                if reason:
                    connection.execute("INSERT OR IGNORE INTO cach_ly VALUES (?,?)", (mail_id, reason))
                last = mail_id
                processed += 1
            connection.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('guard_last',?)", (str(last),))
            connection.commit()
            if processed % 5120 == 0:
                print(json.dumps({"guard_rows_checked": processed, "last_id": last}), flush=True)
        legacy_lineage_guard(connection, root)
        propagate_known_constraints(connection, root)
        result = {
            "verified_train_rows": len(reference["train"]), "verified_validation_rows": len(validation),
            "remaining_expansion_rows": connection.execute("""SELECT COUNT(*) FROM thu t
                WHERE NOT EXISTS(SELECT 1 FROM tham_chieu r WHERE r.id_thu=t.id_thu)
                AND NOT EXISTS(SELECT 1 FROM cach_ly q WHERE q.id_thu=t.id_thu)""").fetchone()[0],
            "quarantine_reasons": list(connection.execute("SELECT ly_do,COUNT(*) FROM cach_ly GROUP BY ly_do")),
            "reference_hashes": reference["source_hashes"], "final_test_ready": False,
            "direct_cross_boundary_checks": "full body/full content word-trigram Jaccard, exact/template body and actual tokenizer inputs",
            "limitations": ["No claim of exhaustive semantic or cross-language relatedness.",
                            "Expansion is Train-only; no weak/unlabeled evaluation references.",
                            "Internal Train-related messages are retained, not asserted to be independent duplicate groups."]}
        connection.execute("INSERT OR REPLACE INTO cau_hinh VALUES ('development_guard_complete','1')")
        connection.commit()
        (root / "data/v23/development_guard.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return result
