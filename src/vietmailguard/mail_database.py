"""SQLite persistence primitives for the local VietMailGuard Mail client.

Database timestamps created by this module use UTC RFC 3339 strings ending in
``Z``. Incoming message timestamps (``ngay_gui``) are preserved as supplied by
the importer and may be null when the source message has no reliable date.

All persistent SQLite identifiers deliberately follow the project policy:
Vietnamese without diacritics, ASCII-only, lowercase ``snake_case``. Python API
names remain English to match the rest of the package.
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "runtime" / "vietmailguard_mail.db"
CURRENT_SCHEMA_VERSION = 1

ALLOWED_FOLDERS = ("hop_thu_den", "thu_rac", "cach_ly", "da_xoa")
ALLOWED_PREDICTIONS = ("normal", "spam", "phishing")
ALLOWED_LANGUAGES = ("en", "vi", "mixed", "unknown")
ALLOWED_RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "phien_ban_co_so_du_lieu": ("phien_ban", "thoi_gian_ap_dung"),
    "thu": (
        "id_thu",
        "ma_thu_ngoai",
        "nguoi_gui",
        "nguoi_nhan",
        "cc",
        "tieu_de",
        "noi_dung",
        "noi_dung_html",
        "ngay_gui",
        "thu_muc",
        "da_doc",
        "da_gan_sao",
        "nguon",
        "ma_nguon",
        "thoi_gian_tao",
        "thoi_gian_cap_nhat",
    ),
    "phan_tich_thu": (
        "id_phan_tich",
        "id_thu",
        "phien_ban_mo_hinh",
        "ngon_ngu_phat_hien",
        "nhan_du_doan",
        "do_tin_cay",
        "diem_rui_ro",
        "muc_rui_ro",
        "hanh_dong_goc",
        "hanh_dong_de_xuat",
        "co_canh_bao",
        "so_phat_hien_bao_mat",
        "ket_qua_json",
        "thoi_gian_phan_tich",
    ),
    "phan_hoi_nguoi_dung": (
        "id_phan_hoi",
        "id_thu",
        "nhan_du_doan_ban_dau",
        "thu_muc_truoc",
        "thu_muc_sau",
        "hanh_dong_nguoi_dung",
        "ghi_chu",
        "thoi_gian_phan_hoi",
    ),
    "lich_su_thu_muc": (
        "id_lich_su",
        "id_thu",
        "thu_muc_cu",
        "thu_muc_moi",
        "ly_do",
        "thoi_gian_thay_doi",
    ),
}

REQUIRED_INDEXES = (
    "chi_muc_thu_thu_muc_ngay_gui",
    "chi_muc_thu_da_doc",
    "chi_muc_thu_da_gan_sao",
    "chi_muc_phan_tich_thu_id_thu",
    "chi_muc_phan_tich_thu_nhan_du_doan",
    "chi_muc_phan_tich_thu_diem_rui_ro",
)

_CREATE_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS phien_ban_co_so_du_lieu (
    phien_ban INTEGER PRIMARY KEY CHECK (phien_ban > 0),
    thoi_gian_ap_dung TEXT NOT NULL
)
"""

_VERSION_ONE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS thu (
        id_thu INTEGER PRIMARY KEY AUTOINCREMENT,
        ma_thu_ngoai TEXT,
        nguoi_gui TEXT NOT NULL DEFAULT '',
        nguoi_nhan TEXT NOT NULL DEFAULT '',
        cc TEXT NOT NULL DEFAULT '',
        tieu_de TEXT NOT NULL DEFAULT '',
        noi_dung TEXT NOT NULL DEFAULT '',
        noi_dung_html TEXT NOT NULL DEFAULT '',
        ngay_gui TEXT,
        thu_muc TEXT NOT NULL DEFAULT 'hop_thu_den'
            CHECK (thu_muc IN ('hop_thu_den', 'thu_rac', 'cach_ly', 'da_xoa')),
        da_doc INTEGER NOT NULL DEFAULT 0 CHECK (da_doc IN (0, 1)),
        da_gan_sao INTEGER NOT NULL DEFAULT 0 CHECK (da_gan_sao IN (0, 1)),
        nguon TEXT NOT NULL DEFAULT '',
        ma_nguon TEXT,
        thoi_gian_tao TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        thoi_gian_cap_nhat TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS phan_tich_thu (
        id_phan_tich INTEGER PRIMARY KEY AUTOINCREMENT,
        id_thu INTEGER NOT NULL,
        phien_ban_mo_hinh TEXT NOT NULL,
        ngon_ngu_phat_hien TEXT NOT NULL
            CHECK (ngon_ngu_phat_hien IN ('en', 'vi', 'mixed', 'unknown')),
        nhan_du_doan TEXT NOT NULL
            CHECK (nhan_du_doan IN ('normal', 'spam', 'phishing')),
        do_tin_cay REAL NOT NULL CHECK (do_tin_cay >= 0.0 AND do_tin_cay <= 1.0),
        diem_rui_ro REAL NOT NULL CHECK (diem_rui_ro >= 0.0 AND diem_rui_ro <= 100.0),
        muc_rui_ro TEXT NOT NULL
            CHECK (muc_rui_ro IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
        hanh_dong_goc TEXT NOT NULL,
        hanh_dong_de_xuat TEXT NOT NULL,
        co_canh_bao INTEGER NOT NULL DEFAULT 0 CHECK (co_canh_bao IN (0, 1)),
        so_phat_hien_bao_mat INTEGER NOT NULL DEFAULT 0
            CHECK (so_phat_hien_bao_mat >= 0),
        ket_qua_json TEXT NOT NULL CHECK (json_valid(ket_qua_json)),
        thoi_gian_phan_tich TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY (id_thu) REFERENCES thu(id_thu) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS phan_hoi_nguoi_dung (
        id_phan_hoi INTEGER PRIMARY KEY AUTOINCREMENT,
        id_thu INTEGER NOT NULL,
        nhan_du_doan_ban_dau TEXT NOT NULL
            CHECK (nhan_du_doan_ban_dau IN ('normal', 'spam', 'phishing')),
        thu_muc_truoc TEXT NOT NULL
            CHECK (thu_muc_truoc IN ('hop_thu_den', 'thu_rac', 'cach_ly', 'da_xoa')),
        thu_muc_sau TEXT NOT NULL
            CHECK (thu_muc_sau IN ('hop_thu_den', 'thu_rac', 'cach_ly', 'da_xoa')),
        hanh_dong_nguoi_dung TEXT NOT NULL,
        ghi_chu TEXT NOT NULL DEFAULT '',
        thoi_gian_phan_hoi TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY (id_thu) REFERENCES thu(id_thu) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lich_su_thu_muc (
        id_lich_su INTEGER PRIMARY KEY AUTOINCREMENT,
        id_thu INTEGER NOT NULL,
        thu_muc_cu TEXT NOT NULL
            CHECK (thu_muc_cu IN ('hop_thu_den', 'thu_rac', 'cach_ly', 'da_xoa')),
        thu_muc_moi TEXT NOT NULL
            CHECK (thu_muc_moi IN ('hop_thu_den', 'thu_rac', 'cach_ly', 'da_xoa')),
        ly_do TEXT NOT NULL,
        thoi_gian_thay_doi TEXT NOT NULL
            DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        FOREIGN KEY (id_thu) REFERENCES thu(id_thu) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS chi_muc_thu_thu_muc_ngay_gui ON thu(thu_muc, ngay_gui)",
    "CREATE INDEX IF NOT EXISTS chi_muc_thu_da_doc ON thu(da_doc)",
    "CREATE INDEX IF NOT EXISTS chi_muc_thu_da_gan_sao ON thu(da_gan_sao)",
    "CREATE INDEX IF NOT EXISTS chi_muc_phan_tich_thu_id_thu ON phan_tich_thu(id_thu)",
    """
    CREATE INDEX IF NOT EXISTS chi_muc_phan_tich_thu_nhan_du_doan
    ON phan_tich_thu(nhan_du_doan)
    """,
    """
    CREATE INDEX IF NOT EXISTS chi_muc_phan_tich_thu_diem_rui_ro
    ON phan_tich_thu(diem_rui_ro)
    """,
    "CREATE INDEX IF NOT EXISTS chi_muc_phan_hoi_id_thu ON phan_hoi_nguoi_dung(id_thu)",
    "CREATE INDEX IF NOT EXISTS chi_muc_lich_su_id_thu ON lich_su_thu_muc(id_thu)",
)

_savepoint_counter = itertools.count(1)


def utc_timestamp() -> str:
    """Return the timestamp convention used for migration records."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def open_connection(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> sqlite3.Connection:
    """Open a configured SQLite connection without implicitly changing schema."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        connection.close()
        raise RuntimeError("Khong the bat rang buoc khoa ngoai SQLite")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run an atomic transaction, using a savepoint when already nested."""

    nested = connection.in_transaction
    savepoint = f"giao_dich_{next(_savepoint_counter)}" if nested else ""
    connection.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
    try:
        yield connection
        connection.execute(f"RELEASE SAVEPOINT {savepoint}" if nested else "COMMIT")
    except BaseException:
        if nested:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        elif connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def initialize_schema(connection: sqlite3.Connection) -> int:
    """Apply known idempotent migrations and return the active schema version."""

    with transaction(connection):
        connection.execute(_CREATE_VERSION_TABLE)
        row = connection.execute(
            "SELECT MAX(phien_ban) FROM phien_ban_co_so_du_lieu"
        ).fetchone()
        existing_version = int(row[0] or 0)
        if existing_version > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "Phien ban co so du lieu moi hon phien ban ma ung dung ho tro: "
                f"{existing_version} > {CURRENT_SCHEMA_VERSION}"
            )

        # Version 1 statements are idempotent, which also makes initialization
        # repair missing schema objects without altering existing row data.
        for statement in _VERSION_ONE_STATEMENTS:
            connection.execute(statement)

        if existing_version < 1:
            connection.execute(
                """
                INSERT INTO phien_ban_co_so_du_lieu(phien_ban, thoi_gian_ap_dung)
                VALUES (?, ?)
                """,
                (1, utc_timestamp()),
            )

    return CURRENT_SCHEMA_VERSION


def open_initialized_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> sqlite3.Connection:
    """Open the runtime database and ensure its schema is initialized."""

    connection = open_connection(database_path)
    try:
        initialize_schema(connection)
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def database_connection(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> Iterator[sqlite3.Connection]:
    """Yield an initialized connection and always close it afterwards."""

    connection = open_initialized_database(database_path)
    try:
        yield connection
    finally:
        connection.close()


def health_check(connection: sqlite3.Connection) -> dict[str, Any]:
    """Return a non-mutating database/schema integrity summary."""

    errors: list[str] = []
    missing_tables: list[str] = []
    missing_columns: dict[str, list[str]] = {}
    schema_version: int | None = None
    quick_check = "error"
    foreign_key_violations = 0

    foreign_keys_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    if not foreign_keys_enabled:
        errors.append("foreign_keys_disabled")

    existing_tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    for table_name, expected_columns in TABLE_COLUMNS.items():
        if table_name not in existing_tables:
            missing_tables.append(table_name)
            continue
        actual_columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})")
        }
        absent = sorted(set(expected_columns) - actual_columns)
        if absent:
            missing_columns[table_name] = absent

    if "phien_ban_co_so_du_lieu" in existing_tables:
        row = connection.execute(
            "SELECT MAX(phien_ban) FROM phien_ban_co_so_du_lieu"
        ).fetchone()
        schema_version = int(row[0]) if row[0] is not None else None
    if schema_version != CURRENT_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")

    quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    if quick_check != "ok":
        errors.append("quick_check_failed")

    foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
    if foreign_key_violations:
        errors.append("foreign_key_violations")
    if missing_tables:
        errors.append("missing_tables")
    if missing_columns:
        errors.append("missing_columns")

    return {
        "healthy": not errors,
        "schema_version": schema_version,
        "foreign_keys_enabled": foreign_keys_enabled,
        "quick_check": quick_check,
        "foreign_key_violations": foreign_key_violations,
        "missing_tables": sorted(missing_tables),
        "missing_columns": missing_columns,
        "errors": errors,
    }
