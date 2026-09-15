from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from vietmailguard.mail_database import (
    CURRENT_SCHEMA_VERSION,
    REQUIRED_INDEXES,
    TABLE_COLUMNS,
    health_check,
    initialize_schema,
    open_connection,
    open_initialized_database,
    transaction,
)


def _insert_email(connection: sqlite3.Connection, *, folder: str = "hop_thu_den") -> int:
    cursor = connection.execute(
        """
        INSERT INTO thu(
            ma_thu_ngoai, nguoi_gui, nguoi_nhan, tieu_de, noi_dung,
            ngay_gui, thu_muc, nguon, ma_nguon
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "message-001@example.com",
            "alice@example.com",
            "bob@example.org",
            "Cap nhat du an",
            "Noi dung thu thu nghiem.",
            "2026-09-15T08:30:00Z",
            folder,
            "eml_import",
            "fixture-001",
        ),
    )
    return int(cursor.lastrowid)


def test_schema_creation_and_health_check(tmp_path: Path) -> None:
    database_path = tmp_path / "mail.db"
    connection = open_connection(database_path)
    try:
        assert initialize_schema(connection) == CURRENT_SCHEMA_VERSION
        assert initialize_schema(connection) == CURRENT_SCHEMA_VERSION
        result = health_check(connection)
    finally:
        connection.close()

    assert result["healthy"] is True
    assert result["foreign_keys_enabled"] is True
    assert result["quick_check"] == "ok"
    assert result["missing_tables"] == []
    assert result["missing_columns"] == {}


def test_foreign_key_enforcement(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO phan_tich_thu(
                    id_thu, phien_ban_mo_hinh, ngon_ngu_phat_hien,
                    nhan_du_doan, do_tin_cay, diem_rui_ro, muc_rui_ro,
                    hanh_dong_goc, hanh_dong_de_xuat, ket_qua_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (999, "v2", "vi", "normal", 0.9, 10, "LOW", "ALLOW", "ALLOW", "{}"),
            )
    finally:
        connection.close()


@pytest.mark.parametrize("folder", ["hop_thu_den", "thu_rac", "cach_ly", "da_xoa"])
def test_allowed_folder_values(tmp_path: Path, folder: str) -> None:
    connection = open_initialized_database(tmp_path / f"{folder}.db")
    try:
        email_id = _insert_email(connection, folder=folder)
        stored = connection.execute(
            "SELECT thu_muc FROM thu WHERE id_thu = ?", (email_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored == folder


def test_rejects_unknown_folder(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            _insert_email(connection, folder="khong_hop_le")
    finally:
        connection.close()


def test_insert_related_records(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        with transaction(connection):
            email_id = _insert_email(connection)
            connection.execute(
                """
                INSERT INTO phan_tich_thu(
                    id_thu, phien_ban_mo_hinh, ngon_ngu_phat_hien,
                    nhan_du_doan, do_tin_cay, diem_rui_ro, muc_rui_ro,
                    hanh_dong_goc, hanh_dong_de_xuat, co_canh_bao,
                    so_phat_hien_bao_mat, ket_qua_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email_id,
                    "vietmailguard-v2",
                    "mixed",
                    "normal",
                    0.82,
                    38,
                    "MEDIUM",
                    "ALLOW",
                    "REVIEW",
                    1,
                    2,
                    json.dumps({"prediction": "normal", "reasons": ["promotion"]}),
                ),
            )
            connection.execute(
                """
                INSERT INTO phan_hoi_nguoi_dung(
                    id_thu, nhan_du_doan_ban_dau, thu_muc_truoc, thu_muc_sau,
                    hanh_dong_nguoi_dung, ghi_chu
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    email_id,
                    "normal",
                    "hop_thu_den",
                    "cach_ly",
                    "chuyen_vao_cach_ly",
                    "Can kiem tra them",
                ),
            )
            connection.execute(
                """
                INSERT INTO lich_su_thu_muc(
                    id_thu, thu_muc_cu, thu_muc_moi, ly_do
                ) VALUES (?, ?, ?, ?)
                """,
                (email_id, "hop_thu_den", "cach_ly", "nguoi_dung_yeu_cau"),
            )

        assert connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM phan_tich_thu").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM phan_hoi_nguoi_dung").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM lich_su_thu_muc").fetchone()[0] == 1
    finally:
        connection.close()


def test_restart_persistence(tmp_path: Path) -> None:
    database_path = tmp_path / "persistent.db"
    first = open_initialized_database(database_path)
    email_id = _insert_email(first)
    first.close()

    second = open_initialized_database(database_path)
    try:
        row = second.execute(
            "SELECT tieu_de FROM thu WHERE id_thu = ?", (email_id,)
        ).fetchone()
        assert row is not None
        assert row[0] == "Cap nhat du an"
    finally:
        second.close()


def test_required_indexes_exist(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        index_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        connection.close()
    assert set(REQUIRED_INDEXES).issubset(index_names)


def test_schema_version_is_one(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        rows = connection.execute(
            "SELECT phien_ban FROM phien_ban_co_so_du_lieu ORDER BY phien_ban"
        ).fetchall()
    finally:
        connection.close()
    assert [row[0] for row in rows] == [1]


def test_transaction_rolls_back_on_failure(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        with pytest.raises(RuntimeError, match="rollback"):
            with transaction(connection):
                _insert_email(connection)
                raise RuntimeError("rollback")
        assert connection.execute("SELECT COUNT(*) FROM thu").fetchone()[0] == 0
    finally:
        connection.close()


def test_all_application_sql_identifiers_are_ascii_snake_case(tmp_path: Path) -> None:
    connection = open_initialized_database(tmp_path / "mail.db")
    try:
        schema_objects = connection.execute(
            """
            SELECT name, type FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' AND type IN ('table', 'index', 'trigger', 'view')
            """
        ).fetchall()
        identifiers = {str(row[0]) for row in schema_objects}
        table_names = {
            str(row[0]) for row in schema_objects if str(row[1]) == "table"
        }
        actual_columns: dict[str, tuple[str, ...]] = {}
        for table_name in TABLE_COLUMNS:
            actual_columns[table_name] = tuple(
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table_name})")
            )
            identifiers.update(actual_columns[table_name])
    finally:
        connection.close()

    ascii_snake_case = re.compile(r"^[a-z][a-z0-9_]*$", flags=re.ASCII)
    invalid = sorted(name for name in identifiers if not ascii_snake_case.fullmatch(name))
    assert invalid == []
    assert table_names == set(TABLE_COLUMNS)
    assert actual_columns == TABLE_COLUMNS
    assert all(name.startswith("chi_muc_") for name in identifiers if name in REQUIRED_INDEXES)
