"""Parameterized SQLite repository for VietMailGuard Mail."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from vietmailguard.mail_database import (
    ALLOWED_FOLDERS,
    ALLOWED_LANGUAGES,
    ALLOWED_PREDICTIONS,
    ALLOWED_RISK_LEVELS,
    transaction,
    utc_timestamp,
)


class MailNotFoundError(LookupError):
    """Raised when a requested mailbox email does not exist."""


def _row_to_email(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id_thu"]),
        "external_id": row["ma_thu_ngoai"],
        "sender": str(row["nguoi_gui"]),
        "receiver": str(row["nguoi_nhan"]),
        "cc": str(row["cc"]),
        "subject": str(row["tieu_de"]),
        "body": str(row["noi_dung"]),
        "body_html": str(row["noi_dung_html"]),
        "sent_at": row["ngay_gui"],
        "folder": str(row["thu_muc"]),
        "is_read": bool(row["da_doc"]),
        "is_starred": bool(row["da_gan_sao"]),
        "source": str(row["nguon"]),
        "source_id": row["ma_nguon"],
        "created_at": str(row["thoi_gian_tao"]),
        "updated_at": str(row["thoi_gian_cap_nhat"]),
    }


def _row_to_analysis(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id_phan_tich"]),
        "email_id": int(row["id_thu"]),
        "model_version": str(row["phien_ban_mo_hinh"]),
        "detected_language": str(row["ngon_ngu_phat_hien"]),
        "prediction": str(row["nhan_du_doan"]),
        "confidence": float(row["do_tin_cay"]),
        "risk_score": float(row["diem_rui_ro"]),
        "risk_level": str(row["muc_rui_ro"]),
        "base_recommended_action": str(row["hanh_dong_goc"]),
        "recommended_action": str(row["hanh_dong_de_xuat"]),
        "has_warning": bool(row["co_canh_bao"]),
        "security_finding_count": int(row["so_phat_hien_bao_mat"]),
        "result": json.loads(str(row["ket_qua_json"])),
        "analyzed_at": str(row["thoi_gian_phan_tich"]),
    }


class MailRepository:
    """Keep all mailbox SQL behind a small persistence API."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """Group multiple repository operations in one safe transaction."""

        with transaction(self.connection):
            yield

    def create_email(
        self,
        *,
        sender: object = "",
        receiver: object = "",
        cc: object = "",
        subject: object = "",
        body: object = "",
        body_html: object = "",
        sent_at: str | None = None,
        folder: str = "hop_thu_den",
        is_read: bool = False,
        is_starred: bool = False,
        source: object = "",
        source_id: str | None = None,
        external_id: str | None = None,
    ) -> int:
        """Insert one email and return its database identifier."""

        self._validate_folder(folder)
        cursor = self.connection.execute(
            """
            INSERT INTO thu(
                ma_thu_ngoai, nguoi_gui, nguoi_nhan, cc, tieu_de, noi_dung,
                noi_dung_html, ngay_gui, thu_muc, da_doc, da_gan_sao,
                nguon, ma_nguon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                external_id,
                str(sender or ""),
                str(receiver or ""),
                str(cc or ""),
                str(subject or ""),
                str(body or ""),
                str(body_html or ""),
                sent_at,
                folder,
                int(bool(is_read)),
                int(bool(is_starred)),
                str(source or ""),
                source_id,
            ),
        )
        return int(cursor.lastrowid)

    def get_email(self, email_id: int) -> dict[str, Any] | None:
        """Return one email or ``None`` when it does not exist."""

        row = self.connection.execute(
            "SELECT * FROM thu WHERE id_thu = ?", (email_id,)
        ).fetchone()
        return None if row is None else _row_to_email(row)

    def get_email_by_external_id(self, external_id: str) -> dict[str, Any] | None:
        """Return the first email carrying a deterministic external identifier."""

        row = self.connection.execute(
            """
            SELECT * FROM thu
            WHERE ma_thu_ngoai = ?
            ORDER BY id_thu
            LIMIT 1
            """,
            (external_id,),
        ).fetchone()
        return None if row is None else _row_to_email(row)

    def list_emails(
        self,
        *,
        folder: str | None = None,
        unread: bool | None = None,
        starred: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List emails using optional mailbox-state filters."""

        clauses: list[str] = []
        parameters: list[object] = []
        if folder is not None:
            self._validate_folder(folder)
            clauses.append("thu_muc = ?")
            parameters.append(folder)
        if unread is not None:
            clauses.append("da_doc = ?")
            parameters.append(int(not unread))
        if starred is not None:
            clauses.append("da_gan_sao = ?")
            parameters.append(int(starred))
        where_clause = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.extend(self._pagination(limit, offset))
        rows = self.connection.execute(
            "SELECT * FROM thu"
            + where_clause
            + " ORDER BY COALESCE(ngay_gui, thoi_gian_tao) DESC, id_thu DESC LIMIT ? OFFSET ?",
            parameters,
        ).fetchall()
        return [_row_to_email(row) for row in rows]

    def mailbox_counts(self) -> dict[str, Any]:
        """Return folder, unread, starred, and warning counts for navigation."""

        folder_counts = {folder: 0 for folder in ALLOWED_FOLDERS}
        unread_counts = {folder: 0 for folder in ALLOWED_FOLDERS}
        for row in self.connection.execute(
            """
            SELECT thu_muc, COUNT(*) AS tong_so,
                   SUM(CASE WHEN da_doc = 0 THEN 1 ELSE 0 END) AS chua_doc
            FROM thu
            GROUP BY thu_muc
            """
        ).fetchall():
            folder = str(row["thu_muc"])
            folder_counts[folder] = int(row["tong_so"])
            unread_counts[folder] = int(row["chua_doc"] or 0)

        starred = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM thu WHERE da_gan_sao = 1"
            ).fetchone()[0]
        )
        warnings = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM phan_tich_thu AS a
                WHERE a.id_phan_tich = (
                    SELECT p.id_phan_tich FROM phan_tich_thu AS p
                    WHERE p.id_thu = a.id_thu
                    ORDER BY p.thoi_gian_phan_tich DESC, p.id_phan_tich DESC
                    LIMIT 1
                )
                AND a.co_canh_bao = 1
                """
            ).fetchone()[0]
        )
        security_counts = self.connection.execute(
            """
            SELECT COUNT(*) AS tong_so,
                   SUM(CASE WHEN a.muc_rui_ro IN ('HIGH', 'CRITICAL')
                            THEN 1 ELSE 0 END) AS cao_nghiem_trong
            FROM phan_tich_thu AS a
            WHERE a.id_phan_tich = (
                SELECT p.id_phan_tich FROM phan_tich_thu AS p
                WHERE p.id_thu = a.id_thu
                ORDER BY p.thoi_gian_phan_tich DESC, p.id_phan_tich DESC
                LIMIT 1
            )
            """
        ).fetchone()
        return {
            "folders": folder_counts,
            "unread": unread_counts,
            "starred": starred,
            "warnings": warnings,
            "security_total": int(security_counts["tong_so"]),
            "high_critical": int(security_counts["cao_nghiem_trong"] or 0),
            "total": sum(folder_counts.values()),
        }

    def list_security_emails(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List stored analyses by risk without creating new security metrics."""

        parameters = self._pagination(limit, offset)
        rows = self.connection.execute(
            """
            SELECT t.*, a.id_phan_tich
            FROM thu AS t
            JOIN phan_tich_thu AS a ON a.id_phan_tich = (
                SELECT p.id_phan_tich FROM phan_tich_thu AS p
                WHERE p.id_thu = t.id_thu
                ORDER BY p.thoi_gian_phan_tich DESC, p.id_phan_tich DESC
                LIMIT 1
            )
            WHERE a.co_canh_bao = 1 OR a.muc_rui_ro IN ('HIGH', 'CRITICAL')
            ORDER BY a.diem_rui_ro DESC, a.id_phan_tich DESC
            LIMIT ? OFFSET ?
            """,
            parameters,
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            email = _row_to_email(row)
            analysis_row = self.connection.execute(
                "SELECT * FROM phan_tich_thu WHERE id_phan_tich = ?",
                (int(row["id_phan_tich"]),),
            ).fetchone()
            output.append({"email": email, "analysis": _row_to_analysis(analysis_row)})
        return output

    def update_read_status(self, email_id: int, is_read: bool) -> dict[str, Any]:
        """Set the read flag and return the updated email."""

        self._update_boolean(email_id, "da_doc", is_read)
        return self._require_email(email_id)

    def update_starred_status(self, email_id: int, is_starred: bool) -> dict[str, Any]:
        """Set the starred flag and return the updated email."""

        self._update_boolean(email_id, "da_gan_sao", is_starred)
        return self._require_email(email_id)

    def move_email(self, email_id: int, folder: str, *, reason: str) -> dict[str, Any]:
        """Move one email and atomically append folder history."""

        self._validate_folder(folder)
        if not reason.strip():
            raise ValueError("Folder move reason must not be empty")
        with transaction(self.connection):
            email = self._require_email(email_id)
            previous_folder = str(email["folder"])
            moved = previous_folder != folder
            if moved:
                self.connection.execute(
                    """
                    UPDATE thu
                    SET thu_muc = ?, thoi_gian_cap_nhat = ?
                    WHERE id_thu = ?
                    """,
                    (folder, utc_timestamp(), email_id),
                )
                self.connection.execute(
                    """
                    INSERT INTO lich_su_thu_muc(
                        id_thu, thu_muc_cu, thu_muc_moi, ly_do
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (email_id, previous_folder, folder, reason),
                )
        return {
            "email": self._require_email(email_id),
            "previous_folder": previous_folder,
            "new_folder": folder,
            "moved": moved,
            "reason": reason,
        }

    def save_analysis(
        self,
        email_id: int,
        *,
        model_version: str,
        detected_language: str,
        prediction: str,
        confidence: float,
        risk_score: float,
        risk_level: str,
        base_recommended_action: str,
        recommended_action: str,
        has_warning: bool,
        security_finding_count: int,
        result: Mapping[str, Any],
    ) -> int:
        """Append one immutable historical inference result."""

        self._require_email(email_id)
        if detected_language not in ALLOWED_LANGUAGES:
            raise ValueError(f"Unsupported detected language: {detected_language!r}")
        if prediction not in ALLOWED_PREDICTIONS:
            raise ValueError(f"Unsupported ML prediction: {prediction!r}")
        if risk_level not in ALLOWED_RISK_LEVELS:
            raise ValueError(f"Unsupported risk level: {risk_level!r}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not 0.0 <= float(risk_score) <= 100.0:
            raise ValueError("Risk score must be between 0 and 100")
        if security_finding_count < 0:
            raise ValueError("Security finding count must not be negative")
        serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
        cursor = self.connection.execute(
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
                model_version,
                detected_language,
                prediction,
                float(confidence),
                float(risk_score),
                risk_level,
                base_recommended_action,
                recommended_action,
                int(has_warning),
                int(security_finding_count),
                serialized,
            ),
        )
        return int(cursor.lastrowid)

    def get_latest_analysis(self, email_id: int) -> dict[str, Any] | None:
        """Return the newest stored analysis without recomputing it."""

        row = self.connection.execute(
            """
            SELECT * FROM phan_tich_thu
            WHERE id_thu = ?
            ORDER BY thoi_gian_phan_tich DESC, id_phan_tich DESC
            LIMIT 1
            """,
            (email_id,),
        ).fetchone()
        return None if row is None else _row_to_analysis(row)

    def save_user_feedback(
        self,
        email_id: int,
        *,
        initial_prediction: str,
        previous_folder: str,
        new_folder: str,
        action: str,
        note: object = "",
    ) -> int:
        """Persist feedback without changing any analysis record."""

        self._require_email(email_id)
        if initial_prediction not in ALLOWED_PREDICTIONS:
            raise ValueError(f"Unsupported ML prediction: {initial_prediction!r}")
        self._validate_folder(previous_folder)
        self._validate_folder(new_folder)
        if not action.strip():
            raise ValueError("Feedback action must not be empty")
        cursor = self.connection.execute(
            """
            INSERT INTO phan_hoi_nguoi_dung(
                id_thu, nhan_du_doan_ban_dau, thu_muc_truoc, thu_muc_sau,
                hanh_dong_nguoi_dung, ghi_chu
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                email_id,
                initial_prediction,
                previous_folder,
                new_folder,
                action,
                str(note or ""),
            ),
        )
        return int(cursor.lastrowid)

    def list_folder_history(self, email_id: int) -> list[dict[str, Any]]:
        """Return folder moves from oldest to newest."""

        self._require_email(email_id)
        rows = self.connection.execute(
            """
            SELECT * FROM lich_su_thu_muc
            WHERE id_thu = ?
            ORDER BY thoi_gian_thay_doi, id_lich_su
            """,
            (email_id,),
        ).fetchall()
        return [
            {
                "id": int(row["id_lich_su"]),
                "email_id": int(row["id_thu"]),
                "previous_folder": str(row["thu_muc_cu"]),
                "new_folder": str(row["thu_muc_moi"]),
                "reason": str(row["ly_do"]),
                "changed_at": str(row["thoi_gian_thay_doi"]),
            }
            for row in rows
        ]

    def search(
        self,
        query: str,
        *,
        folder: str | None = None,
        unread: bool | None = None,
        starred: bool | None = None,
        prediction: str | None = None,
        risk_level: str | None = None,
        language: str | None = None,
        require_analysis: bool = False,
        order_by_risk: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search sender/subject/body with optional safe, parameterized filters."""

        clauses: list[str] = []
        parameters: list[object] = []
        if query:
            escaped_query = (
                query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = f"%{escaped_query}%"
            clauses.append(
                "(t.nguoi_gui LIKE ? ESCAPE '\\' OR t.tieu_de LIKE ? ESCAPE '\\' "
                "OR t.noi_dung LIKE ? ESCAPE '\\')"
            )
            parameters.extend((pattern, pattern, pattern))
        if folder is not None:
            self._validate_folder(folder)
            clauses.append("t.thu_muc = ?")
            parameters.append(folder)
        if unread is not None:
            clauses.append("t.da_doc = ?")
            parameters.append(int(not unread))
        if starred is not None:
            clauses.append("t.da_gan_sao = ?")
            parameters.append(int(starred))
        if prediction is not None:
            if prediction not in ALLOWED_PREDICTIONS:
                raise ValueError(f"Unsupported ML prediction: {prediction!r}")
            clauses.append("a.nhan_du_doan = ?")
            parameters.append(prediction)
        if risk_level is not None:
            if risk_level not in ALLOWED_RISK_LEVELS:
                raise ValueError(f"Unsupported risk level: {risk_level!r}")
            clauses.append("a.muc_rui_ro = ?")
            parameters.append(risk_level)
        if language is not None:
            if language not in ALLOWED_LANGUAGES:
                raise ValueError(f"Unsupported detected language: {language!r}")
            clauses.append("a.ngon_ngu_phat_hien = ?")
            parameters.append(language)
        if require_analysis:
            clauses.append("a.id_phan_tich IS NOT NULL")
        where_clause = " AND ".join(clauses) if clauses else "1 = 1"
        order_clause = (
            "a.diem_rui_ro DESC, a.id_phan_tich DESC"
            if order_by_risk
            else "COALESCE(t.ngay_gui, t.thoi_gian_tao) DESC, t.id_thu DESC"
        )
        parameters.extend(self._pagination(limit, offset))
        rows = self.connection.execute(
            """
            SELECT t.* FROM thu AS t
            LEFT JOIN phan_tich_thu AS a ON a.id_phan_tich = (
                SELECT p.id_phan_tich FROM phan_tich_thu AS p
                WHERE p.id_thu = t.id_thu
                ORDER BY p.thoi_gian_phan_tich DESC, p.id_phan_tich DESC
                LIMIT 1
            )
            WHERE """
            + where_clause
            + " ORDER BY "
            + order_clause
            + " LIMIT ? OFFSET ?",
            parameters,
        ).fetchall()
        return [_row_to_email(row) for row in rows]

    def _require_email(self, email_id: int) -> dict[str, Any]:
        email = self.get_email(email_id)
        if email is None:
            raise MailNotFoundError(f"Email {email_id} does not exist")
        return email

    def _update_boolean(self, email_id: int, column: str, value: bool) -> None:
        if column == "da_doc":
            statement = (
                "UPDATE thu SET da_doc = ?, thoi_gian_cap_nhat = ? WHERE id_thu = ?"
            )
        elif column == "da_gan_sao":
            statement = (
                "UPDATE thu SET da_gan_sao = ?, thoi_gian_cap_nhat = ? WHERE id_thu = ?"
            )
        else:
            raise ValueError(f"Unsupported boolean column: {column}")
        cursor = self.connection.execute(
            statement,
            (int(bool(value)), utc_timestamp(), email_id),
        )
        if cursor.rowcount != 1:
            raise MailNotFoundError(f"Email {email_id} does not exist")

    @staticmethod
    def _validate_folder(folder: str) -> None:
        if folder not in ALLOWED_FOLDERS:
            raise ValueError(f"Unsupported folder: {folder!r}")

    @staticmethod
    def _pagination(limit: int, offset: int) -> tuple[int, int]:
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("Offset must not be negative")
        return int(limit), int(offset)
