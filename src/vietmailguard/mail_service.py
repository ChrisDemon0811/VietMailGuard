"""Business logic for the local VietMailGuard Mail client."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vietmailguard.eml_parser import parse_eml_bytes
from vietmailguard.inference import analyze_email
from vietmailguard.mail_database import DEFAULT_DATABASE_PATH, open_initialized_database
from vietmailguard.mail_repository import MailNotFoundError, MailRepository
from vietmailguard.mail_router import RoutingDecision, route_analysis


Analyzer = Callable[..., dict[str, Any]]

FEEDBACK_DESTINATIONS = {
    "khong_phai_thu_rac": "hop_thu_den",
    "danh_dau_thu_rac": "thu_rac",
    "bao_cao_lua_dao": "cach_ly",
    "chuyen_vao_hop_thu_den": "hop_thu_den",
    "chuyen_vao_cach_ly": "cach_ly",
    "xoa": "da_xoa",
}

FOLDER_ACTIONS = {
    "hop_thu_den": "chuyen_vao_hop_thu_den",
    "thu_rac": "danh_dau_thu_rac",
    "cach_ly": "chuyen_vao_cach_ly",
    "da_xoa": "xoa",
}


def _fingerprint_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split())


def email_fingerprint(parsed_email: Mapping[str, Any]) -> str:
    """Hash stable parsed fields without changing the stored email content."""

    canonical = {
        field: _fingerprint_text(parsed_email.get(field, ""))
        for field in ("sender", "receiver", "cc", "date", "subject", "body")
    }
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MailService:
    """Coordinate inference, routing, persistence, and user mailbox actions."""

    def __init__(
        self,
        repository: MailRepository,
        *,
        analyzer: Analyzer = analyze_email,
        owned_connection: sqlite3.Connection | None = None,
    ) -> None:
        self.repository = repository
        self._analyzer = analyzer
        self._owned_connection = owned_connection

    @classmethod
    def from_database_path(
        cls,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        analyzer: Analyzer = analyze_email,
    ) -> "MailService":
        """Create a service that owns an initialized database connection."""

        connection = open_initialized_database(database_path)
        return cls(
            MailRepository(connection),
            analyzer=analyzer,
            owned_connection=connection,
        )

    def close(self) -> None:
        """Close the connection when this service created it."""

        if self._owned_connection is not None:
            self._owned_connection.close()
            self._owned_connection = None

    def __enter__(self) -> "MailService":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def receive_email(
        self,
        *,
        sender: object = "",
        receiver: object = "",
        cc: object = "",
        subject: object = "",
        body: object = "",
        body_html: object = "",
        sent_at: str | None = None,
        source: object = "local",
        source_id: str | None = None,
        external_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze, route, and atomically persist one received email."""

        if external_id:
            existing = self.repository.get_email_by_external_id(external_id)
            if existing is not None:
                return self._existing_email_result(existing)

        analysis = self._analyzer(sender=sender, subject=subject, body=body)
        decision = route_analysis(analysis)
        with self.repository.atomic():
            if external_id:
                existing = self.repository.get_email_by_external_id(external_id)
                if existing is not None:
                    return self._existing_email_result(existing)
            email_id = self.repository.create_email(
                sender=sender,
                receiver=receiver,
                cc=cc,
                subject=subject,
                body=body,
                body_html=body_html,
                sent_at=sent_at,
                folder="hop_thu_den",
                source=source,
                source_id=source_id,
                external_id=external_id,
            )
            if decision.folder != "hop_thu_den":
                self.repository.move_email(email_id, decision.folder, reason=decision.reason)
            analysis_id = self._save_analysis(email_id, analysis, decision)

        return {
            "email": self.repository.get_email(email_id),
            "analysis": self.repository.get_latest_analysis(email_id),
            "analysis_id": analysis_id,
            "created": True,
            "duplicate": False,
            "routing": self._routing_payload(decision),
        }

    def save_analyzed_email(
        self,
        *,
        analysis_result: Mapping[str, Any],
        sender: object = "",
        receiver: object = "",
        cc: object = "",
        subject: object = "",
        body: object = "",
        sent_at: str | None = None,
        source: str = "quick_analyzer",
        source_id: str | None = None,
        external_id: str | None = None,
        llm_review: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist an already computed, server-side result without inference.

        Callers must supply the exact input snapshot that produced the result,
        not current editable draft fields. This is not an import API for
        untrusted user-supplied prediction JSON. The default identifier binds
        content and the exact analysis so repeated Save clicks are idempotent;
        it never silently replaces a prior model's historical result.
        """
        if not _fingerprint_text(subject) and not _fingerprint_text(body):
            raise ValueError("An analyzed email must contain a subject or body")
        analysis = self._validated_analysis_snapshot(analysis_result)
        decision = route_analysis(analysis)
        review_snapshot = self._validated_review_snapshot(
            llm_review, analysis, sender=sender, subject=subject, body=body,
        )
        fingerprint = email_fingerprint({
            "sender": sender, "receiver": receiver, "cc": cc,
            "date": sent_at, "subject": subject, "body": body,
        })
        result_digest = hashlib.sha256(json.dumps(
            analysis, ensure_ascii=False, sort_keys=True, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        stable_external_id = external_id or f"analyzed-sha256:{fingerprint}:{result_digest}"
        with self.repository.atomic():
            existing = self.repository.get_email_by_external_id(stable_external_id)
            if existing is not None:
                result = self._existing_email_result(existing)
                if (review_snapshot is not None and result["analysis"] is not None
                        and result["analysis"]["result"] == analysis):
                    latest_review = self.repository.get_latest_llm_review(
                        existing["id"], result["analysis_id"],
                    )
                    if latest_review is None or latest_review["result"] != review_snapshot:
                        self.repository.save_llm_review(
                            existing["id"], result["analysis_id"], review_snapshot,
                        )
                return result
            email_id = self.repository.create_email(
                sender=sender, receiver=receiver, cc=cc, subject=subject,
                body=body, sent_at=sent_at, source=source,
                source_id=source_id or fingerprint, external_id=stable_external_id,
                folder="hop_thu_den",
            )
            if decision.folder != "hop_thu_den":
                self.repository.move_email(email_id, decision.folder, reason=decision.reason)
            analysis_id = self._save_analysis(email_id, analysis, decision)
            if review_snapshot is not None:
                self.repository.save_llm_review(email_id, analysis_id, review_snapshot)
        return {
            "email": self.repository.get_email(email_id),
            "analysis": self.repository.get_latest_analysis(email_id),
            "analysis_id": analysis_id, "created": True, "duplicate": False,
            "routing": self._routing_payload(decision),
        }

    @staticmethod
    def _validated_analysis_snapshot(result: Mapping[str, Any]) -> dict[str, Any]:
        """Reject incomplete/non-finite results, not fabricate missing metadata."""
        if not isinstance(result, Mapping):
            raise ValueError("A server-side inference result is required")
        required = {"model_version", "detected_language", "prediction", "confidence",
                    "risk_score", "risk_level"}
        if not required <= result.keys() or not str(result.get("model_version", "")).strip():
            raise ValueError("The inference result is incomplete")
        try:
            snapshot = json.loads(json.dumps(dict(result), ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("The inference result is not valid finite JSON") from error
        for field, upper in (("confidence", 1), ("risk_score", 100)):
            value = snapshot[field]
            if field == "confidence" and value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= upper):
                raise ValueError(f"Invalid inference {field}")
        probabilities = snapshot.get("class_probabilities", {})
        if not isinstance(probabilities, dict):
            raise ValueError("Invalid class probabilities")
        if probabilities and (
            set(probabilities) != {"normal", "spam", "phishing"}
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1
                   for value in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("Invalid class probabilities")
        if snapshot.get("probability_source") == "unavailable_uncalibrated" and (
            snapshot["confidence"] is not None or probabilities
        ):
            raise ValueError("Uncalibrated inference cannot contain probabilities")
        return snapshot

    @staticmethod
    def _validated_review_snapshot(
        review: Mapping[str, Any] | None, analysis: Mapping[str, Any],
        *, sender: object, subject: object, body: object,
    ) -> dict[str, Any] | None:
        """Bind optional reviewer history to exactly the saved ML result."""
        if review is None:
            return None
        from vietmailguard.v21.llm_reviewer import (
            MAX_EMAIL_FIELD_CHARS, ReviewRequest, validate_review_response,
        )
        if (not isinstance(review, Mapping) or review.get("schema_version") != 1
                or review.get("ml_result") != analysis):
            raise ValueError("Reviewer history does not match the saved ML result")
        reviewer = review.get("reviewer")
        if not isinstance(reviewer, Mapping) or reviewer.get("calibrated_probability") is not None:
            raise ValueError("Invalid reviewer history")
        if reviewer.get("status") in {"ok", "mock_only"}:
            validate_review_response(
                {key: reviewer.get(key) for key in (
                    "assessment", "summary", "evidence", "suspicious_indicators")},
                ReviewRequest(str(sender or "")[:MAX_EMAIL_FIELD_CHARS],
                              str(subject or "")[:MAX_EMAIL_FIELD_CHARS],
                              str(body or "")[:MAX_EMAIL_FIELD_CHARS]),
            )
        elif reviewer.get("status") not in {
            "disabled", "empty_input", "invalid_response", "deadline_exceeded", "provider_error",
        } or reviewer.get("assessment") is not None:
            raise ValueError("Invalid reviewer failure status")
        try:
            return json.loads(json.dumps(dict(review), ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Reviewer history is not valid finite JSON") from error

    def import_eml_bytes(
        self,
        eml_data: bytes,
        *,
        source: str = "eml_import",
        source_id: str | None = None,
        external_id: str | None = None,
    ) -> dict[str, Any]:
        """Parse, fingerprint, analyze, route, and persist raw ``.eml`` bytes."""

        if not eml_data:
            raise ValueError("The .eml input is empty")
        parsed = parse_eml_bytes(eml_data)
        if not _fingerprint_text(parsed.get("subject")) and not _fingerprint_text(
            parsed.get("body")
        ):
            raise ValueError("The .eml input has no analyzable subject or body")
        fingerprint = email_fingerprint(parsed)
        stable_external_id = external_id or f"eml-sha256:{fingerprint}"
        return self.receive_email(
            sender=parsed.get("sender", ""),
            receiver=parsed.get("receiver", ""),
            subject=parsed.get("subject", ""),
            body=parsed.get("body", ""),
            sent_at=parsed.get("date") or None,
            source=source,
            source_id=source_id or fingerprint,
            external_id=stable_external_id,
        )

    def import_eml_file(
        self,
        path: str | Path,
        *,
        source: str = "eml_import",
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Import a local ``.eml`` file without fetching external resources."""

        eml_path = Path(path)
        return self.import_eml_bytes(
            eml_path.read_bytes(),
            source=source,
            source_id=source_id,
        )

    def import_email(
        self,
        eml_data: bytes,
        *,
        source: str = "eml_import",
        source_id: str | None = None,
        external_id: str | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible alias for :meth:`import_eml_bytes`."""

        return self.import_eml_bytes(
            eml_data,
            source=source,
            source_id=source_id,
            external_id=external_id,
        )

    def list_folder(
        self,
        folder: str,
        *,
        unread: bool | None = None,
        starred: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List one internal mailbox folder."""

        emails = self.repository.list_emails(
            folder=folder,
            unread=unread,
            starred=starred,
            limit=limit,
            offset=offset,
        )
        return self._attach_latest_analysis(emails)

    def list_starred(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List starred email across folders using stored analysis only."""

        return self._attach_latest_analysis(
            self.repository.list_emails(starred=True, limit=limit, offset=offset)
        )

    def list_security_view(
        self,
        *,
        query: str = "",
        folder: str | None = None,
        unread: bool | None = None,
        starred: bool | None = None,
        prediction: str | None = None,
        risk_level: str | None = None,
        language: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List stored analyses ordered by persisted risk, with safe filters."""

        emails = self.repository.search(
            query,
            folder=folder,
            unread=unread,
            starred=starred,
            prediction=prediction,
            risk_level=risk_level,
            language=language,
            require_analysis=True,
            order_by_risk=True,
            limit=limit,
            offset=offset,
        )
        return self._attach_latest_analysis(emails)

    def mailbox_counts(self) -> dict[str, Any]:
        """Return navigation counts without loading or invoking the ML model."""

        return self.repository.mailbox_counts()

    def open_email(self, email_id: int, *, mark_as_read: bool = True) -> dict[str, Any]:
        """Return stored email, latest analysis, and auditable folder history."""

        email = self.repository.get_email(email_id)
        if email is None:
            return {
                "email": None,
                "analysis": None,
                "routing": None,
                "folder_history": [],
            }
        if mark_as_read and not email["is_read"]:
            email = self.mark_read(email_id, True)
        analysis = self.repository.get_latest_analysis(email_id)
        return {
            "email": email,
            "analysis": analysis,
            "llm_review": self.repository.get_latest_llm_review(
                email_id, None if analysis is None else analysis["id"]),
            "routing": (
                None
                if analysis is None
                else self._routing_payload(route_analysis(analysis["result"]))
            ),
            "folder_history": self.repository.list_folder_history(email_id),
        }

    def review_stored_email(self, email_id: int, *, provider: Any,
                            allow_external: bool = False,
                            explicit_email_sharing_consent: bool = False,
                            timeout_seconds: float = 5.0,
                            max_retries: int = 1) -> dict[str, Any]:
        """On-demand LLM review; append history, never reroute or recompute ML."""
        from vietmailguard.v21.llm_reviewer import review_email

        email = self.repository.get_email(email_id)
        analysis = self.repository.get_latest_analysis(email_id)
        if email is None or analysis is None:
            raise MailNotFoundError("Stored email or analysis is unavailable")
        result = review_email(
            sender=email["sender"], subject=email["subject"], body=email["body"],
            ml_result=analysis["result"], provider=provider,
            allow_external=allow_external,
            explicit_email_sharing_consent=explicit_email_sharing_consent,
            timeout_seconds=timeout_seconds, max_retries=max_retries,
        )
        with self.repository.atomic():
            review_id = self.repository.save_llm_review(email_id, analysis["id"], result)
        return {"id": review_id, "result": result}

    def mark_read(self, email_id: int, is_read: bool = True) -> dict[str, Any]:
        """Update read state and audit a genuine user-visible state change."""

        with self.repository.atomic():
            email, analysis = self._feedback_context(email_id)
            if bool(email["is_read"]) == bool(is_read):
                return email
            updated = self.repository.update_read_status(email_id, is_read)
            self.repository.save_user_feedback(
                email_id,
                initial_prediction=str(analysis["prediction"]),
                previous_folder=str(email["folder"]),
                new_folder=str(email["folder"]),
                action="danh_dau_da_doc" if is_read else "danh_dau_chua_doc",
            )
        return updated

    def toggle_star(self, email_id: int) -> dict[str, Any]:
        """Invert the starred state and audit the user action."""

        with self.repository.atomic():
            email, analysis = self._feedback_context(email_id)
            is_starred = not bool(email["is_starred"])
            updated = self.repository.update_starred_status(email_id, is_starred)
            self.repository.save_user_feedback(
                email_id,
                initial_prediction=str(analysis["prediction"]),
                previous_folder=str(email["folder"]),
                new_folder=str(email["folder"]),
                action="gan_sao" if is_starred else "bo_gan_sao",
            )
        return updated

    def move_email(
        self,
        email_id: int,
        folder: str,
        *,
        reason: str = "nguoi_dung_chuyen",
        feedback_action: str | None = None,
    ) -> dict[str, Any]:
        """Move an email, append history/feedback, and preserve ML analysis."""

        action = feedback_action or FOLDER_ACTIONS.get(folder, "nguoi_dung_chuyen")
        with self.repository.atomic():
            email, analysis = self._feedback_context(email_id)
            move = self.repository.move_email(email_id, folder, reason=reason)
            if str(move["previous_folder"]) != str(move["new_folder"]):
                self.repository.save_user_feedback(
                    email_id,
                    initial_prediction=str(analysis["prediction"]),
                    previous_folder=str(move["previous_folder"]),
                    new_folder=str(move["new_folder"]),
                    action=action,
                )
        return move

    def record_feedback(
        self,
        email_id: int,
        *,
        action: str,
        note: object = "",
    ) -> dict[str, Any]:
        """Apply a mailbox action and preserve the original ML prediction."""

        if action not in FEEDBACK_DESTINATIONS:
            raise ValueError(f"Unsupported feedback action: {action!r}")
        with self.repository.atomic():
            email = self.repository.get_email(email_id)
            if email is None:
                raise MailNotFoundError(f"Email {email_id} does not exist")
            analysis = self.repository.get_latest_analysis(email_id)
            if analysis is None:
                raise ValueError("Feedback requires an existing stored analysis")
            destination = FEEDBACK_DESTINATIONS[action]
            move = self.repository.move_email(email_id, destination, reason=action)
            feedback_id = self.repository.save_user_feedback(
                email_id,
                initial_prediction=str(analysis["prediction"]),
                previous_folder=str(move["previous_folder"]),
                new_folder=str(move["new_folder"]),
                action=action,
                note=note,
            )
        return {
            "feedback_id": feedback_id,
            "email": self.repository.get_email(email_id),
            "analysis": self.repository.get_latest_analysis(email_id),
            "move": move,
        }

    def search_email(
        self,
        query: str,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Search through the repository's parameterized query API."""

        return self._attach_latest_analysis(self.repository.search(query, **filters))

    def _feedback_context(
        self,
        email_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load immutable-analysis context required for an audited user action."""

        email = self.repository.get_email(email_id)
        if email is None:
            raise MailNotFoundError(f"Email {email_id} does not exist")
        analysis = self.repository.get_latest_analysis(email_id)
        if analysis is None:
            raise ValueError("Mailbox actions require an existing stored analysis")
        return email, analysis

    def _save_analysis(
        self,
        email_id: int,
        analysis: Mapping[str, Any],
        decision: RoutingDecision,
    ) -> int:
        prediction = str(analysis.get("prediction", ""))
        security_findings = analysis.get("security_findings", [])
        finding_count = len(security_findings) if isinstance(security_findings, list) else 0
        return self.repository.save_analysis(
            email_id,
            model_version=str(analysis.get("model_version", "unknown")),
            detected_language=str(analysis.get("detected_language", "unknown")),
            prediction=prediction,
            confidence=(None if analysis.get("confidence") is None
                        else float(analysis["confidence"])),
            risk_score=float(analysis.get("risk_score", 0.0)),
            risk_level=str(analysis.get("risk_level", "LOW")),
            base_recommended_action=decision.base_action,
            recommended_action=decision.recommended_action,
            has_warning=decision.has_warning,
            security_finding_count=finding_count,
            result=analysis,
        )

    def _existing_email_result(self, email: Mapping[str, Any]) -> dict[str, Any]:
        analysis = self.repository.get_latest_analysis(int(email["id"]))
        routing: dict[str, Any] | None = None
        if analysis is not None:
            decision = route_analysis(analysis["result"])
            routing = self._routing_payload(decision)
            routing["folder"] = str(email["folder"])
        return {
            "email": dict(email),
            "analysis": analysis,
            "analysis_id": None if analysis is None else int(analysis["id"]),
            "created": False,
            "duplicate": True,
            "routing": routing,
        }

    def _attach_latest_analysis(
        self,
        emails: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                **email,
                "analysis": self.repository.get_latest_analysis(int(email["id"])),
            }
            for email in emails
        ]

    @staticmethod
    def _routing_payload(decision: RoutingDecision) -> dict[str, Any]:
        return {
            "folder": decision.folder,
            "reason": decision.reason,
            "base_action": decision.base_action,
            "recommended_action": decision.recommended_action,
            "has_warning": decision.has_warning,
            "warning_severity": decision.warning_severity,
            "warning_reasons": list(decision.warning_reasons),
        }
