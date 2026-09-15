"""Seed a local VietMailGuard Mail database with non-evaluation demo emails.

These hand-written messages exist only to demonstrate product behavior. Their
scenario coverage is not a scientific benchmark, and the script never reports
the resulting predictions as evaluation metrics. Every newly inserted message
is analyzed by the existing frozen inference layer and routed from its actual
ML prediction.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vietmailguard.inference import analyze_email  # noqa: E402
from vietmailguard.mail_database import DEFAULT_DATABASE_PATH  # noqa: E402
from vietmailguard.mail_service import MailService  # noqa: E402


DEMO_EMAILS: tuple[dict[str, str], ...] = (
    {
        "external_id": "vietmailguard-demo-v1-001",
        "sender": "alice@project.example",
        "receiver": "team@example.test",
        "subject": "Weekly project meeting",
        "body": "The weekly project meeting is tomorrow at 10 AM in room 204.",
        "sent_at": "2026-09-01T03:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-002",
        "sender": "library@university.example",
        "receiver": "student@example.test",
        "subject": "Book return reminder",
        "body": "Your borrowed book is due Friday. Please return it at the library desk.",
        "sent_at": "2026-09-02T02:30:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-003",
        "sender": "orders@shop.example",
        "receiver": "customer@example.test",
        "subject": "Your order has shipped",
        "body": "Order A-1042 has shipped. This notice does not request payment or passwords.",
        "sent_at": "2026-09-03T07:20:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-004",
        "sender": "it-support@company.example",
        "receiver": "staff@example.test",
        "subject": "Scheduled security maintenance",
        "body": "Email access may be unavailable Saturday. IT will never request your password by email.",
        "sent_at": "2026-09-04T09:10:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-005",
        "sender": "offers@market.example",
        "receiver": "reader@example.test",
        "subject": "Autumn clearance sale",
        "body": "Save 40 percent on selected home products this week. Browse the catalogue or unsubscribe.",
        "sent_at": "2026-09-05T05:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-006",
        "sender": "newsletter@software.example",
        "receiver": "reader@example.test",
        "subject": "Productivity bundle offer",
        "body": "Special business software bundle. Subscribe today for promotional pricing and bonus templates.",
        "sent_at": "2026-09-05T06:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-007",
        "sender": "sales@consulting.example",
        "receiver": "owner@example.test",
        "subject": "Grow your website traffic",
        "body": "Our marketing service offers search optimization packages. Request a sales quote today.",
        "sent_at": "2026-09-06T04:45:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-008",
        "sender": "promo@outlet.example",
        "receiver": "reader@example.test",
        "subject": "70% OFF TODAY",
        "body": "Huge discount today. Shop now!",
        "sent_at": "2026-09-06T08:30:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-009",
        "sender": "security-alert@account-check.example",
        "receiver": "user@example.test",
        "subject": "Account suspended",
        "body": "Your account is suspended. Verify your password immediately at http://login.account-check.example/verify.",
        "sent_at": "2026-09-07T01:10:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-010",
        "sender": "payroll@staff-access.example",
        "receiver": "employee@example.test",
        "subject": "Payroll login required",
        "body": "Confirm your payroll login and credentials now at http://192.0.2.44/secure-login.",
        "sent_at": "2026-09-07T02:10:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-011",
        "sender": "support@billing-review.example",
        "receiver": "user@example.test",
        "subject": "Payment account verification",
        "body": "Send your banking details and card information to restore account access today.",
        "sent_at": "2026-09-07T03:10:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-012",
        "sender": "winner@prize-desk.example",
        "receiver": "user@example.test",
        "subject": "You are the selected winner",
        "body": "Claim your prize by paying a processing fee and confirming private account details.",
        "sent_at": "2026-09-07T04:10:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-013",
        "sender": "lan@duan.example",
        "receiver": "nhom@example.test",
        "subject": "Lịch họp nhóm tuần này",
        "body": "Nhóm mình họp lúc 14 giờ thứ Năm để cập nhật tiến độ dự án.",
        "sent_at": "2026-09-08T02:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-014",
        "sender": "thuvien@truong.example",
        "receiver": "sinhvien@example.test",
        "subject": "Thông báo trả sách",
        "body": "Cuốn sách bạn mượn sẽ đến hạn vào thứ Sáu. Vui lòng trả tại quầy thư viện.",
        "sent_at": "2026-09-08T03:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-015",
        "sender": "donhang@cuahang.example",
        "receiver": "khachhang@example.test",
        "subject": "Đơn hàng đã được gửi",
        "body": "Đơn hàng DH-204 đã rời kho. Thông báo này không yêu cầu mật khẩu hoặc thanh toán.",
        "sent_at": "2026-09-08T04:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-016",
        "sender": "baomat@congty.example",
        "receiver": "nhanvien@example.test",
        "subject": "Bảo trì hệ thống định kỳ",
        "body": "Hệ thống sẽ bảo trì tối nay. Bộ phận IT không bao giờ yêu cầu mã OTP qua email.",
        "sent_at": "2026-09-08T05:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-017",
        "sender": "uudai@thitruong.example",
        "receiver": "bandoc@example.test",
        "subject": "Khuyến mãi mùa thu",
        "body": "Giảm giá 40 phần trăm sản phẩm gia dụng. Mua ngay hôm nay hoặc hủy nhận bản tin.",
        "sent_at": "2026-09-09T02:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-018",
        "sender": "khoahoc@daotao.example",
        "receiver": "bandoc@example.test",
        "subject": "Ưu đãi khóa học trực tuyến",
        "body": "Đăng ký khóa học hôm nay để nhận ưu đãi học phí và tài liệu miễn phí.",
        "sent_at": "2026-09-09T03:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-019",
        "sender": "sale@cuahang.example",
        "receiver": "bandoc@example.test",
        "subject": "Giảm 70% hôm nay",
        "body": "Ưu đãi lớn. Mua ngay!",
        "sent_at": "2026-09-09T04:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-020",
        "sender": "canhbao@xac-minh.example",
        "receiver": "nguoidung@example.test",
        "subject": "Tài khoản bị khóa",
        "body": "Tài khoản của bạn đã bị khóa. Hãy xác minh mật khẩu ngay tại http://xac-minh.example/dang-nhap.",
        "sent_at": "2026-09-10T01:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-021",
        "sender": "hotro@otp-check.example",
        "receiver": "nguoidung@example.test",
        "subject": "Xác minh danh tính khẩn cấp",
        "body": "Gửi mã OTP và thông tin đăng nhập trong vòng 24 giờ để tránh tạm khóa tài khoản.",
        "sent_at": "2026-09-10T02:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-022",
        "sender": "thanhtoan@bank-update.example",
        "receiver": "nguoidung@example.test",
        "subject": "Cập nhật tài khoản ngân hàng",
        "body": "Xác nhận thông tin thẻ tín dụng và đăng nhập tại http://bank-update.example/secure ngay hôm nay.",
        "sent_at": "2026-09-10T03:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-023",
        "sender": "manager@project.example",
        "receiver": "team@example.test",
        "subject": "Project Delta cần cập nhật",
        "body": "Please send the revised project plan trước cuộc họp chiều nay. No password is required.",
        "sent_at": "2026-09-11T02:00:00Z",
    },
    {
        "external_id": "vietmailguard-demo-v1-024",
        "sender": "alert@mixed-verify.example",
        "receiver": "user@example.test",
        "subject": "Security alert - xác minh ngay",
        "body": "Your account cần được xác minh ngay. Enter mật khẩu at http://mixed-verify.example/login.",
        "sent_at": "2026-09-11T03:00:00Z",
    },
)


def seed_demo_mailbox(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    analyzer: Callable[..., dict[str, Any]] = analyze_email,
    demo_emails: Sequence[dict[str, str]] = DEMO_EMAILS,
) -> dict[str, Any]:
    """Seed deterministic demo rows and return actual persistence counts."""

    created = 0
    skipped_duplicates = 0
    with MailService.from_database_path(database_path, analyzer=analyzer) as service:
        for email in demo_emails:
            result = service.receive_email(
                sender=email["sender"],
                receiver=email["receiver"],
                subject=email["subject"],
                body=email["body"],
                sent_at=email["sent_at"],
                source="demo_seed_v1",
                source_id=email["external_id"],
                external_id=email["external_id"],
            )
            created += int(bool(result["created"]))
            skipped_duplicates += int(bool(result["duplicate"]))

        folder_counts = {
            folder: len(service.list_folder(folder, limit=1000))
            for folder in ("hop_thu_den", "thu_rac", "cach_ly")
        }
        total = sum(folder_counts.values()) + len(service.list_folder("da_xoa", limit=1000))
    return {
        "created": created,
        "skipped_duplicates": skipped_duplicates,
        "total": total,
        "folder_counts": folder_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database path (default: data/runtime/vietmailguard_mail.db)",
    )
    args = parser.parse_args()
    summary = seed_demo_mailbox(args.database)
    print("VietMailGuard demo data only; these counts are not evaluation metrics.")
    print(f"Database: {args.database.resolve()}")
    print(f"Created: {summary['created']}")
    print(f"Skipped duplicates: {summary['skipped_duplicates']}")
    print(f"Total emails: {summary['total']}")
    for folder, count in summary["folder_counts"].items():
        print(f"{folder}: {count}")


if __name__ == "__main__":
    main()
