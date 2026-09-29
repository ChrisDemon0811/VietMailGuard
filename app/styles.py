"""Small, centralized visual system for the Streamlit application."""

from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    """Apply shared styles and a compact mailbox visual system."""
    st.markdown(
        """
        <style>
        :root {
            --vmg-cyan: #22d3ee;
            --vmg-blue: #2563eb;
            --vmg-navy: #0f172a;
            --vmg-muted: #64748b;
            --vmg-border: rgba(100, 116, 139, 0.28);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--vmg-border);
        }
        [data-testid="stForm"] {
            border: 1px solid var(--vmg-border);
            border-radius: 14px;
            padding: 1.15rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--vmg-border);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            background: rgba(37, 99, 235, 0.035);
        }
        .vmg-hero {
            border-left: 4px solid var(--vmg-cyan);
            padding: 0.1rem 0 0.1rem 1rem;
            margin-bottom: 1.25rem;
        }
        .vmg-eyebrow {
            color: var(--vmg-blue);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .vmg-status {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            padding: 0.45rem 0.8rem;
            border: 1px solid currentColor;
        }
        .vmg-normal { color: #15803d; background: #f0fdf4; }
        .vmg-spam { color: #92400e; background: #fffbeb; }
        .vmg-phishing { color: #b91c1c; background: #fef2f2; }
        .vmg-section-note {
            color: var(--vmg-muted);
            font-size: 0.9rem;
        }
        .vmg-confidence-kicker, .vmg-risk-kicker {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin: 0 0 0.35rem 0;
        }
        .vmg-confidence-kicker { color: #0369a1; }
        .vmg-risk-kicker { color: #b45309; }
        .vmg-signal {
            border-left: 3px solid var(--vmg-blue);
            padding: 0.55rem 0.8rem;
            margin: 0.35rem 0;
            background: rgba(37, 99, 235, 0.045);
            border-radius: 0 8px 8px 0;
        }
        .vmg-mail-badge {
            display: inline-flex;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            padding: 0.12rem 0.4rem;
            white-space: nowrap;
        }
        [class*="st-key-mail_row_"] {
            border-bottom: 1px solid #e2e8f0;
            padding: 0.35rem 0.45rem;
            min-height: 4.75rem;
            background: #fff;
            transition: background-color 140ms ease;
        }
        .st-key-mail_list { gap: 0 !important; }
        [class*="st-key-mail_row_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
            gap: 0.1rem;
        }
        [class*="st-key-mail_row_"]:hover {
            background: #f8fafc;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_select_"] button,
        [class*="st-key-mail_row_"] [class*="st-key-security_select_"] button {
            border: 0;
            background: transparent;
            box-shadow: none;
            color: #0f172a;
            padding: 0.15rem 0;
            min-height: 2.4rem;
            text-align: left;
            justify-content: flex-start;
            white-space: normal;
            overflow-wrap: anywhere;
            font-size: 0.92rem;
            font-weight: 400;
        }
        [class*="st-key-mail_row_unread_"] [class*="st-key-mail_select_"] button,
        [class*="st-key-mail_row_unread_"] [class*="st-key-security_select_"] button {
            font-weight: 600;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_select_"] button:hover,
        [class*="st-key-mail_row_"] [class*="st-key-security_select_"] button:hover {
            background: transparent;
            color: #1d4ed8;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_select_"] button > div,
        [class*="st-key-mail_row_"] [class*="st-key-security_select_"] button > div {
            justify-content: flex-start;
            text-align: left;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_star_"] button {
            color: #475569;
            font-size: 1.3rem;
            min-height: 2.4rem;
            padding: 0.1rem;
        }
        [class*="st-key-mail_row_"] button:focus-visible,
        [class*="st-key-detail_"] button:focus-visible {
            outline: 2px solid #2563eb;
            outline-offset: 2px;
        }
        [class*="st-key-mail_row_"] [data-testid="stCaptionContainer"] {
            color: #64748b;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }
        .vmg-mail-preview {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        [class*="st-key-detail_delete_"] button {
            color: #b91c1c;
        }
        [data-testid="stMainBlockContainer"] .vmg-detail-subject {
            color: #0f172a;
            font-size: clamp(1.45rem, 2vw, 1.75rem) !important;
            font-weight: 600 !important;
            line-height: 1.25;
            overflow-wrap: anywhere;
            margin: 0.5rem 0;
        }
        .vmg-mail-meta {
            color: #64748b;
            font-size: 0.87rem;
            line-height: 1.6;
            overflow-wrap: anywhere;
            margin: 0.25rem 0 0.7rem;
        }
        .vmg-mail-meta strong { color: #0f172a; font-weight: 600; }
        .st-key-mail_body {
            border-top: 1px solid #e2e8f0;
            padding: 1.2rem 0 1.5rem;
        }
        .st-key-mail_body [data-testid="stText"] span {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.95rem;
            line-height: 1.65;
            color: #0f172a !important;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .vmg-analysis-summary {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.55rem 1.1rem;
            padding: 0.55rem 0;
            color: #475569;
            font-size: 0.86rem;
            overflow-wrap: anywhere;
        }
        .vmg-analysis-summary strong {
            color: #0f172a;
            font-weight: 600;
        }
        .vmg-stored-finding {
            border-bottom: 1px solid #e2e8f0;
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.5;
            padding: 0.35rem 0;
            overflow-wrap: anywhere;
        }
        .vmg-stored-finding strong { color: #0f172a; }
        .vmg-sidebar-rule {
            height: 1px;
            background: #e2e8f0;
            margin: 0.3rem 0;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow: none;
        }
        [data-testid="stMainBlockContainer"] button[kind="tertiary"] {
            text-align: left;
        }
        @media (max-width: 760px) {
            .block-container { padding-top: 1.25rem; }
            [data-testid="stMetric"] { padding: 0.65rem 0.75rem; }
            [class*="st-key-mail_row_"] { padding: 0.4rem 0.1rem; }
            .vmg-analysis-summary { display: block; }
            .vmg-analysis-summary span:not(.vmg-mail-badge) {
                display: block;
                margin-top: 0.25rem;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            [class*="st-key-mail_row_"] { transition: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_mailbox_styles() -> None:
    """Keep the mailbox light and flat even when Streamlit defaults to dark mode."""
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #f8fafc !important;
            color: #0f172a;
        }
        [data-testid="stHeader"] {
            background: #f8fafc !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background: #fff !important;
            color: #0f172a;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            color: #0f172a;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
            background: transparent;
            border-radius: 6px;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
            background: #f8fafc;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {
            color: #475569;
        }
        [data-testid="stSidebar"] [class*="st-key-mail_nav_"] button {
            background: transparent;
            border: 0;
            border-radius: 6px;
            color: #0f172a;
            justify-content: flex-start;
            text-align: left;
            min-height: 2.4rem;
        }
        [data-testid="stSidebar"] [class*="st-key-mail_nav_"] button > div {
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] [class*="st-key-mail_nav_"] button:hover {
            background: #f8fafc;
            color: #1d4ed8;
        }
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3,
        [data-testid="stMainBlockContainer"] p,
        [data-testid="stMainBlockContainer"] label {
            color: #0f172a;
        }
        [data-testid="stMainBlockContainer"] h2 {
            font-size: 1.65rem;
            font-weight: 600;
        }
        [data-testid="stMainBlockContainer"] h3 {
            font-size: 1.2rem;
            font-weight: 600;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"],
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p,
        [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] span {
            color: #475569 !important;
            opacity: 1 !important;
        }
        [data-testid="stMainBlockContainer"] input {
            background: #fff;
            color: #0f172a;
            border-color: #cbd5e1;
        }
        [data-testid="stMainBlockContainer"] input::placeholder {
            color: #64748b;
            opacity: 1;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stPopover"] button {
            background: #fff;
            color: #334155;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stPopover"] button:hover {
            background: #f8fafc;
            border-color: #94a3b8;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
            background: #fff;
            color: #0f172a !important;
            border-color: #cbd5e1;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] button {
            color: #475569;
        }
        [data-testid="stMainBlockContainer"] button[kind="tertiary"] {
            background: transparent;
            color: #334155;
        }
        [data-testid="stMainBlockContainer"] button[kind="tertiary"]:hover {
            background: #f1f5f9;
            color: #1d4ed8;
        }
        [data-testid="stMainBlockContainer"] button:focus-visible,
        [data-testid="stSidebar"] a:focus-visible {
            outline: 2px solid #2563eb;
            outline-offset: 2px;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_select_"] button,
        [class*="st-key-mail_row_"] [class*="st-key-security_select_"] button {
            color: #0f172a !important;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_select_"] button:hover,
        [class*="st-key-mail_row_"] [class*="st-key-security_select_"] button:hover {
            background: transparent !important;
            color: #1d4ed8 !important;
        }
        [class*="st-key-mail_row_"] [class*="st-key-mail_star_"] button {
            color: #475569 !important;
        }
        [class*="st-key-detail_delete_"] button {
            color: #b91c1c !important;
        }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] { padding-left: 0.85rem; padding-right: 0.85rem; }
            [class*="st-key-mail_row_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
                display: grid;
                grid-template-columns: 1.8rem minmax(0, 1fr);
                gap: 0.1rem 0.45rem;
            }
            [class*="st-key-mail_row_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 0;
                width: auto;
            }
            [class*="st-key-mail_row_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
                height: auto !important;
            }
            [class*="st-key-mail_row_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {
                margin-bottom: 1.1rem;
            }
            [class*="st-key-mail_row_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) {
                grid-column: 2;
            }
            [class*="st-key-mail_row_"] [class*="st-key-mail_select_"] button,
            [class*="st-key-mail_row_"] [class*="st-key-security_select_"] button {
                font-size: 0.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, intro: str) -> None:
    """Render a consistent page heading."""
    st.markdown(
        f'<div class="vmg-hero"><h1>{title}</h1><p>{intro}</p></div>',
        unsafe_allow_html=True,
    )
