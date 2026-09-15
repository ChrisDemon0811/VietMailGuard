"""Small, centralized visual system for the Streamlit application."""

from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    """Apply restrained cybersecurity-themed styling without changing app logic."""
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
        .vmg-normal { color: #15803d; background: rgba(34, 197, 94, 0.10); }
        .vmg-spam { color: #b45309; background: rgba(245, 158, 11, 0.11); }
        .vmg-phishing { color: #b91c1c; background: rgba(239, 68, 68, 0.10); }
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
        .vmg-mail-header {
            display: flex;
            justify-content: space-between;
            align-items: end;
            border-bottom: 1px solid var(--vmg-border);
            margin-bottom: 0.35rem;
            padding-bottom: 0.65rem;
        }
        .vmg-mail-header h1 {
            margin: 0.1rem 0 0.2rem 0;
            font-size: 1.85rem;
        }
        .vmg-mail-header p {
            color: var(--vmg-muted);
            margin: 0;
        }
        .vmg-mail-badge {
            display: inline-flex;
            border-radius: 999px;
            border: 1px solid currentColor;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            padding: 0.2rem 0.45rem;
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
