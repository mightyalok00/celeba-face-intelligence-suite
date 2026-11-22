"""
UI Styling and Custom CSS Module for CelebA Face Intelligence Suite.
Encapsulates theme variables, glassmorphic card designs, animated motion sliders,
and typography styling.
"""

import streamlit as st

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Dancing+Script:wght@600;700&family=Great+Vibes&display=swap" rel="stylesheet">

<style>
    /* CSS Variables for Trending Color Palette */
    :root {
        --bg-surface: #0b0f19;
        --card-bg: rgba(17, 24, 39, 0.75);
        --card-border: rgba(99, 102, 241, 0.25);
        --accent-indigo: #6366f1;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
    }

    /* Base Font & Layout Adjustments */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Glowing Hero Card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.9) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 20px;
        padding: 26px 36px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 16px;
        font-weight: 400;
    }

    /* Live Telemetry Bar */
    .telemetry-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
    }

    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #e2e8f0;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* =========================================================================
       ANIMATED MOTION SLIDERS (CONTINUOUS CAROUSEL & STREAMLIT SLIDER ANIMATION)
       ========================================================================= */
    .motion-slider-wrapper {
        margin-bottom: 24px;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 15, 25, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 18px;
        padding: 16px 20px 20px 20px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        overflow: hidden;
    }

    .motion-slider-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }

    .motion-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(6, 182, 212, 0.25));
        border: 1px solid rgba(99, 102, 241, 0.45);
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .motion-slider-container {
        width: 100%;
        overflow: hidden;
        position: relative;
        mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent);
        -webkit-mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent);
    }

    .motion-slider-track {
        display: flex;
        gap: 16px;
        width: max-content;
        animation: continuousSlide 30s linear infinite;
    }

    .motion-slider-track:hover {
        animation-play-state: paused;
    }

    @keyframes continuousSlide {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    .motion-card {
        flex: 0 0 160px;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 10px;
        text-align: center;
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), border-color 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }

    .motion-card:hover {
        transform: translateY(-6px) scale(1.05);
        border-color: rgba(56, 189, 248, 0.7);
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.3);
    }

    .motion-card img {
        width: 100%;
        height: 135px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 8px;
    }

    .motion-card-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #f1f5f9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .motion-card-tag {
        font-size: 0.72rem;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border-radius: 4px;
        padding: 2px 6px;
        margin-top: 4px;
        display: inline-block;
        font-weight: 500;
    }

    /* Streamlit Interactive Sliders: Animated Motion Styling */
    .stSlider {
        padding-top: 6px;
        padding-bottom: 8px;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981, #f59e0b, #6366f1) !important;
        background-size: 300% 100% !important;
        animation: sliderGradientMotion 6s linear infinite !important;
        border-radius: 999px !important;
        height: 8px !important;
    }

    @keyframes sliderGradientMotion {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stSlider [role="slider"] {
        border: 2px solid #38bdf8 !important;
        background: #0f172a !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.7) !important;
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.2s ease !important;
        animation: thumbGlowMotion 2.5s infinite ease-in-out !important;
        width: 20px !important;
        height: 20px !important;
    }

    .stSlider [role="slider"]:hover, .stSlider [role="slider"]:active {
        transform: scale(1.3) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.95), 0 0 35px rgba(6, 182, 212, 0.7) !important;
    }

    @keyframes thumbGlowMotion {
        0%, 100% {
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.5);
        }
        50% {
            box-shadow: 0 0 18px rgba(129, 140, 248, 0.9), 0 0 25px rgba(6, 182, 212, 0.5);
        }
    }

    /* Metric Display Units */
    .metric-badge-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-badge-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .metric-badge-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-badge-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }

    /* Chips & Tags */
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .chip-success {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .chip-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #67e8f9;
        border-color: rgba(6, 182, 212, 0.3);
    }

    /* Calligraphy Footer */
    .calligraphy-footer {
        margin-top: 50px;
        padding: 30px 0 20px 0;
        text-align: center;
        position: relative;
    }

    .footer-divider-gradient {
        height: 1px;
        background: linear-gradient(90deg, rgba(0,0,0,0) 0%, rgba(99, 102, 241, 0.5) 50%, rgba(0,0,0,0) 100%);
        border: none;
        margin-bottom: 24px;
    }

    .calligraphy-author {
        font-family: 'Great Vibes', 'Dancing Script', cursive;
        font-size: 3.4rem;
        font-weight: 400;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin: 4px 0 8px 0;
        letter-spacing: 1px;
        text-shadow: 0 4px 15px rgba(129, 140, 248, 0.25);
    }

    .footer-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 400;
    }

    .footer-tags {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 10px;
        font-size: 0.8rem;
        color: #64748b;
    }
</style>
"""


def apply_custom_styles() -> None:
    """Injects custom CSS styles and typography definitions into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
