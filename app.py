import streamlit as st
import pandas as pd
import os
import time
import random
from dotenv import load_dotenv
from google import genai
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import base64
import json

load_dotenv()

st.set_page_config(
    page_title="Driver Activity Monitoring Agent",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SVG Car Definitions ──────────────────────────────────────────────────────

RED_CAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 130 52" width="130" height="52">
  <ellipse cx="65" cy="49" rx="50" ry="4" fill="rgba(0,0,0,0.35)"/>
  <rect x="10" y="25" width="108" height="18" rx="5" fill="#b71c1c"/>
  <path d="M28 25 L36 10 L85 10 L98 25 Z" fill="#c62828"/>
  <path d="M37 24 L43 12 L68 12 L68 24 Z" fill="#90caf9" opacity="0.82"/>
  <path d="M70 24 L70 12 L83 12 L93 24 Z" fill="#90caf9" opacity="0.82"/>
  <line x1="69" y1="12" x2="69" y2="24" stroke="#b71c1c" stroke-width="1.5"/>
  <rect x="6"   y="28" width="7" height="9" rx="2" fill="#7f0000"/>
  <rect x="117" y="28" width="7" height="9" rx="2" fill="#7f0000"/>
  <rect x="3"   y="27" width="7" height="5" rx="1.5" fill="#ff1a1a" opacity="0.9"/>
  <rect x="120" y="26" width="9" height="6" rx="2" fill="#fff176" opacity="0.95"/>
  <circle cx="30" cy="43" r="9" fill="#212121"/><circle cx="30" cy="43" r="5" fill="#616161"/><circle cx="30" cy="43" r="2" fill="#9e9e9e"/>
  <circle cx="95" cy="43" r="9" fill="#212121"/><circle cx="95" cy="43" r="5" fill="#616161"/><circle cx="95" cy="43" r="2" fill="#9e9e9e"/>
  <rect x="50" y="29" width="12" height="3" rx="1.5" fill="#7f0000"/>
</svg>"""

BLUE_SUV = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 145 56" width="145" height="56">
  <ellipse cx="72" cy="53" rx="58" ry="4" fill="rgba(0,0,0,0.3)"/>
  <rect x="8" y="24" width="126" height="24" rx="5" fill="#0d47a1"/>
  <path d="M22 24 L28 9 L108 9 L118 24 Z" fill="#1565c0"/>
  <path d="M30 23 L35 11 L72 11 L72 23 Z" fill="#b3d9f7" opacity="0.8"/>
  <path d="M74 23 L74 11 L105 11 L114 23 Z" fill="#b3d9f7" opacity="0.8"/>
  <line x1="73" y1="11" x2="73" y2="23" stroke="#0d47a1" stroke-width="2"/>
  <rect x="5"   y="30" width="8" height="11" rx="2" fill="#003a80"/>
  <rect x="130" y="30" width="8" height="11" rx="2" fill="#003a80"/>
  <rect x="2"   y="29" width="8" height="6"  rx="1.5" fill="#ff3d3d" opacity="0.9"/>
  <rect x="133" y="28" width="11" height="7" rx="2" fill="#fffde7" opacity="0.95"/>
  <circle cx="35"  cy="48" r="10" fill="#1a1a1a"/><circle cx="35"  cy="48" r="6" fill="#555"/><circle cx="35"  cy="48" r="2.5" fill="#aaa"/>
  <circle cx="105" cy="48" r="10" fill="#1a1a1a"/><circle cx="105" cy="48" r="6" fill="#555"/><circle cx="105" cy="48" r="2.5" fill="#aaa"/>
  <rect x="60" y="32" width="14" height="3" rx="1.5" fill="#003a80"/>
</svg>"""

YELLOW_CAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 50" width="120" height="50">
  <ellipse cx="60" cy="47" rx="46" ry="4" fill="rgba(0,0,0,0.3)"/>
  <rect x="8" y="22" width="104" height="18" rx="5" fill="#f9a825"/>
  <path d="M25 22 L33 8 L82 8 L94 22 Z" fill="#fbc02d"/>
  <path d="M34 21 L40 10 L64 10 L64 21 Z" fill="#b3e5fc" opacity="0.82"/>
  <path d="M66 21 L66 10 L80 10 L90 21 Z" fill="#b3e5fc" opacity="0.82"/>
  <line x1="65" y1="10" x2="65" y2="21" stroke="#f9a825" stroke-width="1.5"/>
  <rect x="4"   y="25" width="7" height="8" rx="2" fill="#e65100"/>
  <rect x="109" y="25" width="7" height="8" rx="2" fill="#e65100"/>
  <rect x="2"   y="25" width="7" height="5" rx="1.5" fill="#ff3d3d" opacity="0.9"/>
  <rect x="112" y="24" width="8" height="6" rx="2" fill="#fffde7" opacity="0.95"/>
  <circle cx="28" cy="41" r="9" fill="#212121"/><circle cx="28" cy="41" r="5" fill="#616161"/><circle cx="28" cy="41" r="2" fill="#9e9e9e"/>
  <circle cx="90" cy="41" r="9" fill="#212121"/><circle cx="90" cy="41" r="5" fill="#616161"/><circle cx="90" cy="41" r="2" fill="#9e9e9e"/>
  <rect x="48" y="27" width="10" height="3" rx="1.5" fill="#e65100"/>
</svg>"""

WHITE_CAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 125 50" width="125" height="50">
  <ellipse cx="62" cy="47" rx="48" ry="4" fill="rgba(0,0,0,0.3)"/>
  <rect x="8" y="22" width="108" height="18" rx="5" fill="#eceff1"/>
  <path d="M26 22 L34 8 L84 8 L96 22 Z" fill="#cfd8dc"/>
  <path d="M35 21 L41 10 L66 10 L66 21 Z" fill="#b2ebf2" opacity="0.8"/>
  <path d="M68 21 L68 10 L82 10 L92 21 Z" fill="#b2ebf2" opacity="0.8"/>
  <line x1="67" y1="10" x2="67" y2="21" stroke="#cfd8dc" stroke-width="1.5"/>
  <rect x="4"   y="25" width="7" height="9" rx="2" fill="#b0bec5"/>
  <rect x="113" y="25" width="7" height="9" rx="2" fill="#b0bec5"/>
  <rect x="2"   y="25" width="7" height="5" rx="1.5" fill="#ff1a1a" opacity="0.9"/>
  <rect x="115" y="24" width="9" height="6" rx="2" fill="#fff9c4" opacity="0.95"/>
  <circle cx="29" cy="41" r="9" fill="#212121"/><circle cx="29" cy="41" r="5" fill="#616161"/><circle cx="29" cy="41" r="2" fill="#9e9e9e"/>
  <circle cx="93" cy="41" r="9" fill="#212121"/><circle cx="93" cy="41" r="5" fill="#616161"/><circle cx="93" cy="41" r="2" fill="#9e9e9e"/>
  <rect x="50" y="27" width="11" height="3" rx="1.5" fill="#b0bec5"/>
</svg>"""

GREEN_CAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 50" width="128" height="50">
  <ellipse cx="64" cy="47" rx="50" ry="4" fill="rgba(0,0,0,0.3)"/>
  <rect x="8" y="22" width="110" height="18" rx="5" fill="#1b5e20"/>
  <path d="M26 22 L34 8 L86 8 L98 22 Z" fill="#2e7d32"/>
  <path d="M35 21 L41 10 L67 10 L67 21 Z" fill="#a5d6a7" opacity="0.78"/>
  <path d="M69 21 L69 10 L84 10 L94 21 Z" fill="#a5d6a7" opacity="0.78"/>
  <line x1="68" y1="10" x2="68" y2="21" stroke="#1b5e20" stroke-width="1.5"/>
  <rect x="4"   y="25" width="7" height="9" rx="2" fill="#145a32"/>
  <rect x="115" y="25" width="7" height="9" rx="2" fill="#145a32"/>
  <rect x="2"   y="25" width="7" height="5" rx="1.5" fill="#ff3d3d" opacity="0.85"/>
  <rect x="116" y="24" width="9" height="6" rx="2" fill="#fff9c4" opacity="0.95"/>
  <circle cx="30" cy="41" r="9" fill="#212121"/><circle cx="30" cy="41" r="5" fill="#616161"/><circle cx="30" cy="41" r="2" fill="#9e9e9e"/>
  <circle cx="95" cy="41" r="9" fill="#212121"/><circle cx="95" cy="41" r="5" fill="#616161"/><circle cx="95" cy="41" r="2" fill="#9e9e9e"/>
  <rect x="52" y="27" width="11" height="3" rx="1.5" fill="#145a32"/>
</svg>"""

SILVER_CAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 52" width="132" height="52">
  <ellipse cx="66" cy="49" rx="52" ry="4" fill="rgba(0,0,0,0.3)"/>
  <rect x="9" y="23" width="112" height="19" rx="5" fill="#78909c"/>
  <path d="M27 23 L35 9 L87 9 L99 23 Z" fill="#90a4ae"/>
  <path d="M36 22 L42 11 L68 11 L68 22 Z" fill="#e3f2fd" opacity="0.78"/>
  <path d="M70 22 L70 11 L85 11 L95 22 Z" fill="#e3f2fd" opacity="0.78"/>
  <line x1="69" y1="11" x2="69" y2="22" stroke="#78909c" stroke-width="1.5"/>
  <rect x="5"   y="26" width="7" height="9" rx="2" fill="#546e7a"/>
  <rect x="118" y="26" width="7" height="9" rx="2" fill="#546e7a"/>
  <rect x="3"   y="26" width="7" height="5" rx="1.5" fill="#ff3d3d" opacity="0.85"/>
  <rect x="119" y="25" width="10" height="6" rx="2" fill="#fff9c4" opacity="0.95"/>
  <circle cx="31" cy="43" r="9" fill="#212121"/><circle cx="31" cy="43" r="5" fill="#555"/><circle cx="31" cy="43" r="2" fill="#aaa"/>
  <circle cx="97" cy="43" r="9" fill="#212121"/><circle cx="97" cy="43" r="5" fill="#555"/><circle cx="97" cy="43" r="2" fill="#aaa"/>
  <rect x="53" y="28" width="12" height="3" rx="1.5" fill="#546e7a"/>
</svg>"""

def svg_to_data_uri(svg_str):
    encoded = base64.b64encode(svg_str.encode()).decode()
    return f"data:image/svg+xml;base64,{encoded}"

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Night highway background */
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #0d1f3c 0%, #060e1c 55%, #000508 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }

    /* Star field */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(1.5px 1.5px at  8% 12%, rgba(255,255,255,0.70) 0%, transparent 100%),
            radial-gradient(1px   1px   at 18% 35%, rgba(255,255,255,0.45) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 29%  6%, rgba(255,255,255,0.65) 0%, transparent 100%),
            radial-gradient(1px   1px   at 42% 22%, rgba(255,255,255,0.50) 0%, transparent 100%),
            radial-gradient(2px   2px   at 55%  9%, rgba(255,255,255,0.80) 0%, transparent 100%),
            radial-gradient(1px   1px   at 63% 44%, rgba(255,255,255,0.40) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 72% 18%, rgba(255,255,255,0.60) 0%, transparent 100%),
            radial-gradient(1px   1px   at 80%  5%, rgba(255,255,255,0.55) 0%, transparent 100%),
            radial-gradient(1px   1px   at 88% 31%, rgba(255,255,255,0.45) 0%, transparent 100%),
            radial-gradient(2px   2px   at 95% 14%, rgba(255,255,255,0.70) 0%, transparent 100%),
            radial-gradient(1px   1px   at 12% 55%, rgba(255,255,255,0.35) 0%, transparent 100%),
            radial-gradient(1px   1px   at 35% 48%, rgba(255,255,255,0.30) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 50% 38%, rgba(255,255,255,0.50) 0%, transparent 100%),
            radial-gradient(1px   1px   at 76% 52%, rgba(255,255,255,0.38) 0%, transparent 100%),
            radial-gradient(1px   1px   at 92% 47%, rgba(255,255,255,0.42) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    /* Road glow at bottom */
    .stApp::after {
        content: '';
        position: fixed;
        bottom: 0; left: 0;
        width: 100%; height: 220px;
        background: linear-gradient(0deg,
            rgba(0,40,80,0.55) 0%,
            rgba(0,20,45,0.35) 50%,
            transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Car animations ── */
    @keyframes driveAcross {
        0%   { transform: translateX(-200px); }
        100% { transform: translateX(110vw); }
    }
    @keyframes driveReverse {
        0%   { transform: translateX(110vw) scaleX(-1); }
        100% { transform: translateX(-200px) scaleX(-1); }
    }
    @keyframes roadDash {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-80px); }
    }
    @keyframes headlightPulse {
        0%, 100% { opacity: 0.18; }
        50%       { opacity: 0.32; }
    }

    .bg-car-layer {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
    }
    .bg-car {
        position: absolute;
        animation-name: driveAcross;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.8));
    }
    .bg-car.rev {
        animation-name: driveReverse;
    }

    /* Road dashes */
    .road-strip {
        position: fixed;
        bottom: 72px; left: 0;
        width: 200%; height: 3px;
        background: repeating-linear-gradient(
            90deg,
            rgba(240,200,60,0.65) 0px, rgba(240,200,60,0.65) 55px,
            transparent 55px, transparent 110px
        );
        animation: roadDash 1.1s linear infinite;
        pointer-events: none;
        z-index: 1;
    }
    .road-strip.mid  { bottom: 130px; opacity: 0.35; animation-duration: 1.6s; }
    .road-strip.far  { bottom: 175px; opacity: 0.18; animation-duration: 2.2s; }

    /* Road surface */
    .road-surface {
        position: fixed;
        bottom: 0; left: 0;
        width: 100%; height: 200px;
        background: linear-gradient(0deg,
            #0a0e14 0%,
            #0d1520 40%,
            transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Title ── */
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 12px rgba(0,180,255,0.5)); }
        50%       { filter: drop-shadow(0 0 28px rgba(0,180,255,0.85)) drop-shadow(0 0 55px rgba(0,100,255,0.4)); }
    }
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        background: linear-gradient(100deg, #00b4ff 0%, #e0f4ff 45%, #ff8c00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.7rem;
        text-align: center;
        padding: 18px 10px 6px;
        animation: titleGlow 3.5s ease-in-out infinite;
        position: relative; z-index: 2;
    }
    .main-subtitle {
        text-align: center;
        color: #6a8aaa;
        font-size: 1.05rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        margin-top: -4px;
        position: relative; z-index: 2;
    }

    /* ── Metric cards ── */
    @keyframes floatCard {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-5px); }
    }
    .metric-card {
        background: linear-gradient(150deg, rgba(8,20,45,0.92), rgba(4,12,28,0.97));
        backdrop-filter: blur(14px);
        border: 1px solid rgba(0,180,255,0.22);
        border-radius: 16px;
        padding: 1.3rem 1rem 1.1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 28px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: all 0.3s ease;
        animation: floatCard 4s ease-in-out infinite;
        position: relative; z-index: 2;
    }
    .metric-card:hover {
        border-color: rgba(0,180,255,0.6);
        box-shadow: 0 8px 40px rgba(0,180,255,0.18), inset 0 1px 0 rgba(255,255,255,0.07);
        transform: translateY(-5px) scale(1.025);
    }
    .metric-card .value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #00c8ff;
        text-shadow: 0 0 14px rgba(0,200,255,0.45);
        line-height: 1.1;
    }
    .metric-card .label {
        color: #7a90aa;
        font-size: 0.75rem;
        margin-top: 0.35rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .sub {
        color: #ff8c00;
        font-size: 0.75rem;
        margin-top: 0.25rem;
        font-family: 'Inter', sans-serif;
    }

    /* ── Chat messages ── */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-14px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(14px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* Agent bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        animation: slideIn 0.4s ease-out;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stMarkdownContainer"] {
        background: linear-gradient(135deg, rgba(0,40,80,0.85), rgba(0,20,50,0.92)) !important;
        border: 1px solid rgba(0,180,255,0.28) !important;
        border-radius: 0 16px 16px 16px !important;
        padding: 12px 16px !important;
        color: #d4eaf8 !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,180,255,0.08) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.93rem !important;
        line-height: 1.65 !important;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        animation: slideInRight 0.4s ease-out;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stMarkdownContainer"] {
        background: linear-gradient(135deg, #0a3d6b, #0d5c99) !important;
        border: 1px solid rgba(0,180,255,0.45) !important;
        border-radius: 16px 0 16px 16px !important;
        padding: 12px 16px !important;
        color: #ffffff !important;
        box-shadow: 0 2px 16px rgba(0,100,200,0.3) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.93rem !important;
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        background: rgba(4,12,28,0.9) !important;
        border: 1px solid rgba(0,180,255,0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(0,180,255,0.7) !important;
        box-shadow: 0 0 18px rgba(0,180,255,0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, rgba(6,18,40,0.95), rgba(10,30,60,0.95));
        color: #00b4ff;
        border: 1px solid rgba(0,180,255,0.35);
        border-radius: 10px;
        padding: 0.5rem 0.9rem;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
        transition: all 0.22s ease;
        width: 100%;
        font-family: 'Inter', sans-serif;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0a3d6b, #0d5c99);
        border-color: #00b4ff;
        color: #ffffff;
        box-shadow: 0 0 22px rgba(0,180,255,0.28);
        transform: translateY(-2px);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(4,12,28,0.85);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(0,180,255,0.14);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #7a90aa;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0a3d6b, #0d5c99) !important;
        color: #00c8ff !important;
        border: 1px solid rgba(0,180,255,0.4) !important;
    }

    /* ── Status dot ── */
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.35; }
    }
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 14px;
        background: rgba(4,12,28,0.82);
        border-radius: 20px;
        border: 1px solid rgba(0,180,255,0.18);
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        animation: blink 2s ease-in-out infinite;
    }
    .status-dot.active {
        background: #00c8ff;
        box-shadow: 0 0 8px #00c8ff;
    }
    .status-dot.warn {
        background: #fbbf24;
        box-shadow: 0 0 8px #fbbf24;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(3,9,20,0.97) !important;
        border-right: 1px solid rgba(0,180,255,0.12) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: rgba(4,12,28,0.6); }
    ::-webkit-scrollbar-thumb { background: #0d5c99; border-radius: 5px; }

    /* ensure all content sits above backgrounds */
    section.main > div { position: relative; z-index: 2; }

    /* dividers */
    hr { border-color: rgba(0,180,255,0.12) !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Driver Activity Monitoring Agent. I can analyze driver behavior, stress levels, and mood patterns from the dataset. How can I assist you today?"}
    ]

if 'df' not in st.session_state:
    try:
        df = pd.read_csv("mental_health_monitoring_dataset.csv")
        stress_map = {"Low": 1, "Medium": 2, "High": 3}
        df["Stress_Score"] = df["Stress_Level"].map(stress_map)
        st.session_state.df = df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'mental_health_monitoring_dataset.csv' is in the project directory.")
        st.stop()

# ── Gemini client ─────────────────────────────────────────────────────────────

@st.cache_resource
def init_gemini():
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            return None
        client = genai.Client(api_key=api_key)
        client.models.list()
        return client
    except Exception:
        return None

client = init_gemini()

# ── Helper functions ──────────────────────────────────────────────────────────

def get_stress_distribution():
    return st.session_state.df["Stress_Level"].value_counts().to_dict()

def get_correlation_matrix():
    df = st.session_state.df
    numeric_cols = [
        "Heart_Rate", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
        "Skin_Temperature", "Respiration_Rate", "Sleep_Duration",
        "Average_Speed", "Work_Hours", "Resilience_Factors"
    ]
    correlations = {}
    for col in numeric_cols:
        if col in df.columns:
            correlations[col] = df[col].corr(df["Stress_Score"])
    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

def get_summary_stats():
    df = st.session_state.df
    return {
        "total_records":      len(df),
        "avg_heart_rate":     float(df["Heart_Rate"].mean())     if "Heart_Rate"     in df.columns else 0,
        "avg_sleep_duration": float(df["Sleep_Duration"].mean()) if "Sleep_Duration" in df.columns else 0,
        "avg_work_hours":     float(df["Work_Hours"].mean())     if "Work_Hours"     in df.columns else 0,
        "avg_speed":          float(df["Average_Speed"].mean())  if "Average_Speed"  in df.columns else 0,
        "most_common_mood":   df["Mood"].mode()[0]               if "Mood"           in df.columns else "N/A",
        "most_common_condition": df["Driving_Conditions"].mode()[0] if "Driving_Conditions" in df.columns else "N/A",
    }

def analyze_driver_behavior():
    df = st.session_state.df
    avg_speed   = df['Average_Speed'].mean()
    max_speed   = df['Average_Speed'].max()
    stress_levels = df['Stress_Level'].value_counts()
    common_mood = df['Mood'].mode()[0] if 'Mood' in df.columns else "Unknown"
    speed_factor  = min(1, avg_speed / 80)
    stress_factor = 1 - (df['Stress_Score'].mean() / 3)
    sleep_factor  = min(1, df['Sleep_Duration'].mean() / 8) if 'Sleep_Duration' in df.columns else 0.5
    safety_score  = (speed_factor * 0.3 + stress_factor * 0.4 + sleep_factor * 0.3) * 100
    return {
        'avg_speed': avg_speed, 'max_speed': max_speed,
        'stress_distribution': stress_levels, 'common_mood': common_mood,
        'safety_score': safety_score, 'stress_factor': stress_factor,
        'sleep_factor': sleep_factor, 'speed_factor': speed_factor,
    }

def analyze_mood_safety():
    df = st.session_state.df
    if 'Mood' not in df.columns or 'Stress_Score' not in df.columns:
        return None
    mood_stress = df.groupby('Mood')['Stress_Score'].mean().sort_values()
    mood_safety = {}
    for mood in mood_stress.index:
        stress_mean  = mood_stress[mood]
        safety_score = 100 - (stress_mean * 20)
        count        = len(df[df['Mood'] == mood])
        heart_rate   = df[df['Mood'] == mood]['Heart_Rate'].mean()    if 'Heart_Rate'    in df.columns else None
        speed        = df[df['Mood'] == mood]['Average_Speed'].mean() if 'Average_Speed' in df.columns else None
        mood_safety[mood] = {
            'stress_mean': stress_mean, 'safety_score': safety_score,
            'count': count, 'heart_rate': heart_rate, 'speed': speed,
        }
    mood_df = pd.DataFrame(mood_safety).T.sort_values('safety_score', ascending=False)
    return mood_df

def process_query(question):
    df = st.session_state.df
    q = question.lower()

    # ── 1. EXISTING EXPERT TOPICS (unchanged) ─────────────────────────────

    # Mood Safety
    if "mood" in q and any(w in q for w in ["good", "safe", "drive", "best"]):
        mood_df = analyze_mood_safety()
        if mood_df is not None and len(mood_df) > 0:
            response = "**Mood Analysis for Safe Driving**\n\n**Top 3 Moods:**\n\n"
            for idx, (mood, row) in enumerate(mood_df.head(3).iterrows(), 1):
                medals = ["🥇", "🥈", "🥉"]
                response += f"{medals[idx-1]} **{mood}** — Safety Score: **{row['safety_score']:.1f}/100** | Avg Stress: **{row['stress_mean']:.2f}/3** | Drivers: **{int(row['count'])}**\n"
                if not pd.isna(row['heart_rate']) and row['heart_rate'] is not None:
                    response += f"   Heart Rate: **{row['heart_rate']:.1f} BPM**\n"
                response += "\n"
            best = mood_df.index[0]
            worst = mood_df.index[-1]
            response += f"**Best mood for driving:** {best}\n"
            response += f"**Mood to avoid:** {worst}\n"
            response += f"\nSafety gap: **{mood_df.loc[best, 'safety_score'] - mood_df.loc[worst, 'safety_score']:.1f}** points"
            return {"response": response, "type": "analysis"}
        return {"response": "Unable to analyze mood data.", "type": "error"}

    # Driver Behavior
    if "driver" in q or "behavior" in q:
        a = analyze_driver_behavior()
        response = f"**Driver Activity Analysis**\n\n"
        response += f"• Average Speed: **{a['avg_speed']:.1f} mph**\n"
        response += f"• Max Speed: **{a['max_speed']:.1f} mph**\n"
        response += f"• Most Common Mood: **{a['common_mood']}**\n"
        response += f"• Safety Score: **{a['safety_score']:.1f}/100**\n\n**Stress Distribution:**\n"
        for level, count in a['stress_distribution'].items():
            response += f"• {level}: **{count}** drivers\n"
        response += "\n**Recommendations:**\n"
        response += f"• {'Good speed management' if a['speed_factor'] > 0.6 else 'Consider reducing speed'}\n"
        response += f"• {'Good stress management' if a['stress_factor'] > 0.6 else 'High stress levels detected'}\n"
        response += f"• {'Adequate sleep' if a['sleep_factor'] > 0.6 else 'Consider improving sleep habits'}\n"
        return {"response": response, "type": "analysis"}

    # Stress Distribution
    if "distribution" in q or "stress level distribution" in q:
        result = get_stress_distribution()
        return {"response": result, "type": "distribution"}

    # Driving Conditions
    if "driving condition" in q:
        r = df["Driving_Conditions"].mode()[0] if "Driving_Conditions" in df.columns else "Unknown"
        return {"response": f"The most common driving condition is **{r}**", "type": "statistic"}

    # Most Common Mood
    if "most common mood" in q or "common mood" in q:
        r = df["Mood"].mode()[0] if "Mood" in df.columns else "Unknown"
        return {"response": f"The most common mood among drivers is **{r}**", "type": "statistic"}

    # Sleep & Stress Correlation
    if "sleep and stress" in q:
        if "Sleep_Duration" in df.columns:
            corr = df["Sleep_Duration"].corr(df["Stress_Score"])
            label = "Strong negative — more sleep reduces stress" if corr < -0.5 else "Moderate correlation" if abs(corr) < 0.5 else "Strong positive correlation"
            return {"response": f"Correlation between sleep and stress: **{corr:.2f}**\n\n{label}", "type": "statistic"}
        return {"response": "Sleep duration data not available.", "type": "error"}

    # Stress Factors
    if "factors" in q and "stress" in q:
        correlations = get_correlation_matrix()
        if correlations:
            response = "**Top Factors Related to Driver Stress:**\n\n"
            for factor, corr in correlations.items():
                response += f"• {factor}: **{corr:.3f}**\n"
            return {"response": response, "type": "correlation"}
        return {"response": "No correlation data available.", "type": "error"}

    # Summary Statistics
    if "summary" in q or "statistics" in q:
        stats = get_summary_stats()
        response = (
            f"**Driver Activity Summary**\n\n"
            f"• Total Records: **{stats['total_records']}**\n"
            f"• Avg Heart Rate: **{stats['avg_heart_rate']:.1f} BPM**\n"
            f"• Avg Sleep Duration: **{stats['avg_sleep_duration']:.1f} h**\n"
            f"• Avg Work Hours: **{stats['avg_work_hours']:.1f} h**\n"
            f"• Avg Speed: **{stats['avg_speed']:.1f} mph**\n"
            f"• Most Common Mood: **{stats['most_common_mood']}**\n"
            f"• Most Common Condition: **{stats['most_common_condition']}**"
        )
        return {"response": response, "type": "summary"}

    # ── 2. NEW: SMART GENERIC DATA QUERY HANDLER ────────────────────────────
    # This handles ANY question about dataset columns dynamically.

    # Map user-friendly names to actual column names
    COLUMN_MAP = {
        "heart rate": "Heart_Rate",
        "heart": "Heart_Rate",
        "blood pressure systolic": "Blood_Pressure_Systolic",
        "systolic": "Blood_Pressure_Systolic",
        "blood pressure diastolic": "Blood_Pressure_Diastolic",
        "diastolic": "Blood_Pressure_Diastolic",
        "blood pressure": ["Blood_Pressure_Systolic", "Blood_Pressure_Diastolic"],
        "skin temperature": "Skin_Temperature",
        "temperature": "Skin_Temperature",
        "respiration rate": "Respiration_Rate",
        "respiration": "Respiration_Rate",
        "sleep duration": "Sleep_Duration",
        "sleep": "Sleep_Duration",
        "average speed": "Average_Speed",
        "speed": "Average_Speed",
        "work hours": "Work_Hours",
        "work": "Work_Hours",
        "resilience": "Resilience_Factors",
        "stress score": "Stress_Score",
        "stress": "Stress_Score",
    }

    # Map operation words to pandas methods
    STAT_MAP = {
        "average": "mean", "mean": "mean", "avg": "mean",
        "maximum": "max", "max": "max", "highest": "max",
        "minimum": "min", "min": "min", "lowest": "min",
        "median": "median",
        "standard deviation": "std", "std": "std", "variation": "std",
    }

    matched_cols = []
    for key, col in COLUMN_MAP.items():
        if key in q:
            if isinstance(col, list):
                matched_cols.extend(col)
            else:
                matched_cols.append(col)

    # If we found a column mention, compute the requested statistic
    if matched_cols:
        matched_cols = list(set(matched_cols))  # remove duplicates

        # Detect operation (default to mean if not specified)
        operation = "mean"
        for op_key, op_val in STAT_MAP.items():
            if op_key in q:
                operation = op_val
                break

        response = f"**Analysis for {', '.join(matched_cols)}**\n\n"
        for col in matched_cols:
            if col not in df.columns:
                response += f"• {col}: Not found in dataset.\n"
                continue

            if operation == "mean":
                val = df[col].mean()
                label = "Average"
            elif operation == "max":
                val = df[col].max()
                label = "Maximum"
            elif operation == "min":
                val = df[col].min()
                label = "Minimum"
            elif operation == "median":
                val = df[col].median()
                label = "Median"
            elif operation == "std":
                val = df[col].std()
                label = "Standard Deviation"
            else:
                val = df[col].mean()
                label = "Average"

            # Format nicely
            col_display = col.replace('_', ' ')
            response += f"• {label} of **{col_display}**: **{val:.2f}**\n"

        return {"response": response, "type": "statistic"}

    # ── 3. FINAL FALLBACK (with helpful suggestions) ──────────────────────
    return {"response": (
        "I can help with specific topics like **Mood Safety**, **Driver Behavior**, **Stress Distribution**, etc.\n\n"
        "**Or you can ask me directly about any column in the dataset!**\n"
        "Available columns: Heart Rate, Blood Pressure (Systolic/Diastolic), Skin Temperature, Respiration Rate, Sleep Duration, Average Speed, Work Hours.\n\n"
        "Try asking:\n"
        "• *What is the maximum heart rate?*\n"
        "• *Show me the average sleep duration.*\n"
        "• *What is the median speed?*\n"
        f"You asked: *\"{question}\"*"
    ), "type": "help"}

# ── Plotly theme helper ───────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(4,12,28,0.0)",
    plot_bgcolor="rgba(4,12,28,0.0)",
    font=dict(color="#8aa8c8", family="Inter, sans-serif", size=12),
    title_font=dict(color="#c8dff0", family="Orbitron, sans-serif", size=14),
    xaxis=dict(
        gridcolor="rgba(0,180,255,0.08)",
        linecolor="rgba(0,180,255,0.15)",
        tickcolor="rgba(0,180,255,0.15)",
        zerolinecolor="rgba(0,180,255,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(0,180,255,0.08)",
        linecolor="rgba(0,180,255,0.15)",
        tickcolor="rgba(0,180,255,0.15)",
        zerolinecolor="rgba(0,180,255,0.1)",
    ),
    legend=dict(
        bgcolor="rgba(4,12,28,0.7)",
        bordercolor="rgba(0,180,255,0.2)",
        borderwidth=1,
        font=dict(color="#8aa8c8"),
    ),
    margin=dict(l=12, r=12, t=44, b=12),
)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<h1 class="main-title">Driver Activity Monitoring</h1>
<p class="main-subtitle">Real-time Behavior Analysis &amp; Mental Health Monitoring</p>
""", unsafe_allow_html=True)

# Status row
stats    = get_summary_stats()
analysis = analyze_driver_behavior()

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    ai_status = "AI Active" if client else "Local Mode"
    dot_class = "active" if client else "warn"
    st.markdown(f"""
    <div style="display:flex; justify-content:center; gap:12px; margin:14px 0 20px;">
      <div class="status-indicator">
        <div class="status-dot {dot_class}"></div>
        <span style="color:#8aa8c8;">{ai_status}</span>
      </div>
      <div class="status-indicator">
        <div class="status-dot active"></div>
        <span style="color:#8aa8c8;">Monitoring {len(st.session_state.df):,} Drivers</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────

st.markdown("### Live Driver Metrics")

SPEED_ICON = svg_to_data_uri("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 28">
  <rect x="4" y="10" width="40" height="13" rx="3" fill="#0d5c99"/>
  <path d="M12 10 L17 3 L34 3 L39 10 Z" fill="#1a7abf"/>
  <path d="M13 9 L17 4 L25 4 L25 9 Z" fill="#b3d9f7" opacity="0.75"/>
  <path d="M27 9 L27 4 L33 4 L37 9 Z" fill="#b3d9f7" opacity="0.75"/>
  <rect x="2"  y="12" width="5" height="4" rx="1" fill="#ff3d3d" opacity="0.9"/>
  <rect x="41" y="11" width="6" height="4" rx="1" fill="#fffde7" opacity="0.95"/>
  <circle cx="14" cy="23" r="5" fill="#111"/><circle cx="14" cy="23" r="3" fill="#555"/>
  <circle cx="34" cy="23" r="5" fill="#111"/><circle cx="34" cy="23" r="3" fill="#555"/>
</svg>""")

HEART_ICON = svg_to_data_uri("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 36">
  <polyline points="2,18 10,18 14,8 18,28 22,14 26,22 30,18 46,18"
            fill="none" stroke="#00c8ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>""")

MOON_ICON = svg_to_data_uri("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <path d="M20 4 A14 14 0 1 0 20 32 A10 10 0 1 1 20 4 Z" fill="#00c8ff" opacity="0.85"/>
  <circle cx="26" cy="8"  r="1.5" fill="#fff" opacity="0.7"/>
  <circle cx="30" cy="14" r="1"   fill="#fff" opacity="0.5"/>
  <circle cx="28" cy="20" r="1.2" fill="#fff" opacity="0.6"/>
</svg>""")

SHIELD_ICON = svg_to_data_uri("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 40">
  <path d="M18 2 L34 8 L34 20 Q34 32 18 38 Q2 32 2 20 L2 8 Z"
        fill="none" stroke="#00c8ff" stroke-width="2.2" stroke-linejoin="round"/>
  <path d="M10 20 L16 26 L26 14" fill="none" stroke="#00c8ff" stroke-width="2.5"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>""")

FACE_ICON = svg_to_data_uri("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">
  <circle cx="18" cy="18" r="15" fill="none" stroke="#00c8ff" stroke-width="2"/>
  <circle cx="13" cy="14" r="2" fill="#00c8ff"/>
  <circle cx="23" cy="14" r="2" fill="#00c8ff"/>
  <path d="M11 22 Q18 29 25 22" fill="none" stroke="#00c8ff" stroke-width="2"
        stroke-linecap="round"/>
</svg>""")

safety_color = "#00c8ff" if analysis['safety_score'] > 70 else "#fbbf24" if analysis['safety_score'] > 50 else "#f87171"
safety_label = "Good" if analysis['safety_score'] > 70 else "Moderate" if analysis['safety_score'] > 50 else "Needs Attention"

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    (col1, SPEED_ICON,  f"{stats['avg_speed']:.1f}",        "Avg Speed (mph)",   f"{stats['total_records']:,} trips",  "0s"),
    (col2, HEART_ICON,  f"{stats['avg_heart_rate']:.0f}",   "Heart Rate (BPM)",  f"{stats['most_common_mood']} mood",  "0.8s"),
    (col3, MOON_ICON,   f"{stats['avg_sleep_duration']:.1f}","Sleep Duration (h)", "Rest quality",                     "1.6s"),
    (col4, SHIELD_ICON, f"{analysis['safety_score']:.0f}",  "Safety Score",      safety_label,                        "2.4s"),
    (col5, FACE_ICON,   stats['most_common_mood'],           "Most Common Mood",  "Driver state",                      "3.2s"),
]
for col, icon, value, label, sub, delay in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay:{delay};">
          <img src="{icon}" width="42" style="margin:0 auto 8px; display:block; opacity:0.85;"/>
          <div class="value">{value}</div>
          <div class="label">{label}</div>
          <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Chat + Visualizations ─────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Agent & Analytics")

col_chat, col_viz = st.columns([1, 1])

with col_chat:
    for message in st.session_state.messages:
        with st.chat_message(
    message["role"],
    avatar="chat bot.jpg" if message["role"] == "assistant" else "driver.jpg"
):
            if message["role"] == "assistant" and isinstance(message["content"], dict):
                st.write("**Stress Level Distribution:**")
                for level, count in message["content"].items():
                    st.write(f"• {level}: **{count}**")
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about driver activity, stress, or behavior…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Analyzing driver data…"):
            result   = process_query(prompt)
            response = result['response']
            if isinstance(response, dict):
                txt = "**Stress Level Distribution:**\n\n"
                for level, count in response.items():
                    txt += f"• {level}: **{count}**\n"
                response = txt
            st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col_viz:
    tab1, tab2, tab3 = st.tabs(["Stress Analysis", "Driver Behavior", "Data Explorer"])
    df = st.session_state.df

    with tab1:
        stress_counts = df["Stress_Level"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=stress_counts.index,
            values=stress_counts.values,
            hole=0.45,
            marker=dict(
                colors=["#00c8ff","#fbbf24","#f87171"],
                line=dict(color="rgba(4,12,28,0.8)", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(color="#c8dff0", size=12),
        )])
        fig.update_layout(
            title="Stress Level Distribution",
            height=360,
            showlegend=True,
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'Mood' in df.columns:
            mood_stress = df.groupby('Mood')['Stress_Score'].mean().sort_values()
            fig2 = px.bar(
                x=mood_stress.values,
                y=mood_stress.index,
                orientation='h',
                color=mood_stress.values,
                color_continuous_scale=[[0,"#00c8ff"],[0.5,"#fbbf24"],[1,"#f87171"]],
                title="Average Stress by Mood",
            )
            fig2.update_layout(height=290, coloraxis_showscale=False, **PLOT_LAYOUT)
            fig2.update_traces(marker_line_color="rgba(4,12,28,0.6)", marker_line_width=1)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Average_Speed'],
            nbinsx=22,
            marker=dict(
                color="rgba(0,180,255,0.55)",
                line=dict(color="#00c8ff", width=1),
            ),
            name="Speed",
        ))
        fig.update_layout(
            title="Speed Distribution",
            xaxis_title="Speed (mph)",
            yaxis_title="Count",
            height=290,
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'Heart_Rate' in df.columns:
            color_map = {"Low":"#00c8ff","Medium":"#fbbf24","High":"#f87171"}
            fig2 = px.scatter(
                df, x='Heart_Rate', y='Stress_Score', color='Stress_Level',
                color_discrete_map=color_map,
                title="Heart Rate vs Stress Level",
                labels={"Heart_Rate":"Heart Rate (BPM)","Stress_Score":"Stress Score"},
                opacity=0.65,
            )
            fig2.update_traces(marker=dict(size=5, line=dict(width=0.5, color="rgba(4,12,28,0.5)")))
            fig2.update_layout(height=300, **PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            stress_filter = st.selectbox("Stress Level", ["All"] + list(df["Stress_Level"].unique()), key="stress_filter")
        with fc2:
            mood_filter = st.selectbox("Mood", ["All"] + list(df["Mood"].unique()), key="mood_filter") if "Mood" in df.columns else "All"
        with fc3:
            cond_filter  = st.selectbox("Condition",    ["All"] + list(df["Driving_Conditions"].unique()), key="cond_filter") if "Driving_Conditions" in df.columns else "All"

        filtered = df.copy()
        if stress_filter != "All":
            filtered = filtered[filtered["Stress_Level"] == stress_filter]
        if mood_filter != "All" and "Mood" in df.columns:
            filtered = filtered[filtered["Mood"] == mood_filter]
        if cond_filter  != "All" and "Driving_Conditions" in df.columns:
            filtered = filtered[filtered["Driving_Conditions"] == cond_filter]

        st.write(f"Showing **{len(filtered):,}** records")
        st.dataframe(
            filtered, use_container_width=True,
            column_config={
                "Heart_Rate":     st.column_config.NumberColumn("Heart Rate",  format="%d BPM"),
                "Sleep_Duration": st.column_config.NumberColumn("Sleep",       format="%.1f h"),
                "Work_Hours":     st.column_config.NumberColumn("Work Hours",  format="%.1f h"),
                "Average_Speed":  st.column_config.NumberColumn("Speed",       format="%.1f mph"),
                "Stress_Score":   st.column_config.NumberColumn("Stress Score",format="%d"),
            },
        )

# ── Quick Actions ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Quick Analysis Actions")

quick_actions = [
    ("Best Mood for Driving",      "which mood is good to drive"),
    ("Driver Behavior Analysis",   "Show driver behavior analysis"),
    ("Stress Distribution",        "stress level distribution"),
    ("Most Common Mood",           "most common mood"),
    ("Sleep & Stress Correlation", "sleep and stress correlation"),
    ("Driving Conditions",         "driving condition"),
    ("Stress-Related Factors",     "factors contribute most to stress"),
    ("Summary Statistics",         "summary statistics"),
]

for i in range(0, len(quick_actions), 4):
    cols = st.columns(4)
    for j in range(4):
        if i + j < len(quick_actions):
            label, query = quick_actions[i + j]
            with cols[j]:
                if st.button(label, use_container_width=True, key=f"qa_{i}_{j}"):
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.spinner("Analyzing…"):
                        result   = process_query(query)
                        response = result['response']
                        if isinstance(response, dict):
                            txt = "**Stress Level Distribution:**\n\n"
                            for level, count in response.items():
                                txt += f"• {level}: **{count}**\n"
                            response = txt
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("<div style='text-align:center;color:#4a6a8a;font-size:0.82rem;'>Real-time Driver Activity Monitoring</div>", unsafe_allow_html=True)
with f2:
    st.markdown("<div style='text-align:center;color:#4a6a8a;font-size:0.82rem;'>AI-Powered Behavior Analysis</div>", unsafe_allow_html=True)
with f3:
    st.markdown("<div style='text-align:center;color:#4a6a8a;font-size:0.82rem;'>Smart Recommendations for Safe Driving</div>", unsafe_allow_html=True)