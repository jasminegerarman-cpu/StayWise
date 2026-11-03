"""StayWise visual language: premium hotel-tech dashboard for Streamlit.

Presentation-only layer; all predictions still come from the trained ML model.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parents[1]

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --sw-navy:#0B2944; --sw-deep:#042C41; --sw-teal:#008C8B;
  --sw-blue:#1871DF; --sw-amber:#E79116; --sw-ink:#132742;
  --sw-muted:#657991; --sw-surface:#FFFFFF; --sw-edge:#E1EAF2;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family:'DM Sans',sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:radial-gradient(ellipse at 84% 0%,rgba(210,231,255,.42),transparent 42%),
             linear-gradient(135deg,#F9FCFF 0%,#F3F8FC 100%);
  color:var(--sw-ink);
}
[data-testid="stHeader"] {background:transparent !important;}
[data-testid="stDecoration"] {display:none;}
.block-container {max-width:1540px; padding-top:1.45rem; padding-bottom:3.5rem;}
[data-testid="stMainBlockContainer"] {padding-left:2.0rem;padding-right:2.0rem;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;letter-spacing:-.04em; color:var(--sw-ink);}
h3 {font-size:1.1rem!important;}
p, li {color:#486079;}

/* BRAND / NAVIGATION */
[data-testid="stSidebar"] {
  background:radial-gradient(110% 60% at -10% 100%,rgba(15,186,170,.36),transparent 65%),
             linear-gradient(155deg,#08213C 0%,#06374B 55%,#006069 100%) !important;
  border-right:0!important;
}
[data-testid="stSidebar"] > div {background:transparent!important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top:1.3rem;}
[data-testid="stSidebar"] *,[data-testid="stSidebar"] p {color:#E4F1F5!important;}
[data-testid="stSidebar"] hr {border-color:rgba(223,245,247,.23)!important;margin:18px 0!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color:#AED1D9!important;font-size:11px!important;line-height:1.5!important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {gap:7px;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
  padding:10px 10px;border-radius:11px;border:1px solid transparent;
  font-weight:600; transition:background .2s ease, border-color .2s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
  background:rgba(255,255,255,.09);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
  background:linear-gradient(105deg,rgba(242,179,83,.28),rgba(4,168,161,.25));
  border-color:rgba(253,220,152,.42);
  box-shadow:0 5px 24px rgba(1,16,34,.18);
}
[data-testid="stSidebar"] [data-testid="stRadio"] input {accent-color:#F9BC63!important;}
.sidebar-brand {display:flex;align-items:center;gap:12px;margin:12px 0 7px;}
.sidebar-mark {display:flex;align-items:center;justify-content:center;width:47px;height:47px;
  background:linear-gradient(140deg,#FFE1A4,#EBA94F);color:#17334D!important;
  border-radius:13px;font-family:Manrope,sans-serif;font-size:24px;font-weight:900;
  box-shadow:0 9px 22px rgba(3,18,37,.22);}
.sidebar-title {font-family:Manrope,sans-serif;font-size:25px;font-weight:850;color:white!important;letter-spacing:-.045em;}
.sidebar-caption {font-size:11px;letter-spacing:.10em;color:#BBD5DD!important;margin:0 0 30px;text-transform:uppercase;}
.nav-kicker {font-size:10px;letter-spacing:.14em;color:#A7C7CE!important;margin:0 0 11px;font-weight:800;}
.nav-status {padding:11px 12px;border:1px solid rgba(176,235,224,.23);background:rgba(6,61,72,.5);border-radius:13px;font-size:12px;}
.nav-dot {display:inline-block;width:8px;height:8px;background:#48DA9D;border-radius:50%;margin-right:8px;box-shadow:0 0 0 3px rgba(72,218,157,.12);}

/* HEADER / HOTEL BANNER */
.page-hero {position:relative;overflow:hidden;min-height:172px;margin:0 0 23px;
 border-radius:22px;display:flex;align-items:center;background:linear-gradient(115deg,#fff 0%,#F6FBFF 70%,#F9EDFC 100%);
 border:1px solid #E5EBF4;box-shadow:0 16px 33px rgba(38,84,129,.045);}
.page-hero .hotel-photo {position:absolute;inset:0 0 0 auto;width:43%;background-size:cover;background-position:center 39%;}
.page-hero .hotel-photo:before {content:"";position:absolute;inset:0;background:linear-gradient(90deg,#FAFCFF 0%,rgba(250,252,255,.82) 18%,rgba(250,252,255,.08) 75%);}
.page-hero .hotel-photo:after {content:"";position:absolute;left:-15%;right:-6%;bottom:-65px;height:110px;background:#F7FBFF;border-radius:50% 0 0 0;transform:rotate(-7deg);}
.page-hero .intro-content {position:relative;z-index:2;padding:26px 33px;width:78%;}
.page-eyebrow {color:#008C8B;font-size:10px;letter-spacing:.19em;font-weight:800;text-transform:uppercase;margin-bottom:7px;}
.page-title {color:#152B47;font-family:Manrope,sans-serif;font-size:clamp(24px,3vw,34px);font-weight:850;letter-spacing:-.047em;line-height:1.18;margin:0 0 9px;}
.page-subtitle {color:#617892;line-height:1.55;font-size:13px;max-width:630px;}
.hero {position:relative;overflow:hidden;padding:35px 34px;margin:0 0 24px;min-height:195px;background:
  linear-gradient(100deg,rgba(7,39,62,.98) 0%,rgba(10,81,92,.96) 65%,rgba(14,138,133,.89) 100%);
  border-radius:23px;border:1px solid #28526A;box-shadow:0 16px 38px rgba(10,50,76,.09);}
.hero:after {content:"";position:absolute;right:-40px;top:-80px;width:290px;height:290px;border:1px solid rgba(255,255,255,.24);
  border-radius:50%;box-shadow:0 0 0 48px rgba(255,255,255,.04),0 0 0 105px rgba(255,255,255,.025);}
.hero .eyebrow {color:#F9CA87;font-weight:800;font-size:10px;letter-spacing:.2em;margin:0 0 8px;}
.hero h1 {color:white!important;font-size:clamp(25px,3vw,37px)!important;margin:0 0 11px!important;line-height:1.22;position:relative;z-index:2;}
.hero p {color:#D9EFF1;max-width:720px;line-height:1.65;font-size:13px;margin:0;position:relative;z-index:2;}

/* CONTENT CARDS */
.section-label {font-family:Manrope,sans-serif;color:#142B46;font-size:16px;font-weight:800;letter-spacing:-.025em;margin:17px 0 12px;}
.mini-card,.result-card,[data-testid="stMetric"] {background:#fff;border:1px solid var(--sw-edge);border-radius:18px;
 box-shadow:0 8px 26px rgba(28,79,115,.04);}
.mini-card {padding:20px;margin-bottom:12px;min-height:137px;}
.mini-card .overline {font-size:10px;letter-spacing:.14em;font-weight:800;color:#12928C;text-transform:uppercase;}
.mini-card strong {display:block;margin:8px 0 5px;font-family:Manrope,sans-serif;color:#1B304B;font-size:17px;}
.mini-card p {font-size:12px;line-height:1.6;color:#6D8296;}
[data-testid="stMetric"] {padding:16px 19px;}
[data-testid="stMetricLabel"] p {font-weight:700!important;color:#657B91!important;}
[data-testid="stMetricValue"] {font-family:Manrope,sans-serif;font-weight:800!important;color:#142D48!important;letter-spacing:-.045em;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px!important;}
[data-testid="stForm"] {background:#fff;border:1px solid var(--sw-edge);border-radius:20px;
 padding:24px 24px 19px;box-shadow:0 12px 28px rgba(26,65,98,.055);}
.form-lead {display:flex;gap:13px;align-items:center;margin:0 0 17px;}
.form-icon {width:46px;height:46px;display:flex;align-items:center;justify-content:center;border-radius:14px;
 background:linear-gradient(140deg,#F1F8FB,#E7F1F4);color:#046B72;font-size:22px;}
.form-lead h3,.summary-lead h3 {margin:0;font-family:Manrope,sans-serif;color:#142B48;font-size:20px;letter-spacing:-.04em;font-weight:800;}
.form-lead p,.summary-lead p {margin:2px 0 0;color:#7E8EA1;font-size:12px;}
[data-testid="stForm"] label p {font-weight:700!important;color:#233C56!important;font-size:12px!important;}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {
 background:#FCFDFE!important;border-color:#DCE5EE!important;border-radius:10px!important;
 color:#223B56!important;box-shadow:none!important;}
[data-baseweb="select"] span {color:#263D56!important;}
[data-testid="stForm"] [data-testid="stExpander"] {border:1px solid #E7EDF4;border-radius:11px;background:#FBFDFF;}

/* SCENARIOS — native Streamlit buttons stay fully functional */
.scenario-note {color:#7B8EA2;font-size:11px;margin:7px 0 14px;}
.st-key-scenario_typical button,.st-key-scenario_short button,.st-key-scenario_long button {
  border:0!important;color:#fff!important;min-height:60px!important;border-radius:12px!important;
  font-size:14px!important;font-weight:800!important;letter-spacing:-.012em;
  box-shadow:0 8px 18px rgba(26,64,105,.12)!important;transition:transform .16s ease, filter .16s ease!important;
}
.st-key-scenario_typical button {background:linear-gradient(110deg,#09AAA8,#006A78)!important;}
.st-key-scenario_short button {background:linear-gradient(110deg,#3396FF,#1159D5)!important;}
.st-key-scenario_long button {background:linear-gradient(110deg,#F6A81B,#BF6B08)!important;}
.st-key-scenario_typical button:hover,.st-key-scenario_short button:hover,.st-key-scenario_long button:hover {
  filter:brightness(1.06);transform:translateY(-2px);}
.st-key-scenario_typical button p,.st-key-scenario_short button p,.st-key-scenario_long button p {color:white!important;}
/* universal secondary-button fallback: never black-on-black */
div.stButton > button[kind="secondary"],div.stDownloadButton > button {background:#fff!important;border:1px solid #DCE6EF!important;color:#193851!important;border-radius:10px!important;font-weight:700;}
div.stButton > button[kind="secondary"] p {color:inherit!important;}
div.stButton > button[kind="secondary"]:hover,div.stDownloadButton > button:hover {border-color:#0DA3A0!important;color:#057B79!important;background:#F5FFFE!important;}
/* Strong specificity against the generic secondary button rules. The st-key class
   can be on the wrapper OR on the stButton element depending on Streamlit. */
.st-key-scenario_typical div.stButton > button[kind="secondary"],div.stButton.st-key-scenario_typical > button[kind="secondary"] {background:linear-gradient(110deg,#09AAA8,#006A78)!important;color:#fff!important;border:none!important;}
.st-key-scenario_short div.stButton > button[kind="secondary"],div.stButton.st-key-scenario_short > button[kind="secondary"] {background:linear-gradient(110deg,#3396FF,#1159D5)!important;color:#fff!important;border:none!important;}
.st-key-scenario_long div.stButton > button[kind="secondary"],div.stButton.st-key-scenario_long > button[kind="secondary"] {background:linear-gradient(110deg,#F6A81B,#BF6B08)!important;color:#fff!important;border:none!important;}
.st-key-scenario_typical div.stButton > button p,.st-key-scenario_short div.stButton > button p,.st-key-scenario_long div.stButton > button p {color:#fff!important;}
/* Primary actions */
button[kind="primary"], [data-testid="stFormSubmitButton"] button {
  background:linear-gradient(100deg,#05AAA8,#00636E)!important;color:#fff!important;
  border:0!important;border-radius:11px!important;min-height:49px!important;
  font-weight:800!important;box-shadow:0 9px 19px rgba(3,126,127,.2)!important;}
button[kind="primary"] p,[data-testid="stFormSubmitButton"] button p {color:white!important;}
button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] button:hover {background:linear-gradient(100deg,#07BCB5,#007F88)!important;}

/* PREDICTION PANEL */
.assessment-panel {background:#fff;border:1px solid #E0EAF2;border-radius:20px;padding:23px;
 box-shadow:0 12px 28px rgba(26,65,98,.055);margin-bottom:14px;}
.summary-lead {display:flex;align-items:center;gap:11px;margin-bottom:17px;}
.summary-icon {width:44px;height:44px;display:flex;align-items:center;justify-content:center;background:#FFF3E4;
 color:#D97810;border-radius:13px;font-size:21px;flex-shrink:0;}
.summary-lead h3 {font-size:17px;}
.risk-summary {background:linear-gradient(132deg,#FFF7EE,#FFF2E7);border-radius:15px;padding:18px 19px 17px;}
.risk-summary .risk-eyebrow {font-weight:800;font-size:13px;color:#A85416;margin-bottom:5px;}
.risk-big {font-size:49px;font-family:Manrope,sans-serif;font-weight:850;color:#D46A15;letter-spacing:-.07em;line-height:1.2;}
.risk-summary.risk-low {background:linear-gradient(130deg,#F1FBF6,#EAF8F3);}
.risk-summary.risk-low .risk-eyebrow,.risk-summary.risk-low .risk-big {color:#108060;}
.risk-summary.risk-high {background:linear-gradient(130deg,#FFF3F1,#FFEDEA);}
.risk-summary.risk-high .risk-eyebrow,.risk-summary.risk-high .risk-big {color:#D3433C;}
.risk-pill {display:inline-block;border-radius:99px;padding:5px 10px;margin-left:9px;
 vertical-align:middle;background:#FFE0AE;color:#99520C;font-size:11px;font-weight:800;letter-spacing:0;}
.risk-pill.low {background:#D3F4E6;color:#0A785A;}.risk-pill.high {background:#FFD9D6;color:#C63736;}
.risk-meter {margin:13px 0 8px;height:13px;border-radius:99px;background:#E5E9EF;overflow:hidden;}
.risk-meter > div {height:100%;border-radius:99px;background:linear-gradient(90deg,#FFB32E,#F08917);}
.risk-summary.risk-low .risk-meter > div {background:linear-gradient(90deg,#32C99A,#068B77);}
.risk-summary.risk-high .risk-meter > div {background:linear-gradient(90deg,#FF8175,#DC4340);}
.risk-scale {display:flex;justify-content:space-between;color:#7C8DA1;font-size:11px;}
.interpretation {margin-top:15px;padding:15px 16px;background:linear-gradient(135deg,#EDF7F9,#F2F8FA);
 border-radius:13px;color:#2F5969;font-size:12px;line-height:1.65;}
.interpretation strong {display:block;color:#065F70;font-size:13px;margin-bottom:3px;}
.risk-key {border-top:1px solid #E5EAF1;margin-top:18px;padding-top:14px;}
.risk-key strong {display:block;font-weight:800;font-size:13px;color:#213A54;margin-bottom:12px;}
.risk-row {display:flex;gap:8px;justify-content:space-between;margin:8px 0;font-size:12px;color:#35516A;}
.risk-row .risk-name {display:flex;gap:9px;align-items:center;font-weight:700;}
.risk-swatch {display:inline-block;width:11px;height:11px;border-radius:50%;}
.risk-swatch.green {background:#28C981}.risk-swatch.amber {background:#F49A20}.risk-swatch.red {background:#EC514D}
.ready-placeholder {display:flex;align-items:center;flex-direction:column;text-align:center;padding:20px 8px 5px;}
.ready-placeholder .ready-icon {background:#EAF6F8;color:#098B87;border-radius:50%;width:65px;height:65px;
 display:flex;align-items:center;justify-content:center;font-size:28px;margin-bottom:13px;}
.ready-placeholder h4 {font-size:16px;margin:0 0 5px;}
.ready-placeholder p {max-width:250px;font-size:12px;color:#718398;line-height:1.65;}
.outcome-label {font-size:11px;color:#7A8B9B;margin-top:12px;line-height:1.55;}
.footer {margin-top:40px;padding:19px 0;border-top:1px solid #E0E8EF;display:flex;gap:16px;justify-content:space-between;
 font-size:11px;color:#7F91A4;}
[data-testid="stDataFrame"], [data-testid="stFileUploader"] {border:1px solid #E0E9F1;border-radius:13px;overflow:hidden;}
[data-testid="stFileUploader"] {background:white;padding:12px;}
@media (max-width:1000px) {
  .page-hero .intro-content {width:100%;}
  .page-hero .hotel-photo {opacity:.23;width:54%;}
  [data-testid="stMainBlockContainer"] {padding-left:1rem;padding-right:1rem;}
  .risk-big {font-size:39px;}
}
@media (max-width:690px) {
  .block-container {padding-top:1.0rem;}
  .page-hero {min-height:152px;}
  .page-hero .intro-content {padding:21px;}
  .page-hero .hotel-photo {width:72%;opacity:.13;}
  .page-title {font-size:23px;}
  .hero {padding:25px 21px;}
  .hero h1 {font-size:24px!important;}
  [data-testid="stForm"] {padding:16px;}
  .assessment-panel {padding:16px;}
  .footer {display:block;line-height:2;}
}
</style>
"""


@lru_cache(maxsize=1)
def _hero_image_data() -> str:
    asset = BASE / "assets" / "hotel-banner.jpg"
    if not asset.is_file():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(asset.read_bytes()).decode("ascii")


def install_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "OPERATIONS INTELLIGENCE") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def heading(title: str, subtitle: str = "", eyebrow: str = "STAYWISE / PLATFORM") -> None:
    photo = _hero_image_data()
    photo_div = (
        f"<div class='hotel-photo' style=\"background-image:url('{photo}')\"></div>"
        if photo else ""
    )
    st.markdown(
        f'<div class="page-hero">{photo_div}<div class="intro-content">'
        f'<div class="page-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="page-title">{escape(title)}</div>'
        f'<div class="page-subtitle">{escape(subtitle)}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{escape(label)}</div>', unsafe_allow_html=True)


def info_card(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="mini-card"><div class="overline">{escape(eyebrow)}</div>'
        f'<strong>{escape(title)}</strong><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )
