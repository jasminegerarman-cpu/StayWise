"""Visual system for a polished, responsive Streamlit dashboard."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#18283A; --muted:#6D7C8D; --accent:#137D75; --dark:#0D2333; --gold:#D6A866; --line:#E3E9EF; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans', sans-serif;}
[data-testid="stAppViewContainer"] {background:#F5F8FA;color:var(--ink);}
[data-testid="stSidebar"] {background:linear-gradient(164deg,#0C2536,#143B48 70%,#1A504F);border-right:0;}
[data-testid="stSidebar"] * {color:#D9E8E8!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:9px 4px;border-radius:8px;}
[data-testid="stSidebar"] hr {border-color:#38616B;}
[data-testid="stHeader"] {background:rgba(245,248,250,.92);}
.block-container {max-width:1460px;padding-top:1.55rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#1A2D3D;}
h1{font-weight:800!important;}
h3{font-size:1.05rem!important;}
[data-testid="stMetric"] {background:#FFF;border:1px solid #E3E9EF;border-radius:15px;padding:18px 18px 15px;box-shadow:0 5px 15px rgba(15,42,57,.025);}
[data-testid="stMetricLabel"] {font-weight:600;letter-spacing:.008em;}
[data-testid="stMetricValue"] {font-family:Manrope,sans-serif;font-size:1.65rem;letter-spacing:-.045em;color:var(--ink);}
[data-testid="stForm"] {background:#FFF;padding:1.5rem 1.65rem;border:1px solid #E2E9ED;border-radius:17px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px!important;}
div.stButton>button[kind="primary"], div.stFormSubmitButton>button[kind="primary"] {background:#147B73;color:white;border:none;border-radius:9px;font-weight:700;padding:.65em 1.4em;box-shadow:0 4px 12px rgba(20,123,115,.17);}
div.stButton>button[kind="primary"]:hover,div.stFormSubmitButton>button[kind="primary"]:hover {background:#0E645F;color:white;}
div.stButton>button,div.stDownloadButton>button{border-radius:9px;border-color:#DDE5EA;font-weight:600;}
[data-testid="stFileUploader"] {background:#FFF;border:1px solid #E3E9EF;padding:1.1rem;border-radius:14px;}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
.stTabs [data-baseweb="tab-list"] {gap:10px;}
.stTabs [data-baseweb="tab"] {font-weight:700;}
.hero{position:relative;overflow:hidden;background:linear-gradient(110deg,#102D3E 0%,#174750 65%,#19695F 100%);border-radius:20px;padding:32px 36px;margin:0 0 22px;min-height:170px;color:#fff;}
.hero:after{content:'';position:absolute;width:310px;height:310px;border:1px solid rgba(255,255,255,.16);border-radius:50%;right:-75px;top:-145px;box-shadow:0 0 0 55px rgba(255,255,255,.04),0 0 0 110px rgba(255,255,255,.03);}
.hero .eyebrow{color:#E7C28E;letter-spacing:.19em;font-size:11px;font-weight:800;text-transform:uppercase;margin-bottom:8px;}
.hero h1{color:white!important;font-size:32px!important;line-height:1.15;margin:0 0 10px!important;max-width:720px;}
.hero p{max-width:730px;color:#C7DEE2;font-size:14px;line-height:1.65;margin:0;}
.page-eyebrow{color:#137D75;letter-spacing:.18em;font-size:11px;font-weight:800;text-transform:uppercase;margin:2px 0 7px;}
.page-title{font-size:29px;font-family:Manrope,sans-serif;letter-spacing:-.04em;font-weight:800;color:#1B2D3E;margin:0 0 4px;}
.page-subtitle{color:#758492;font-size:13px;margin-bottom:18px;}
.section-label{font-family:Manrope,sans-serif;font-weight:800;color:#23384A;font-size:16px;letter-spacing:-.02em;margin:16px 0 12px;}
.mini-card{background:#FFF;border:1px solid #E3E9EF;border-radius:15px;padding:18px 20px;margin-bottom:12px;}
.mini-card .overline{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:#8896A3;font-weight:700;}
.mini-card strong{display:block;color:#233C4A;font-size:17px;margin-top:5px;font-weight:800;}
.mini-card p{font-size:12px;color:#667686;margin:6px 0 0;line-height:1.55;}
.result-card{padding:25px;border-radius:17px;background:#FFFFFF;border:1px solid #E3E9EF;box-shadow:0 10px 27px rgba(18,47,64,.04);}
.result-card .result-title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#7A8998;font-weight:700;}
.result-card .huge{font-family:Manrope,sans-serif;font-size:59px;letter-spacing:-.07em;line-height:1.1;font-weight:800;color:#1B3944;margin:7px 0;}
.pill{display:inline-block;font-size:12px;font-weight:800;letter-spacing:.03em;border-radius:30px;padding:7px 12px;}
.pill-low{background:#E2F6EE;color:#116B55}.pill-moderate{background:#FFF1D5;color:#9D661C}.pill-high{background:#FDEAE7;color:#B74736}
.footnote{font-size:12px;color:#7A8998;line-height:1.7;margin-top:12px;}
.sidebar-brand{display:flex;align-items:center;gap:11px;margin:15px 0 6px;}
.sidebar-mark{display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#E9C18D,#DDA666);color:#203F49!important;width:42px;height:42px;border-radius:12px;font-size:23px;font-weight:900;}
.sidebar-title{font-family:Manrope,sans-serif;font-size:23px;color:#FFF!important;font-weight:800;letter-spacing:-.045em;}
.sidebar-caption{color:#B2CBCD!important;font-size:11px;letter-spacing:.035em;margin:3px 0 28px;}
.footer{margin-top:42px;border-top:1px solid #E2E9EF;padding:18px 0;color:#8795A3;font-size:11px;display:flex;justify-content:space-between;gap:20px;}
@media (max-width:700px){.hero{padding:25px}.hero h1{font-size:24px!important}.block-container{padding-top:1.1rem}.result-card .huge{font-size:42px}.footer{display:block;}}
</style>
"""


def install_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, eyebrow="OPERATIONS INTELLIGENCE"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def heading(title, subtitle="", eyebrow="STAYWISE / PLATFORM"):
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div><div class="page-title">{title}</div><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def info_card(eyebrow, title, description):
    st.markdown(f'<div class="mini-card"><div class="overline">{eyebrow}</div><strong>{title}</strong><p>{description}</p></div>', unsafe_allow_html=True)
