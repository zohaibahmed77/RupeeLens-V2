"""RupeeLens V2 | compact single-screen Streamlit interface.

This remains a ONE-denomination-per-photo classifier. Multi-note counting,
condition assessment and counterfeit detection are NOT implemented.

Run from D:\\RupeeLens_V2 with the existing trained model:
  python -m streamlit run rupeelens_v2_app.py
"""

from __future__ import annotations

from hashlib import sha256
from html import escape
from io import BytesIO
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT / "runs" / "classifier_smoke" / "weights" / "best.pt"
SUPPORTED = {"10", "20", "50", "100", "500", "1000"}
MIN_CONFIDENCE = 0.60
MIN_MARGIN = 0.30
ANALYSIS_MIN_DISPLAY_SECONDS = 2.8  # UI feedback; actual inference may take longer
STARTUP_MIN_DISPLAY_SECONDS = 2.4  # Keep the mascot visible briefly on first load

st.set_page_config(
    page_title="RupeeLens V2 | Pakistani Currency AI",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root { --rl-green:#07876b; --rl-ink:#162d38; --rl-muted:#71808a; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(140deg, #f7faf9 0%, #f5f9fb 55%, #f9fbfb 100%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    max-width: 1440px !important;
    padding-top: .75rem !important;
    padding-bottom: .8rem !important;
    padding-left: 1.6rem !important;
    padding-right: 1.6rem !important;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stVerticalBlock"] { gap: .55rem; }
[data-testid="stHorizontalBlock"] { gap: 1.35rem; }
.rl-header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; margin-bottom: .8rem;
}
.rl-brand { font-size: 2.0rem; font-weight: 850; letter-spacing: -.055em; color: var(--rl-ink); line-height: 1.1; }
.rl-brand span { color: var(--rl-green); }
.rl-subtitle { font-size: .84rem; color: #75868e; margin-top: .2rem; }
.rl-pill {
    padding: .38rem .75rem; background: #e8f5f0; border: 1px solid #cde9df;
    border-radius: 999px; color: #16765b; font-size: .78rem; font-weight: 650;
    white-space: nowrap;
}
.rl-dot { display:inline-block; width:7px; height:7px; margin-right:6px;
    background:#0ea77d; border-radius:100%; vertical-align:middle; }
.rl-panel-label {
    color: #7a8b94; font-size: .74rem; letter-spacing: .12em;
    text-transform: uppercase; font-weight: 800; margin: .12rem 0 .2rem;
}
.rl-panel-title { font-size: 1.17rem; font-weight: 780; color:#213b47; margin-bottom:.18rem; }
.rl-small { color: #74858e; font-size: .79rem; line-height:1.4; }
.rl-image-empty {
    height: 315px; border-radius: 16px; background: #eef3f4;
    border: 1px dashed #c9d9d9; display:flex; align-items:center;
    justify-content:center; flex-direction:column; text-align:center; gap:8px;
    color:#7d9195; padding: 1.5rem;
}
.rl-image-empty .rl-camera { font-size:2.5rem; line-height:1.2; }
div[data-testid="stImage"] img {
    border-radius: 16px; border: 1px solid #e0e9e8;
    max-height: 350px; object-fit: contain; background: #edf1f2;
}
div[data-testid="stFileUploader"] section {
    background:#fff; border: 1px dashed #b7cfcb;
    border-radius: 12px; padding: .5rem;
}
div[data-testid="stFileUploader"] section:hover { border-color:#0d9c7d; }
/* Primary Analyze Banknote button: green, clean, no icon. */
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
div.stButton > button[kind="primary"] {
    background: linear-gradient(110deg, #0e9b73 0%, #087b61 100%) !important;
    color: #ffffff !important;
    border: 1px solid #087b61 !important;
    border-radius: 12px !important;
    min-height: 47px !important;
    padding: .62rem 1rem !important;
    font-size: 1rem !important;
    font-weight: 750 !important;
    letter-spacing: .01em !important;
    box-shadow: 0 6px 16px rgba(8, 123, 97, .15) !important;
    transition: background .2s ease, box-shadow .2s ease, transform .2s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled),
div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover:not(:disabled),
div.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: linear-gradient(110deg, #0a8564 0%, #075f4d 100%) !important;
    border-color: #075f4d !important;
    box-shadow: 0 8px 20px rgba(8, 123, 97, .21) !important;
}
div[data-testid="stButton"] button[kind="primary"]:focus-visible,
div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:focus-visible {
    outline: 3px solid #91dfc4 !important;
    outline-offset: 2px !important;
}
div[data-testid="stButton"] button[kind="primary"]:disabled,
div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:disabled {
    background: #a9c9be !important;
    border-color: #a9c9be !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
div[data-testid="stRadio"] { background: transparent; }
.rl-upload-note { font-size:.77rem; color:#768a92; margin-top:.25rem; }
.rl-result {
    background: #fff; border:1px solid #e4ebeb;
    border-radius: 18px; box-shadow: 0 10px 30px rgba(31, 74, 70, .055);
    padding: 1.3rem 1.4rem; min-height: 435px;
}
.rl-status { display:inline-block; border-radius:999px; font-size:.73rem;
    font-weight:800; padding:.35rem .7rem; letter-spacing:.025em; }
.rl-status-good { background:#e3f7ed; color:#08724e; }
.rl-status-uncertain { background:#fff1da; color:#a06715; }
.rl-amount { color:#173c37; font-size:3.15rem; font-weight:850;
    letter-spacing:-.045em; line-height:1.12; margin: .9rem 0 .15rem; }
.rl-unknown { color:#8b601c; font-size:2.0rem; font-weight:800;
    letter-spacing:-.035em; line-height:1.2; margin:.95rem 0 .35rem; }
.rl-result-caption { color:#73868b; font-size:.86rem; margin-bottom:1.2rem; }
.rl-stats { display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin:1.2rem 0; }
.rl-stat { background:#f6f9f9; border-radius:12px; padding:.85rem .9rem; }
.rl-stat-label { font-size:.76rem; color:#819098; margin-bottom:.3rem; }
.rl-stat-value { color:#183b3d; font-size:1.6rem; font-weight:800; line-height:1.2; }
.rl-section { border-top:1px solid #e9eeee; padding-top:.85rem; margin-top:1rem; }
.rl-section-title { font-weight:770; color:#344c54; font-size:.82rem; margin-bottom:.6rem; }
.rl-rank-line { display:flex; justify-content:space-between; gap:.7rem;
    font-size:.82rem; color:#4e676e; margin:.45rem 0 .2rem; }
.rl-bar { background:#e8f0ef; border-radius:100px; height:6px; overflow:hidden; }
.rl-bar > div { background:#0c9473; height:100%; border-radius:100px; }
.rl-bar.secondary > div { background:#9fbbc1; }
.rl-prompt {
    display:flex; flex-direction:column; justify-content:center; align-items:center;
    text-align:center; color:#728991; padding:2.7rem 1rem 1rem;
}
.rl-prompt-icon { background:#e8f5f0; border-radius:18px; padding:14px 19px;
    font-size:2rem; margin-bottom:.9rem; }
.rl-prompt-head { color:#284952; font-size:1.25rem; font-weight:800; margin-bottom:.35rem; }
.rl-prompt-copy { max-width:275px; font-size:.87rem; line-height:1.55; }
.rl-bottom-note { margin-top:.55rem; color:#8b979c; font-size:.74rem; line-height:1.4; }

/* A dedicated startup splash while the trained model is being loaded. */
.rl-boot-screen {
    position:fixed; inset:0; z-index:2147483600;
    display:flex; align-items:center; justify-content:center;
    box-sizing:border-box; padding:24px;
    background:radial-gradient(circle at 50% 37%, #e5f8ef 0%, #f5fcf9 45%, #f8fbfc 100%);
    font-family:inherit; animation:rl-boot-fade .22s ease-out both;
}
.rl-boot-panel {
    width:min(450px, 100%); padding:42px 34px 33px;
    border:1px solid #d9eee3; border-radius:28px; text-align:center;
    background:rgba(255,255,255,.97);
    box-shadow:0 25px 75px rgba(7,85,66,.11);
    box-sizing:border-box;
}
.rl-boot-kicker {
    display:inline-flex; align-items:center; gap:7px;
    font-weight:800; letter-spacing:.12em; font-size:11px;
    color:#0a8b6b; text-transform:uppercase;
}
.rl-boot-kicker::before { content:""; display:inline-block; width:7px; height:7px;
    background:#19b88b; border-radius:50%; box-shadow:0 0 0 4px #e4f8ef; }
.rl-boot-mascot-stage {
    height:158px; width:192px; margin:22px auto 10px; position:relative;
}
.rl-boot-halo {
    position:absolute; width:150px; height:150px; left:21px; top:2px;
    border-radius:50%; background:radial-gradient(circle,#def8e9 0%,#edf9f2 64%,transparent 73%);
    animation:rl-boot-pulse 2.4s ease-in-out infinite;
}
.rl-boot-mascot {
    position:absolute; width:122px; height:100px; top:25px; left:35px;
    background:linear-gradient(140deg,#8be6bd 0%,#35bd91 78%,#23ae88 100%);
    border:3px solid #148269; border-radius:21px;
    box-shadow:0 12px 0 -5px #a9dcc5, 0 19px 26px rgba(8,112,83,.15);
    transform:rotate(-7deg); animation:rl-boot-bob 2.25s ease-in-out infinite;
}
.rl-boot-mascot::before {
    content:""; position:absolute; inset:8px; border:2px dashed rgba(255,255,255,.54);
    border-radius:14px; pointer-events:none;
}
.rl-boot-note-fold {
    position:absolute; top:-3px; right:-3px; width:31px; height:31px;
    background:linear-gradient(135deg,#e4fff1 45%,#a7eacb 100%);
    border-left:3px solid #148269; border-bottom:3px solid #148269;
    border-radius:0 18px 0 11px;
}
.rl-boot-eye {
    position:absolute; top:40px; width:10px; height:14px; background:#164d42;
    border-radius:50%; animation:rl-boot-blink 4s infinite;
}
.rl-boot-eye.left { left:35px; }
.rl-boot-eye.right { right:35px; }
.rl-boot-cheek { position:absolute; width:18px; height:9px; border-radius:50%;
    background:#ff96a7; opacity:.82; top:60px; }
.rl-boot-cheek.left { left:17px; }
.rl-boot-cheek.right { right:17px; }
.rl-boot-smile { position:absolute; width:24px; height:13px; left:calc(50% - 12px);
    top:55px; border-bottom:3px solid #164d42; border-radius:0 0 60% 60%; }
.rl-boot-arm {
    position:absolute; width:28px; height:24px; border-bottom:5px solid #168369;
    border-radius:0 0 25px 25px; top:61px; z-index:-1;
}
.rl-boot-arm.left { left:-23px; transform:rotate(31deg); }
.rl-boot-arm.right { right:-23px; transform-origin:left center;
    transform:rotate(-28deg); animation:rl-boot-wave 1.2s ease-in-out infinite alternate; }
.rl-boot-feet { position:absolute; width:17px; height:15px; bottom:-17px;
    background:#168369; border-radius:0 0 9px 9px; }
.rl-boot-feet.left { left:26px; } .rl-boot-feet.right { right:26px; }
.rl-boot-star { position:absolute; font-size:21px; color:#e7b544;
    animation:rl-boot-sparkle 1.8s ease-in-out infinite; }
.rl-boot-star.one { top:7px; right:14px; }
.rl-boot-star.two { bottom:13px; left:5px; font-size:16px; animation-delay:.7s; }
.rl-boot-star.three { top:29px; left:4px; font-size:11px; animation-delay:1.1s; }
.rl-boot-title { font-size:25px; font-weight:850; line-height:1.2; letter-spacing:-.04em;
    color:#183d37; margin:4px 0 9px; }
.rl-boot-title span { color:#0a9c77; }
.rl-boot-caption { font-size:14px; color:#597d74; margin-bottom:23px; }
.rl-boot-track { height:8px; width:100%; background:#e7f4ee; border-radius:100px;
    overflow:hidden; position:relative; }
.rl-boot-track::after { content:""; position:absolute; height:100%; left:-42%; width:42%;
    background:linear-gradient(90deg,#39c79b,#098a6c); border-radius:100px;
    animation:rl-boot-sweep 1.45s ease-in-out infinite; }
.rl-boot-small { margin-top:15px; color:#7f9b91; font-size:12px; }
@keyframes rl-boot-bob { 0%,100% { transform:translateY(0) rotate(-7deg); }
    50% { transform:translateY(-10px) rotate(-3deg); } }
@keyframes rl-boot-blink { 0%,43%,47%,100% { transform:scaleY(1); }
    45% { transform:scaleY(.12); } }
@keyframes rl-boot-pulse { 0%,100% { transform:scale(.97); opacity:.75; }
    50% { transform:scale(1.08); opacity:1; } }
@keyframes rl-boot-wave { from { transform:rotate(-12deg); }
    to { transform:rotate(-55deg); } }
@keyframes rl-boot-sparkle { 0%,100% { transform:scale(.8) rotate(-12deg); opacity:.45; }
    50% { transform:scale(1.18) rotate(12deg); opacity:1; } }
@keyframes rl-boot-sweep { from { left:-42%; } to { left:100%; } }
@keyframes rl-boot-fade { from { opacity:0; } to { opacity:1; } }
@media(max-width:500px) { .rl-boot-panel { padding:32px 20px 26px; }
    .rl-boot-title { font-size:22px; } }
@media(prefers-reduced-motion:reduce) {
    .rl-boot-screen,.rl-boot-halo,.rl-boot-mascot,.rl-boot-eye,
    .rl-boot-arm.right,.rl-boot-star,.rl-boot-track::after { animation:none!important; }
    .rl-boot-track::after { left:30%; }
}

/* Centered, animated scanning popup while inference is actually running. */
.rl-scan-overlay {
    position: fixed; inset: 0; z-index: 999999;
    display: flex; align-items: center; justify-content: center;
    background: rgba(13, 35, 42, .36); backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    animation: rl-overlay-in .22s ease-out both;
}
.rl-scan-dialog {
    width: min(390px, calc(100vw - 32px)); box-sizing:border-box;
    background:#ffffff; border:1px solid rgba(198,229,216,.9);
    border-radius:24px; padding:31px 30px 27px; text-align:center;
    box-shadow:0 24px 70px rgba(9,41,38,.23);
    animation:rl-popup-in .27s cubic-bezier(.2,.85,.3,1.1) both;
    font-family:inherit;
}
.rl-scan-emblem {
    width:86px; height:86px; margin:0 auto 19px; position:relative;
    display:grid; place-items:center; border-radius:50%;
    background:radial-gradient(circle, #ebf9f3 44%, #d7f4e6 100%);
}
.rl-scan-emblem::before, .rl-scan-emblem::after {
    content:""; position:absolute; inset:8px; border-radius:50%;
    border:2px solid #0a916c; border-right-color:transparent;
    animation:rl-spin 1.6s linear infinite;
}
.rl-scan-emblem::after {
    inset:18px; border:2px solid #7fd3af; border-bottom-color:transparent;
    animation:rl-spin 2s linear infinite reverse;
}
.rl-scan-center { font-size:18px; font-weight:850; color:#0a8262; letter-spacing:-.05em; }
.rl-scan-title { color:#173d37; font-size:22px; font-weight:800; letter-spacing:-.03em; }
.rl-scan-dots span { display:inline-block; animation:rl-dot-bounce 1.05s ease-in-out infinite; }
.rl-scan-dots span:nth-child(2) { animation-delay:.15s; }
.rl-scan-dots span:nth-child(3) { animation-delay:.30s; }
.rl-scan-subtitle { color:#68847d; font-size:13px; margin:10px 0 22px; line-height:1.5; }
.rl-scan-track { background:#e8f3ee; height:7px; border-radius:99px; overflow:hidden; position:relative; }
.rl-scan-track::after {
    content:""; position:absolute; left:-42%; top:0; bottom:0;
    width:42%; border-radius:99px;
    background:linear-gradient(90deg,#29b68c,#058965);
    animation:rl-scan-bar 1.5s ease-in-out infinite;
}
.rl-scan-foot { margin-top:12px; color:#89a199; font-size:11px; }
@keyframes rl-spin { to { transform:rotate(360deg); } }
@keyframes rl-dot-bounce { 0%,70%,100% { transform:translateY(0); opacity:.55; } 35% { transform:translateY(-4px); opacity:1; } }
@keyframes rl-scan-bar { from { left:-42%; } to { left:100%; } }
@keyframes rl-overlay-in { from { opacity:0; } to { opacity:1; } }
@keyframes rl-popup-in { from { transform:translateY(12px) scale(.96); opacity:0; } to { transform:translateY(0) scale(1); opacity:1; } }
@media (prefers-reduced-motion:reduce) {
    .rl-scan-overlay,.rl-scan-dialog,.rl-scan-emblem::before,.rl-scan-emblem::after,
    .rl-scan-dots span,.rl-scan-track::after { animation:none !important; }
    .rl-scan-track::after { left:25%; }
}

@media(max-width:900px) {
  .rl-header { flex-wrap:wrap; }
  .rl-brand { font-size:1.65rem; }
  .rl-result { min-height:0; }
  .block-container { padding-left:.85rem !important; padding-right:.85rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO
    trained_model = YOLO(model_path)
    # Run one CPU inference so that backend initialization happens while the
    # startup animation is visible, not after the upload dashboard appears.
    trained_model.predict(
        source=Image.new("RGB", (224, 224), "#eaf7f0"),
        imgsz=224, device="cpu", verbose=False,
    )
    return trained_model


def analyze_image(model, image: Image.Image) -> dict:
    """One image -> one class (never individual-note detection)."""
    result = model.predict(source=image, imgsz=224, device="cpu", verbose=False)[0]
    if result.probs is None:
        raise RuntimeError("Model did not return classification probabilities")

    scores = result.probs.data.detach().cpu().numpy()
    ranking = np.argsort(scores)[::-1]
    best_id, next_id = int(ranking[0]), int(ranking[1])
    best_score, next_score = float(scores[best_id]), float(scores[next_id])
    margin = best_score - next_score
    names = result.names

    def class_name(index: int) -> str:
        return str(names[index] if isinstance(names, dict) else names[index])

    prediction = class_name(best_id)
    return {
        "prediction": prediction,
        "confidence": best_score,
        "margin": margin,
        "accepted": prediction in SUPPORTED and best_score >= MIN_CONFIDENCE and margin >= MIN_MARGIN,
        "ranking": [(class_name(int(i)), float(scores[int(i)])) for i in ranking],
    }


def results_html(output: dict | None) -> str:
    if output is None:
        return """
<div class="rl-result">
  <span class="rl-status rl-status-good">READY TO ANALYZE</span>
  <div class="rl-prompt">
    <div class="rl-prompt-icon">◎</div>
    <div class="rl-prompt-head">Ready when you are</div>
    <div class="rl-prompt-copy">Upload or capture one Pakistani banknote on the left. Click Analyze Banknote to see the predicted denomination and confidence.</div>
  </div>
</div>"""

    accepted = bool(output["accepted"])
    prediction = str(output["prediction"])
    if accepted:
        status = '<span class="rl-status rl-status-good">✓ POSSIBLE NOTE IDENTIFIED</span>'
        title = f'<div class="rl-amount">Rs. {int(prediction):,}</div>'
        subtitle = '<div class="rl-result-caption">Predicted denomination</div>'
    else:
        status = '<span class="rl-status rl-status-uncertain">! UNCERTAIN RESULT</span>'
        title = '<div class="rl-unknown">Unable to identify</div>'
        subtitle = '<div class="rl-result-caption">Try a clearer picture of one supported note.</div>'

    ranked = []
    for index, (name, score) in enumerate(output["ranking"][:3]):
        pretty_name = f"Rs. {int(name):,}" if name in SUPPORTED else escape(name)
        ranked.append(
            f'<div class="rl-rank-line"><span>{pretty_name}</span><strong>{score * 100:.2f}%</strong></div>'
            f'<div class="rl-bar{" secondary" if index > 0 else ""}"><div style="width:{max(0.0, min(score * 100, 100.0)):.2f}%"></div></div>'
        )

    return f"""
<div class="rl-result">
  {status}
  {title}
  {subtitle}
  <div class="rl-stats">
    <div class="rl-stat"><div class="rl-stat-label">AI confidence score</div><div class="rl-stat-value">{output['confidence'] * 100:.2f}%</div></div>
    <div class="rl-stat"><div class="rl-stat-label">Lead over next class</div><div class="rl-stat-value">{output['margin'] * 100:.2f} pts</div></div>
  </div>
  <div class="rl-section">
    <div class="rl-section-title">Top AI predictions</div>
    {''.join(ranked)}
  </div>
</div>"""


# Only shown when the user clicks Analyze Banknote. This is intentionally
# plain HTML/CSS, not a separate UI component or any new model prediction.
ANALYZING_POPUP_HTML = """
<div class="rl-scan-overlay" role="status" aria-live="polite" aria-label="AI is analyzing your banknote">
  <div class="rl-scan-dialog">
    <div class="rl-scan-emblem"><span class="rl-scan-center">AI</span></div>
    <div class="rl-scan-title">AI is analyzing<span class="rl-scan-dots"><span>.</span><span>.</span><span>.</span></span></div>
    <div class="rl-scan-subtitle">Examining your banknote and checking its denomination</div>
    <div class="rl-scan-track"></div>
    <div class="rl-scan-foot">RupeeLens V2 · AI-powered recognition</div>
  </div>
</div>
"""


# Shown only once per browser session. The markup is pure HTML/CSS and
# therefore the animation keeps moving while Python loads/warmups the model.
STARTUP_SPLASH_HTML = """
<div class="rl-boot-screen" role="status" aria-live="polite" aria-label="Loading trained RupeeLens AI model">
  <div class="rl-boot-panel">
    <div class="rl-boot-kicker">Preparing your AI assistant</div>
    <div class="rl-boot-mascot-stage" aria-hidden="true">
      <div class="rl-boot-halo"></div>
      <div class="rl-boot-star one">✦</div>
      <div class="rl-boot-star two">✦</div>
      <div class="rl-boot-star three">✦</div>
      <div class="rl-boot-mascot">
        <div class="rl-boot-note-fold"></div>
        <div class="rl-boot-arm left"></div>
        <div class="rl-boot-arm right"></div>
        <div class="rl-boot-eye left"></div>
        <div class="rl-boot-eye right"></div>
        <div class="rl-boot-cheek left"></div>
        <div class="rl-boot-cheek right"></div>
        <div class="rl-boot-smile"></div>
        <div class="rl-boot-feet left"></div>
        <div class="rl-boot-feet right"></div>
      </div>
    </div>
    <div class="rl-boot-title">Waking up <span>RupeeLens AI</span></div>
    <div class="rl-boot-caption">Loading your trained banknote recognition model...</div>
    <div class="rl-boot-track"></div>
    <div class="rl-boot-small">Getting ready to recognize 6 Pakistani denominations</div>
  </div>
</div>
"""

st.markdown(
    """
<div class="rl-header">
  <div><div class="rl-brand">Rupee<span>Lens</span> V2</div>
       <div class="rl-subtitle">Pakistani currency recognition · AI demonstration</div></div>
  <div class="rl-pill"><span class="rl-dot"></span> 6 denominations · Single-note recognition</div>
</div>
""",
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(f"Model not found: `{MODEL_PATH}`. Keep the trained best.pt in its original folder.")
    st.stop()

# On initial entry, show the cute 2D startup animation during actual model
# loading. Later Streamlit reruns (uploads, button clicks) skip the splash.
first_visit = not st.session_state.get("rl_boot_complete", False)
startup_placeholder = st.empty() if first_visit else None
started_boot = time.monotonic()
if startup_placeholder is not None:
    startup_placeholder.markdown(STARTUP_SPLASH_HTML, unsafe_allow_html=True)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    if startup_placeholder is not None:
        startup_placeholder.empty()
    st.error(f"Unable to load model: {type(exc).__name__}: {exc}")
    st.stop()

if first_visit:
    remaining_boot = STARTUP_MIN_DISPLAY_SECONDS - (time.monotonic() - started_boot)
    if remaining_boot > 0:
        time.sleep(remaining_boot)
    startup_placeholder.empty()
    st.session_state["rl_boot_complete"] = True

left, right = st.columns([1.08, .92], gap="large")

with left:
    st.markdown('<div class="rl-panel-label">01 / Your photograph</div>', unsafe_allow_html=True)
    st.markdown('<div class="rl-panel-title">Scan a banknote</div>', unsafe_allow_html=True)
    source_method = st.radio(
        "Image source",
        ["Upload photo", "Use camera"],
        horizontal=True,
        label_visibility="collapsed",
        key="rl_source_method",
    )

    if source_method == "Upload photo":
        uploaded = st.file_uploader(
            "Choose one banknote image", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed", key="rl_photo_upload",
        )
    else:
        uploaded = st.camera_input("Take a photo of one banknote", key="rl_camera")

    image = None
    digest = None
    image_error = None
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            digest = sha256(raw).hexdigest()
            image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            image_error = f"Couldn't open this picture: {exc}"

    if image_error:
        st.error(image_error)

    # Reset results when an image is selected or removed.
    if st.session_state.get("rl_current_photo_digest") != digest:
        st.session_state["rl_current_photo_digest"] = digest
        st.session_state.pop("rl_analysis_output", None)
        st.session_state.pop("rl_analysis_error", None)
        st.session_state.pop("rl_analysis_digest", None)

    analyze_clicked = st.button(
        "Analyze Banknote",
        type="primary",
        use_container_width=True,
        disabled=image is None,
        key="rl_analyze_button",
    )

    if image is None:
        st.markdown(
            '<div class="rl-image-empty"><div class="rl-camera">▧</div>'
            '<strong>Image preview</strong><div>Choose a banknote photo to begin</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Predictable preview dimensions prevent tall images from making the page scroll.
        canvas = Image.new("RGB", (960, 450), "#eef3f4")
        preview = ImageOps.contain(image, (948, 438), Image.Resampling.LANCZOS)
        canvas.paste(preview, ((960 - preview.width) // 2, (450 - preview.height) // 2))
        st.image(canvas, use_container_width=True)

    st.markdown(
        '<div class="rl-upload-note">Tips: One visible note, good lighting, minimal blur. '
        'Rs. 75 and Rs. 5,000 are not supported.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="rl-panel-label">02 / AI analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="rl-panel-title">Recognition results</div>', unsafe_allow_html=True)

    output = None
    if analyze_clicked and image is not None and digest is not None:
        # The pop-up starts when the user clicks and stays while real inference
        # runs. Keep it visible for ~2.8 seconds if inference is faster.
        popup = st.empty()
        started = time.monotonic()
        popup.markdown(ANALYZING_POPUP_HTML, unsafe_allow_html=True)
        try:
            st.session_state["rl_analysis_output"] = analyze_image(model, image)
            st.session_state["rl_analysis_error"] = None
        except Exception as exc:
            st.session_state["rl_analysis_output"] = None
            st.session_state["rl_analysis_error"] = f"Analysis failed: {type(exc).__name__}: {exc}"
        finally:
            remaining = ANALYSIS_MIN_DISPLAY_SECONDS - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            popup.empty()
        st.session_state["rl_analysis_digest"] = digest

    # Keep results on reruns for this image only.
    if image is not None and st.session_state.get("rl_analysis_digest") == digest:
        output = st.session_state.get("rl_analysis_output")
        if st.session_state.get("rl_analysis_error"):
            st.error(st.session_state["rl_analysis_error"])

    st.markdown(results_html(output), unsafe_allow_html=True)

    st.markdown(
        '<div class="rl-bottom-note">Research prototype · A high score does not establish '
        'authenticity or guarantee a correct denomination. Unknown images may still be '
        'misidentified. Multiple-note counting is not available yet.</div>',
        unsafe_allow_html=True,
    )