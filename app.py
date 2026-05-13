import sys
import torchvision.transforms.functional as F

# Fix for basicsr error: redirect the old path to the new one
sys.modules['torchvision.transforms.functional_tensor'] = F

import os
import gradio as gr
# ... your other imports like from realesrgan import ...

import os

import gradio as gr
import numpy as np
import torch
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ── Load model ─────────────────────────────────────────────────────────────────
def load_upsampler():
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=6, num_grow_ch=32,
        scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus_anime_6B.pth",
        model=model,
        tile=256, tile_pad=10, pre_pad=0, half=False
    )
    return upsampler

print("Loading model...")
upsampler = load_upsampler()
print("Model ready ✅")

def enhance(image, scale):
    if image is None:
        return None, "⚠️  No image uploaded."
    # Ensure scale is valid (2 or 4)
    scale = int(scale) if scale in [2, 4] else 4
    # Resize large images for speed — cap at 256px on shortest side
    max_side = 256
    w, h = image.size
    if max(w, h) > max_side * 2:
        ratio = (max_side * 2) / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    upsampler.scale = scale
    img_np = np.array(image)
    try:
        output, _ = upsampler.enhance(img_np, outscale=scale)
    except RuntimeError as e:
        return None, f"Error: {str(e)}"
    result = Image.fromarray(output)
    orig_w, orig_h = image.size
    new_w, new_h = result.size
    info = f"✅  Original: {orig_w}×{orig_h}px  →  Enhanced: {new_w}×{new_h}px  ({scale}× upscale)"
    return result, info

# ── CSS ─────────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --cream:  #f5f0e8;
    --ink:    #0f0e0c;
    --accent: #c8502a;
    --muted:  #8a8070;
    --border: #d8d0c0;
    --card:   #faf7f2;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background-color: var(--cream) !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--ink) !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem 3rem !important;
}

/* ── HEADER ── */
.site-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 0;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.header-left h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    color: var(--ink); letter-spacing: -2px;
    -webkit-text-fill-color: var(--ink);
    line-height: 1; margin: 0;
}
.header-left h1 span {
    color: var(--accent);
    -webkit-text-fill-color: var(--accent);
}
.header-left p {
    font-size: 0.68rem; letter-spacing: 2px;
    text-transform: uppercase; color: var(--muted);
    -webkit-text-fill-color: var(--muted);
    margin: 0.4rem 0 0 0;
}
.header-right {
    display: flex; flex-direction: column;
    align-items: flex-end; gap: 0.4rem;
}
.header-right img {
    height: 64px; width: auto;
    object-fit: contain;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));
    display: none !important;
}
.header-right .inst-name {
    font-size: 0.6rem; letter-spacing: 1px;
    text-transform: uppercase; color: var(--muted); text-align: right;
    line-height: 1.5;
    display: none !important;
}
.badge-row { display: flex; gap: 0.4rem; margin-top: 0.75rem; flex-wrap: wrap; }
.badge {
    background: var(--ink); color: var(--cream);
    -webkit-text-fill-color: var(--cream);
    font-size: 0.6rem; letter-spacing: 2px;
    text-transform: uppercase; padding: 3px 9px;
    border-radius: 1px; font-family: 'DM Mono', monospace;
}
.badge.red { background: var(--accent); }

/* ── TABS ── */
.tab-nav button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: var(--muted) !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.9rem 1.2rem !important; border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── PROCESSING INDICATOR ── */
.processing-banner {
    display: none;
    background: var(--ink);
    color: var(--cream);
    padding: 0.75rem 1rem;
    border-radius: 2px;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-align: center;
    margin-bottom: 0.75rem;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── INPUTS ── */
textarea, input[type="text"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important; border-radius: 2px !important;
}
label, label span, .label-wrap span {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.66rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: var(--muted) !important;
}
input[type="range"] { accent-color: var(--accent) !important; }

/* ── BUTTONS ── */
button.primary {
    background: var(--ink) !important; color: var(--cream) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 2px !important; padding: 0.75rem 1.5rem !important;
    transition: background 0.2s !important; width: 100% !important;
}
button.primary:hover { background: var(--accent) !important; }
button.secondary {
    background: transparent !important; color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important; width: 100% !important;
}
button.secondary:hover { border-color: var(--ink) !important; }

/* ── IMAGE PANELS ── */
.gr-image img {
    border-radius: 2px !important;
    border: 1px solid var(--border) !important;
    width: 100% !important; height: auto !important;
}
.image-wrap {
    border: 1px solid var(--border);
    border-radius: 2px;
    overflow: hidden;
    background: #1a1a1a;
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.image-placeholder {
    color: var(--cream);
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    padding: 3rem;
}
.image-placeholder .icon { font-size: 2rem; margin-bottom: 0.75rem; }

/* ── TIPS BOX ── */
.tips-box {
    padding: 1rem; border: 1px solid var(--border);
    border-radius: 2px; margin-top: 0.5rem;
    background: var(--card) !important;
}
.tips-title {
    font-size: 0.64rem; letter-spacing: 2px;
    text-transform: uppercase; color: var(--muted) !important; margin-bottom: 0.5rem;
}
.tips-body { font-size: 0.73rem; color: var(--ink) !important; line-height: 2.1; background: var(--card) !important; }
.tip-chip {
    display: inline-block;
    background: var(--ink); color: var(--cream);
    font-size: 0.6rem; letter-spacing: 1px;
    padding: 2px 8px; border-radius: 1px; margin: 2px 4px 2px 0;
}

/* ── TEAM ── */
.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1rem; margin-top: 1.25rem;
}
.team-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 2px; padding: 1.1rem 1.25rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.team-card:hover { border-color: var(--ink); box-shadow: 4px 4px 0 var(--ink); }
.team-card.leader { border-color: var(--accent); background: #fff8f5; }
.team-card.leader:hover { box-shadow: 4px 4px 0 var(--accent); }
.card-role { font-size: 0.6rem; letter-spacing: 3px; text-transform: uppercase; color: var(--muted) !important; margin-bottom: 0.3rem; }
.card-name { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--ink) !important; margin-bottom: 0.15rem; }
.card-roll { font-size: 0.68rem; color: var(--muted) !important; margin-bottom: 0.6rem; }
.card-desc { font-size: 0.72rem; color: var(--ink) !important; line-height: 1.7; border-top: 1px solid var(--border); padding-top: 0.6rem; }

.course-strip {
    background: var(--ink); color: var(--cream);
    padding: 1rem 1.5rem; border-radius: 2px;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 0.5rem;
    margin-bottom: 1.25rem;
}
.course-strip span { font-size: 0.64rem; letter-spacing: 2px; text-transform: uppercase; opacity: 0.6; color: var(--cream) !important; }
.course-strip strong { font-family: 'Syne', sans-serif; font-size: 0.92rem; font-weight: 700; display: block; margin-top: 2px; color: var(--cream) !important; }

/* ── MODEL TAB ── */
.model-hero {
    background: var(--card);
    color: var(--ink);
    border: 1px solid var(--border);
    padding: 2rem 1.75rem; border-radius: 2px;
    margin-bottom: 1.25rem; position: relative; overflow: hidden;
}
.model-hero::before {
    content: 'GAN'; position: absolute; right: -10px; top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif; font-size: 7rem; font-weight: 800;
    color: rgba(15,14,12,0.05); pointer-events: none;
}
.model-hero h2 { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; margin: 0 0 0.5rem 0; color: var(--ink); }
.model-hero h2 span { color: #e8a87c; }
.model-hero p { font-size: 0.8rem; line-height: 1.8; color: var(--muted); max-width: 600px; margin: 0; }

.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 2px; padding: 1rem; text-align: center; }
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--accent); line-height: 1; }
.stat-lbl { font-size: 0.62rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-top: 0.3rem; }

.section-title {
    font-size: 0.66rem; letter-spacing: 3px; text-transform: uppercase;
    color: var(--muted); margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}
.explainer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.explainer-card { background: var(--card); border: 1px solid var(--border); border-radius: 2px; padding: 1.25rem; }
.explainer-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
.explainer-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: var(--ink) !important; margin-bottom: 0.4rem; }
.explainer-desc { font-size: 0.73rem; color: var(--muted) !important; line-height: 1.75; }
.dark-note {
    background: var(--card);
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 1.25rem 1.5rem; border-radius: 2px;
    font-size: 0.75rem; line-height: 1.8; margin-top: 1rem;
}
.dark-note strong { color: var(--ink) !important; font-family: 'Syne', sans-serif; }
.dark-note em {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    opacity: 1 !important;
    background: transparent !important;
}

/* render stability: avoid washed-out text during browser repaint/translation */
.team-card,
.team-card *,
.explainer-card,
.explainer-card *,
.dark-note,
.dark-note * {
    text-shadow: none !important;
}

/* ── HF SPACES AGGRESSIVE TEXT COLOR FIX ── */
/* HF Spaces forces light text (#fff or #fafafa), which disappears on light backgrounds */
/* This section uses maximum specificity and !important to force dark text everywhere */

body, html, .gradio-container {
    color: var(--ink) !important;
}

/* Team cards: explicitly force all text dark */
.team-card {
    color: var(--ink) !important;
}
.team-card h3,
.team-card h4,
.team-card p,
.team-card span,
.team-card div {
    color: var(--ink) !important;
}

.card-name, .card-role, .card-roll, .card-desc {
    color: var(--ink) !important;
}

/* Model/explainer cards */
.explainer-card,
.explainer-card h3,
.explainer-card h4,
.explainer-card p,
.explainer-card div,
.explainer-card span {
    color: var(--ink) !important;
}

.explainer-title {
    color: var(--ink) !important;
}

/* Header section: force ink color on everything */
.site-header {
    color: var(--ink) !important;
}

.header-left h1,
.header-left p {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    opacity: 1 !important;
}

.header-left h1 span {
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
}

.header-left p {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}

.badge,
.badge * {
    color: var(--cream) !important;
    -webkit-text-fill-color: var(--cream) !important;
    opacity: 1 !important;
}

.badge.red,
.badge.red * {
    color: #fff7ef !important;
    -webkit-text-fill-color: #fff7ef !important;
}

/* Keep key warning/info banners dark and readable */
.dark-banner,
.dark-banner * {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border-color: #2b2a27 !important;
}

.dark-banner strong {
    color: #e8a87c !important;
}

.dark-chip {
    display: inline-block;
    background: #1a1a1a !important;
    color: var(--cream) !important;
    border: 1px solid #343331 !important;
    padding: 2px 6px;
    border-radius: 1px;
    margin: 0 3px;
    font-size: 0.68rem;
}

/* Course strip: this should stay cream on dark background */
.course-strip,
.course-strip span,
.course-strip strong {
    color: var(--cream) !important;
}

/* Dark note section: force ink for strong, muted for normal */
.dark-note {
    color: var(--muted) !important;
}
.dark-note strong,
.dark-note h3,
.dark-note h4 {
    color: var(--ink) !important;
}

/* Stat cards and model hero */
.stat-card,
.stat-card *,
.model-hero,
.model-hero * {
    color: var(--ink) !important;
}

.stat-val {
    color: var(--accent) !important;
}

.model-hero h2 {
    color: var(--ink) !important;
}

.model-hero h2 span {
    color: #e8a87c !important;
}

/* ── GRADIO FORM ELEMENTS: FORCE DARK TEXT ── */
/* HF Spaces applies light blue (#4a9eff or similar) to all Gradio form components */
/* This section targets ALL form elements on the Enhance tab */

.label-wrap,
.label-wrap *,
.label-wrap span,
.label-wrap label {
    color: var(--ink) !important;
    background: var(--card) !important;
}

.gr-image,
.gr-image *,
.gr-image label,
.gr-image span,
.gr-image > div,
.gr-image > div > div,
.gr-image [data-testid="image"],
.gr-image [data-testid="image"] * {
    color: var(--ink) !important;
    background: var(--card) !important;
}

.gr-textbox,
.gr-textbox *,
.gr-textbox label {
    color: var(--ink) !important;
    background: var(--card) !important;
}

.gr-button,
.gr-button *,
.gr-button label {
    color: var(--ink) !important;
}

/* Tips and instructions boxes */
.tips-title,
.tips-body,
.tips-box,
.tips-box * {
    color: var(--ink) !important;
    background: var(--card) !important;
}

.tip-chip {
    background: var(--ink) !important;
    color: var(--cream) !important;
}

/* Processing indicator and info text */
.processing-banner,
.processing-banner *,
.info-text,
.info-text * {
    color: var(--ink) !important;
}

/* Form wrapper containers: ensure light backgrounds */
.gradio-container > div,
.gradio-container .form-box,
.gradio-container form {
    background: var(--card) !important;
}

/* ── SCOPED HF COLOR FIXES (avoid black-on-black) ── */
.gradio-container,
.gradio-container label,
.gradio-container input,
.gradio-container textarea {
    color: var(--ink) !important;
}

/* Keep form controls readable without overriding themed dark content blocks */
.gr-textbox,
.gr-number,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
}

input::placeholder,
textarea::placeholder {
    color: var(--muted) !important;
}

footer { display: none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 2px; }
"""

# ── SVG: ESRGAN Pipeline ────────────────────────────────────────────────────────
esrgan_diagram = """
<svg viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:2px;margin-bottom:1rem;">
  <rect width="900" height="200" fill="#0f0e0c"/>
  <rect x="20" y="60" width="100" height="80" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="70" y="94" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="9">LOW-RES</text>
  <text x="70" y="108" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="8">Input Image</text>
  <rect x="35" y="120" width="70" height="12" rx="1" fill="#2a2a2a"/>
  <text x="70" y="130" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">e.g. 64×64 px</text>
  <line x1="120" y1="100" x2="155" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="155" y="70" width="80" height="60" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="195" y="97" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">TILE</text>
  <text x="195" y="110" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">SPLIT</text>
  <rect x="165" y="116" width="60" height="10" rx="1" fill="#2a2a2a"/>
  <text x="195" y="124" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">256×256 tiles</text>
  <line x1="235" y1="100" x2="270" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="270" y="40" width="260" height="120" rx="2" fill="#1a1208" stroke="#c8502a" stroke-width="1.5"/>
  <text x="400" y="60" text-anchor="middle" fill="#e8a87c" font-family="monospace" font-size="9" font-weight="bold">GENERATOR (RRDB Network)</text>
  <rect x="285" y="70" width="55" height="40" rx="2" fill="#c8502a" opacity="0.85"/>
  <text x="312" y="88" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">RRDB</text>
  <text x="312" y="100" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">Block 1</text>
  <line x1="340" y1="90" x2="355" y2="90" stroke="#c8502a" stroke-width="1" marker-end="url(#arr2)"/>
  <rect x="355" y="70" width="55" height="40" rx="2" fill="#c8502a" opacity="0.85"/>
  <text x="382" y="88" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">RRDB</text>
  <text x="382" y="100" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">Block 2</text>
  <text x="430" y="93" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="10">···</text>
  <rect x="445" y="70" width="55" height="40" rx="2" fill="#c8502a" opacity="0.85"/>
  <text x="472" y="88" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">RRDB</text>
  <text x="472" y="100" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">Block 6</text>
  <text x="400" y="145" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Dense skip connections · learns textures, edges &amp; fine detail</text>
  <line x1="530" y1="100" x2="560" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="560" y="70" width="80" height="60" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="600" y="94" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">PIXEL</text>
  <text x="600" y="107" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">SHUFFLE</text>
  <rect x="570" y="116" width="60" height="10" rx="1" fill="#2a2a2a"/>
  <text x="600" y="124" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">×4 upscale</text>
  <line x1="640" y1="100" x2="670" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="670" y="70" width="80" height="60" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="710" y="94" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">TILE</text>
  <text x="710" y="107" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">MERGE</text>
  <rect x="680" y="116" width="60" height="10" rx="1" fill="#2a2a2a"/>
  <text x="710" y="124" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">stitch tiles</text>
  <line x1="750" y1="100" x2="775" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="775" y="60" width="105" height="80" rx="2" fill="#1a2a1a" stroke="#00cc7a" stroke-width="1.5"/>
  <text x="827" y="90" text-anchor="middle" fill="#00ff9d" font-family="monospace" font-size="9">HIGH-RES</text>
  <text x="827" y="103" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="8">Output Image</text>
  <rect x="790" y="114" width="75" height="12" rx="1" fill="#1a3a1a"/>
  <text x="827" y="123" text-anchor="middle" fill="#00cc7a" font-family="monospace" font-size="7">256×256 px</text>
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3a3a3a"/></marker>
    <marker id="arr2" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/></marker>
  </defs>
</svg>
"""

# ── SVG: GAN Diagram ────────────────────────────────────────────────────────────
gan_diagram = """
<svg viewBox="0 0 860 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:2px;margin:0.75rem 0 1rem 0;">
  <rect width="860" height="180" fill="#0f0e0c"/>
  <rect x="20" y="55" width="110" height="70" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="75" y="84" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">LOW-RES</text>
  <text x="75" y="97" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">IMAGE</text>
  <text x="75" y="113" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">degraded input</text>
  <line x1="130" y1="90" x2="165" y2="90" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#b1)"/>
  <rect x="165" y="40" width="130" height="100" rx="2" fill="#1a1208" stroke="#c8502a" stroke-width="1.5"/>
  <text x="230" y="65" text-anchor="middle" fill="#e8a87c" font-family="monospace" font-size="9" font-weight="bold">GENERATOR</text>
  <text x="230" y="82" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">RRDB Network</text>
  <text x="230" y="98" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Creates fake HR image</text>
  <text x="230" y="114" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Tries to fool detector</text>
  <text x="230" y="128" text-anchor="middle" fill="#c8502a" font-family="monospace" font-size="7">↑ improved via feedback</text>
  <line x1="295" y1="90" x2="330" y2="90" stroke="#c8502a" stroke-width="1.5" marker-end="url(#b2)"/>
  <rect x="330" y="50" width="100" height="55" rx="2" fill="#1a1a1a" stroke="#c8502a" stroke-width="1"/>
  <text x="380" y="74" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">FAKE HR</text>
  <text x="380" y="88" text-anchor="middle" fill="#c8502a" font-family="monospace" font-size="7">generated output</text>
  <rect x="330" y="125" width="100" height="40" rx="2" fill="#1a2a1a" stroke="#00cc7a" stroke-width="1"/>
  <text x="380" y="143" text-anchor="middle" fill="#00ff9d" font-family="monospace" font-size="8">REAL HR</text>
  <text x="380" y="157" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">ground truth</text>
  <line x1="430" y1="77" x2="490" y2="108" stroke="#c8502a" stroke-width="1.2" marker-end="url(#b3)"/>
  <line x1="430" y1="145" x2="490" y2="125" stroke="#00cc7a" stroke-width="1.2" marker-end="url(#b4)"/>
  <rect x="490" y="45" width="130" height="100" rx="2" fill="#0a1a1a" stroke="#00cc7a" stroke-width="1.5"/>
  <text x="555" y="68" text-anchor="middle" fill="#00ff9d" font-family="monospace" font-size="9" font-weight="bold">DISCRIMINATOR</text>
  <text x="555" y="85" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Is this real or fake?</text>
  <text x="555" y="101" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Scores both images</text>
  <text x="555" y="117" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Sends loss signal</text>
  <text x="555" y="133" text-anchor="middle" fill="#00cc7a" font-family="monospace" font-size="7">↑ trains generator</text>
  <line x1="620" y1="95" x2="655" y2="95" stroke="#00cc7a" stroke-width="1.5" marker-end="url(#b4)"/>
  <rect x="655" y="60" width="110" height="70" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="710" y="85" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">LOSS SCORE</text>
  <text x="710" y="100" text-anchor="middle" fill="#c8502a" font-family="monospace" font-size="7">Real / Fake verdict</text>
  <text x="710" y="116" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Updates both networks</text>
  <path d="M 710 130 Q 710 165 400 165 Q 230 165 230 140" stroke="#444" stroke-width="1" fill="none" stroke-dasharray="4,3" marker-end="url(#b5)"/>
  <text x="460" y="175" text-anchor="middle" fill="#444" font-family="monospace" font-size="7">training feedback loop — repeated millions of times</text>
  <defs>
    <marker id="b1" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3a3a3a"/></marker>
    <marker id="b2" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/></marker>
    <marker id="b3" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/></marker>
    <marker id="b4" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#00cc7a"/></marker>
    <marker id="b5" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#444"/></marker>
  </defs>
</svg>
"""

# ── SVG: Loss Function Diagram ──────────────────────────────────────────────────
loss_diagram = """
<svg viewBox="0 0 860 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:2px;margin:0.75rem 0 1rem 0;">
  <rect width="860" height="160" fill="#0f0e0c"/>
  <!-- L_total -->
  <rect x="20" y="50" width="120" height="60" rx="2" fill="#1a1208" stroke="#c8502a" stroke-width="1.5"/>
  <text x="80" y="75" text-anchor="middle" fill="#e8a87c" font-family="monospace" font-size="9" font-weight="bold">Total Loss</text>
  <text x="80" y="92" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">L = Lp + λLadv + ηLperc</text>
  <line x1="140" y1="80" x2="180" y2="55" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="140" y1="80" x2="180" y2="80" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <line x1="140" y1="80" x2="180" y2="108" stroke="#555" stroke-width="1" stroke-dasharray="3,2"/>
  <!-- Pixel Loss -->
  <rect x="180" y="30" width="150" height="50" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="255" y="52" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">Pixel Loss (L1 / MSE)</text>
  <text x="255" y="67" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">pixel-by-pixel difference</text>
  <!-- Adversarial Loss -->
  <rect x="180" y="62" width="150" height="50" rx="2" fill="#1a1a1a" stroke="#c8502a" stroke-width="1"/>
  <text x="255" y="83" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">Adversarial Loss</text>
  <text x="255" y="98" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">fool the discriminator</text>
  <!-- Perceptual Loss -->
  <rect x="180" y="95" width="150" height="50" rx="2" fill="#1a2a1a" stroke="#00cc7a" stroke-width="1"/>
  <text x="255" y="116" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">Perceptual Loss (VGG)</text>
  <text x="255" y="131" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">feature-level similarity</text>
  <!-- Arrows to effect -->
  <line x1="330" y1="55" x2="370" y2="55" stroke="#3a3a3a" stroke-width="1" marker-end="url(#c1)"/>
  <line x1="330" y1="87" x2="370" y2="87" stroke="#c8502a" stroke-width="1" marker-end="url(#c2)"/>
  <line x1="330" y1="120" x2="370" y2="120" stroke="#00cc7a" stroke-width="1" marker-end="url(#c3)"/>
  <!-- Effect boxes -->
  <rect x="370" y="30" width="200" height="50" rx="2" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
  <text x="470" y="52" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="7">Ensures structural fidelity</text>
  <text x="470" y="66" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Prevents totally wrong outputs</text>
  <rect x="370" y="62" width="200" height="50" rx="2" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
  <text x="470" y="83" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="7">Produces sharp, realistic textures</text>
  <text x="470" y="97" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Prevents blurry outputs</text>
  <rect x="370" y="95" width="200" height="50" rx="2" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
  <text x="470" y="116" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="7">Matches high-level features</text>
  <text x="470" y="130" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Preserves semantic content</text>
  <!-- Combined result -->
  <line x1="570" y1="87" x2="610" y2="87" stroke="#555" stroke-width="1" marker-end="url(#c4)"/>
  <rect x="610" y="55" width="230" height="65" rx="2" fill="#1a1208" stroke="#c8502a" stroke-width="1.5"/>
  <text x="725" y="78" text-anchor="middle" fill="#e8a87c" font-family="monospace" font-size="8" font-weight="bold">Result: Perceptually Sharp</text>
  <text x="725" y="94" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Structurally correct + photorealistic</text>
  <text x="725" y="108" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">texture + semantically meaningful</text>
  <defs>
    <marker id="c1" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3a3a3a"/></marker>
    <marker id="c2" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/></marker>
    <marker id="c3" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#00cc7a"/></marker>
    <marker id="c4" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker>
  </defs>
</svg>
"""

# ── UI ──────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="ArtUpscale — AI Super Resolution") as demo:

    gr.HTML("""
    <div class="site-header notranslate" translate="no">
        <div class="header-left">
            <h1 class="notranslate" translate="no">Art<span>Upscale</span></h1>
            <p class="notranslate" translate="no">AI-Powered Super Resolution · CS360 Artificial Intelligence · Semester 4</p>
            <div class="badge-row notranslate" translate="no">
                <div class="badge red">Real-ESRGAN</div>
                <div class="badge">4× Upscale</div>
                <div class="badge">Artwork Optimized</div>
                <div class="badge">Deep Learning</div>
                <div class="badge">CS360</div>
            </div>
        </div>
        <div class="header-right">
            <img src="https://www.rgipt.ac.in/images/logo.png"
                 alt="RGIPT Logo"
                 onerror="this.style.display='none'"/>
            <div class="inst-name">Rajiv Gandhi Institute<br>of Petroleum Technology<br>Jais, Amethi</div>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):

        # ── TAB 1: ENHANCE ───────────────────────────────────────────────────
        with gr.Tab("✦  Enhance"):

            gr.HTML("""
            <div class="dark-banner" style="padding:0.75rem 1rem;
                        border-radius:2px;font-size:0.72rem;letter-spacing:1px;
                        margin:1rem 0 0.75rem 0;line-height:1.7;">
                <strong>⚡ For fast demo results</strong> — use small artwork images (under 300×300px).
                Try anime thumbnails, pixel art, game sprites, or low-res manga panels.
                Processing time on CPU: <strong>~8–15s for small images, ~30–60s for large ones.</strong>
                <br>Good test sources:
                <span class="dark-chip">Google Images → Tools → Size → Small</span>
                <span class="dark-chip">Pinterest low-res art</span>
                <span class="dark-chip">Game sprite sheets</span>
            </div>
            """)

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    input_image = gr.Image(
                        type="pil", label="Upload Artwork",
                        elem_classes=["gr-image"],
                        height=280
                    )
                    scale = gr.Slider(
                        minimum=4, maximum=4, value=4, step=1,
                        label="Upscale Factor",
                        visible=False
                    )
                    enhance_btn = gr.Button("✦  Enhance Image", variant="primary")
                    clear_btn   = gr.Button("⌫  Clear", variant="secondary")

                    gr.HTML("""
                    <div class="tips-box" style="margin-top:0.75rem;background:#faf7f2;border:1px solid #d8d0c0;color:#0f0e0c;">
                        <div class="tips-title">Tips for best results</div>
                        <div class="tips-body">
                            · Anime, manga &amp; illustrations → best output<br>
                            · Pixel art → use 4× for crisp upscale<br>
                            · Photos work too but are slower<br>
                            · Start with 4× — it is the supported upscale<br>
                            · The queue timer is normal — it's processing!
                        </div>
                        <div style="margin-top:0.75rem;">
                            <div class="tips-title">Good test images</div>
                            <span class="tip-chip">anime thumbnail</span>
                            <span class="tip-chip">pixel art sprite</span>
                            <span class="tip-chip">manga panel</span>
                            <span class="tip-chip">game screenshot</span>
                            <span class="tip-chip">low-res painting</span>
                        </div>
                    </div>
                    """)

                with gr.Column(scale=1, min_width=320):
                    gr.HTML("""
                    <div id="processing-hint" class="dark-banner" style="
                        border:1px solid #c8502a;padding:0.6rem 1rem;border-radius:2px;
                        font-size:0.7rem;letter-spacing:1px;margin-bottom:0.5rem;
                        text-align:center;">
                        ⏳ Processing takes 10–60s on CPU · The queue timer below the image is normal · Please wait
                    </div>
                    """)
                    output_image = gr.Image(
                        type="pil", label="Enhanced Output",
                        elem_classes=["gr-image"],
                        height=280
                    )
                    info_box = gr.Textbox(
                        label="Resolution Info",
                        interactive=False,
                        placeholder="Enhancement details will appear here..."
                    )
                    gr.HTML("""
                    <div style="padding:1rem;background:#faf7f2;border:1px solid #d8d0c0;border-radius:2px;margin-top:0.5rem;">
                        <div style="font-size:0.64rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.5rem;">What just happened?</div>
                        <div style="font-size:0.72rem;color:#0f0e0c;line-height:1.9;">
                            The model analysed every 256×256 tile of your image,
                            ran it through 6 RRDB neural network blocks,
                            and reconstructed fine detail at up to 4× the original resolution —
                            synthesizing sharp edges and textures that weren't in the original.
                        </div>
                    </div>
                    """)

            enhance_btn.click(fn=enhance, inputs=[input_image, scale], outputs=[output_image, info_box])
            clear_btn.click(fn=lambda: (None, None, ""), outputs=[input_image, output_image, info_box])

        # ── TAB 2: THE MODEL ─────────────────────────────────────────────────
        with gr.Tab("◈  The Model"):
            gr.HTML(f"""
            <div style="margin-top:1.5rem;">

                <div class="model-hero">
                    <h2>Real-<span>ESRGAN</span></h2>
                    <p>
                        Real-Enhanced Super-Resolution Generative Adversarial Network —
                        a deep learning model that doesn't just stretch pixels,
                        it <em>invents</em> the fine detail that was lost in compression or downsizing.
                        Think of it as an AI that has studied millions of high-resolution images
                        and learned exactly what sharp edges and crisp textures look like —
                        then applies that knowledge to reconstruct your blurry image from scratch.
                    </p>
                </div>

                <div class="stat-row">
                    <div class="stat-card"><div class="stat-val">4×</div><div class="stat-lbl">Max Upscale</div></div>
                    <div class="stat-card"><div class="stat-val">6B</div><div class="stat-lbl">RRDB Blocks</div></div>
                    <div class="stat-card"><div class="stat-val">17M</div><div class="stat-lbl">Parameters</div></div>
                    <div class="stat-card"><div class="stat-val">3</div><div class="stat-lbl">Loss Functions</div></div>
                </div>

                <div class="section-title">What is a GAN? — The Forger &amp; The Detective</div>
                {gan_diagram}
                <div class="explainer-grid">
                    <div class="explainer-card">
                        <div class="explainer-icon">🎨</div>
                        <div class="explainer-title">Generator — The Art Forger</div>
                        <div class="explainer-desc">
                            Takes a blurry, low-res input and tries to produce a crisp,
                            photorealistic high-res version. It starts terrible and gets better
                            every time the discriminator catches it out. Its job is to make fakes
                            so good that even an expert can't tell them apart from the real thing.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🔍</div>
                        <div class="explainer-title">Discriminator — The Detective</div>
                        <div class="explainer-desc">
                            A second network trained purely to spot fakes. It sees both real
                            high-res images and the generator's outputs, and tries to label each one.
                            Its feedback is what drives the generator to keep improving — without it,
                            the generator would just produce blurry averages.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🔄</div>
                        <div class="explainer-title">Adversarial Training Loop</div>
                        <div class="explainer-desc">
                            The two networks compete millions of times. The forger gets better at
                            fooling, the detective gets better at catching. After enough rounds,
                            the generator has learned to produce images so convincing the
                            discriminator genuinely can't tell what's real.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">✨</div>
                        <div class="explainer-title">Why Not Just Bicubic?</div>
                        <div class="explainer-desc">
                            Bicubic interpolation averages nearby pixels — it makes mathematical sense
                            but looks blurry. GANs instead ask "what would a real high-res image look like
                            here?" and synthesise convincing detail. The difference is
                            <em>interpolation</em> vs <em>learned perception</em>.
                        </div>
                    </div>
                </div>

                <div class="section-title">ESRGAN Architecture &amp; Pipeline</div>
                {esrgan_diagram}
                <div class="explainer-grid">
                    <div class="explainer-card">
                        <div class="explainer-icon">🧱</div>
                        <div class="explainer-title">RRDB — Residual-in-Residual Dense Blocks</div>
                        <div class="explainer-desc">
                            The core building unit. Each RRDB contains dense connections where every layer
                            receives input from all previous layers — letting gradients flow freely
                            during training and allowing the network to learn extremely fine texture patterns.
                            Stacking 6 of these creates a deep feature extractor without gradient degradation.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🔲</div>
                        <div class="explainer-title">Tile-Based Inference</div>
                        <div class="explainer-desc">
                            Images are split into overlapping 256×256 tiles processed individually,
                            then stitched back. The overlap (tile padding) prevents visible seams.
                            This lets the model run on consumer hardware — no GPU needed —
                            and makes RAM usage predictable regardless of image size.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">📐</div>
                        <div class="explainer-title">Pixel Shuffle Upscaling</div>
                        <div class="explainer-desc">
                            Rather than resizing first (causing blur), the network works at the original
                            resolution throughout. At the very end, a pixel-shuffle layer rearranges
                            learned sub-pixel channels into the final upscaled output —
                            preserving all the sharpness built up through the RRDB blocks.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🎌</div>
                        <div class="explainer-title">Anime 6B Variant</div>
                        <div class="explainer-desc">
                            Fine-tuned on anime and illustrated art using only 6 RRDB blocks
                            (vs 23 in the full model) for faster CPU inference.
                            It's optimised for clean linework, flat colour shading,
                            and the vibrant palette of illustrated content — exactly what we use here.
                        </div>
                    </div>
                </div>

                <div class="section-title">Loss Functions — How the Model Learns</div>
                {loss_diagram}
                <div class="explainer-grid">
                    <div class="explainer-card">
                        <div class="explainer-icon">📏</div>
                        <div class="explainer-title">Pixel Loss (L1 / MSE)</div>
                        <div class="explainer-desc">
                            Compares the output image to the real high-res image pixel by pixel.
                            Simple but essential — it ensures the overall structure and colour
                            are correct. On its own it produces blurry results because it
                            averages all possible sharp outputs into one smooth prediction.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">⚔️</div>
                        <div class="explainer-title">Adversarial Loss</div>
                        <div class="explainer-desc">
                            Measures how well the generator fools the discriminator.
                            This is the engine of sharpness — it pushes the generator to produce
                            outputs that look statistically indistinguishable from real high-res images,
                            forcing it to synthesise convincing textures and edges.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🧠</div>
                        <div class="explainer-title">Perceptual Loss (VGG Features)</div>
                        <div class="explainer-desc">
                            Uses a pre-trained VGG network to compare images at a feature level
                            rather than pixel level. Instead of asking "are these pixels identical?",
                            it asks "do these images contain the same objects and structures?"
                            This preserves semantic meaning even when pixel values differ.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">⚖️</div>
                        <div class="explainer-title">Combined Loss = Best of All Three</div>
                        <div class="explainer-desc">
                            The final loss is a weighted sum: pixel loss keeps structure correct,
                            adversarial loss adds sharpness and realism, perceptual loss preserves
                            semantic content. Together they prevent the three failure modes:
                            structural wrong output, blurriness, and semantic distortion.
                        </div>
                    </div>
                </div>

                <div class="dark-note">
                    <strong>The Key Insight</strong><br>
                    Traditional methods ask <em>"what pixel value makes mathematical sense here?"</em>
                    Real-ESRGAN asks <em>"what would a human photographer actually have captured here?"</em>
                    That shift — from interpolation to learned perception — is why the output looks
                    sharp and natural rather than smooth and artificial.
                    The three-loss training framework ensures the model is simultaneously accurate,
                    sharp, and semantically meaningful.
                </div>
            </div>
            """)

        # ── TAB 3: TEAM ─────────────────────────────────────────────────────
        with gr.Tab("◉  Our Team"):
            gr.HTML("""
            <div style="margin-top:1.5rem;">
                <div class="course-strip">
                    <div><span>Course</span><strong>CS360 · Artificial Intelligence</strong></div>
                    <div><span>Semester</span><strong>4th Semester · 2025–26</strong></div>
                    <div><span>Instructor</span><strong>Dr. Susham Biswas</strong></div>
                    <div><span>Institute</span><strong>RGIPT, Jais, Amethi</strong></div>
                </div>
                <div class="team-grid">
                    <div class="team-card leader">
                        <div class="card-role">⭐ Team Leader</div>
                        <div class="card-name">Anurag Sharma</div>
                        <div class="card-roll">24MC3008</div>
                        <div class="card-desc">Project architecture, system integration, environment setup, and final submission. Coordinates all modules into a cohesive working application.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">Model Research</div>
                        <div class="card-name">Nityansh Pant</div>
                        <div class="card-roll">24MC3033</div>
                        <div class="card-desc">Deep-dives into Real-ESRGAN architecture, RRDB block design, and GAN theory. Responsible for literature review and model selection rationale.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">UI / UX Developer</div>
                        <div class="card-name">Vaibhav</div>
                        <div class="card-roll">24MC3059</div>
                        <div class="card-desc">Designs and implements the Gradio interface — layout, custom CSS theming, tab structure, and overall visual experience of the application.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">Image Preprocessing</div>
                        <div class="card-name">Shubhayu Brahmachari</div>
                        <div class="card-roll">24MC3046</div>
                        <div class="card-desc">Handles input validation, format conversion, colour space management, and post-processing of enhanced outputs before display.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">Inference Engine</div>
                        <div class="card-name">Himanshu Sachdeva</div>
                        <div class="card-roll">24MC3021</div>
                        <div class="card-desc">Implements model loading, tiling strategy, upsampler configuration, and CPU inference pipeline for efficient processing.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">Testing & Evaluation</div>
                        <div class="card-name">Arjit Anand</div>
                        <div class="card-roll">24MC3009</div>
                        <div class="card-desc">Conducts before/after quality comparisons, documents metrics, and validates model performance across different artwork styles.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">Documentation</div>
                        <div class="card-name">Taksh Agarwal</div>
                        <div class="card-roll">24MC3051</div>
                        <div class="card-desc">Writes the README, inline code documentation, report structure, and technical write-up covering methodology and results.</div>
                    </div>
                    <div class="team-card">
                        <div class="card-role">Demo & Presentation</div>
                        <div class="card-name">Utkarsh Dixit</div>
                        <div class="card-roll">24MC3057</div>
                        <div class="card-desc">Records the video demonstration, coordinates individual narration segments, and prepares the final presentation for submission.</div>
                    </div>
                </div>
            </div>
            """)

demo.launch(css=css)
