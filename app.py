import os
os.chdir("/Users/anuragsharma/Documents/sraics360")

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
        return None, "No image uploaded."
    upsampler.scale = int(scale)
    img_np = np.array(image)
    try:
        output, _ = upsampler.enhance(img_np, outscale=int(scale))
    except RuntimeError as e:
        return None, f"Error: {str(e)}"
    result = Image.fromarray(output)
    orig_w, orig_h = image.size
    new_w, new_h = result.size
    info = f"Original: {orig_w}×{orig_h}px  →  Enhanced: {new_w}×{new_h}px  ({scale}× upscale)"
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
    padding: 0 1.5rem !important;
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
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -2px;
    line-height: 1;
    margin: 0;
}
.header-left h1 span { color: var(--accent); }
.header-left p {
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.4rem 0 0 0;
}
.header-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.3rem;
}
.header-right img {
    height: 60px;
    width: auto;
    object-fit: contain;
}
.header-right .inst-name {
    font-size: 0.62rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    text-align: right;
}
.badge-row {
    display: flex;
    gap: 0.4rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.badge {
    background: var(--ink);
    color: var(--cream);
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 1px;
    font-family: 'DM Mono', monospace;
}
.badge.red { background: var(--accent); }

/* ── TABS ── */
.tab-nav button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.9rem 1.2rem !important;
    border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── ENHANCE TAB LAYOUT FIX ── */
.enhance-wrap {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-top: 1.5rem;
    width: 100%;
}
.enhance-left, .enhance-right {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    min-width: 0;
}

/* ── INPUTS ── */
textarea, input[type="text"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 2px !important;
}
label span, .label-wrap span {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.66rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
input[type="range"] { accent-color: var(--accent) !important; }

/* ── BUTTONS ── */
button.primary {
    background: var(--ink) !important;
    color: var(--cream) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.7rem 1.5rem !important;
    transition: background 0.2s !important;
    width: 100% !important;
}
button.primary:hover { background: var(--accent) !important; }
button.secondary {
    background: transparent !important;
    color: var(--ink) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    width: 100% !important;
}
button.secondary:hover { border-color: var(--ink) !important; }

/* ── IMAGES ── */
.gr-image img {
    border-radius: 2px !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
    height: auto !important;
}

/* ── TEAM ── */
.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 1rem;
    margin-top: 1.25rem;
}
.team-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1.1rem 1.25rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.team-card:hover { border-color: var(--ink); box-shadow: 4px 4px 0 var(--ink); }
.team-card.leader { border-color: var(--accent); background: #fff8f5; }
.team-card.leader:hover { box-shadow: 4px 4px 0 var(--accent); }
.card-role { font-size: 0.6rem; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.3rem; }
.card-name { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--ink); margin-bottom: 0.15rem; }
.card-roll { font-size: 0.68rem; color: var(--muted); margin-bottom: 0.6rem; }
.card-desc { font-size: 0.72rem; color: var(--ink); line-height: 1.7; border-top: 1px solid var(--border); padding-top: 0.6rem; }

.course-strip {
    background: var(--ink); color: var(--cream);
    padding: 1rem 1.5rem; border-radius: 2px;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 0.5rem;
    margin-bottom: 1.25rem;
}
.course-strip span { font-size: 0.68rem; letter-spacing: 2px; text-transform: uppercase; }
.course-strip strong { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; }

/* ── MODEL TAB ── */
.model-hero {
    background: var(--ink); color: var(--cream);
    padding: 2.5rem 2rem; border-radius: 2px;
    margin-bottom: 1.25rem; position: relative; overflow: hidden;
}
.model-hero::before {
    content: 'GAN'; position: absolute; right: -10px; top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif; font-size: 7rem; font-weight: 800;
    color: rgba(255,255,255,0.04); pointer-events: none;
}
.model-hero h2 { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; margin: 0 0 0.5rem 0; color: var(--cream); }
.model-hero h2 span { color: #e8a87c; }
.model-hero p { font-size: 0.8rem; line-height: 1.8; color: #a09888; max-width: 600px; margin: 0; }

.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 2px; padding: 1rem; text-align: center; }
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--accent); line-height: 1; }
.stat-lbl { font-size: 0.62rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-top: 0.3rem; }

.section-title {
    font-size: 0.66rem; letter-spacing: 3px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 1rem; margin-top: 1.5rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}

.explainer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.explainer-card { background: var(--card); border: 1px solid var(--border); border-radius: 2px; padding: 1.25rem; }
.explainer-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
.explainer-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: var(--ink); margin-bottom: 0.4rem; }
.explainer-desc { font-size: 0.73rem; color: var(--muted); line-height: 1.75; }

.dark-note {
    background: var(--ink); color: #a09888;
    padding: 1.25rem 1.5rem; border-radius: 2px;
    font-size: 0.75rem; line-height: 1.8; margin-top: 1rem;
}
.dark-note strong { color: var(--cream); font-family: 'Syne', sans-serif; }

footer { display: none !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 2px; }
"""

# ── ESRGAN SVG DIAGRAM ──────────────────────────────────────────────────────────
esrgan_diagram = """
<svg viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:2px;margin-bottom:1rem;">
  <rect width="900" height="200" fill="#0f0e0c"/>

  <!-- Low Res -->
  <rect x="20" y="60" width="100" height="80" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="70" y="94" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="9">LOW-RES</text>
  <text x="70" y="108" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="8">Input Image</text>
  <rect x="35" y="120" width="70" height="12" rx="1" fill="#2a2a2a"/>
  <text x="70" y="130" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">64×64 px</text>

  <!-- Arrow -->
  <line x1="120" y1="100" x2="155" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Tile Split -->
  <rect x="155" y="70" width="80" height="60" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="195" y="97" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">TILE</text>
  <text x="195" y="110" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">SPLIT</text>
  <rect x="165" y="115" width="60" height="10" rx="1" fill="#2a2a2a"/>
  <text x="195" y="123" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">256×256 tiles</text>

  <line x1="235" y1="100" x2="270" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- RRDB Generator — main box -->
  <rect x="270" y="40" width="260" height="120" rx="2" fill="#1a1208" stroke="#c8502a" stroke-width="1.5"/>
  <text x="400" y="60" text-anchor="middle" fill="#e8a87c" font-family="monospace" font-size="9" font-weight="bold">GENERATOR (RRDB Network)</text>

  <!-- RRDB blocks inside -->
  <rect x="285" y="70" width="55" height="40" rx="2" fill="#c8502a" opacity="0.8"/>
  <text x="312" y="88" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">RRDB</text>
  <text x="312" y="100" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">Block 1</text>

  <line x1="340" y1="90" x2="355" y2="90" stroke="#c8502a" stroke-width="1" marker-end="url(#arr2)"/>

  <rect x="355" y="70" width="55" height="40" rx="2" fill="#c8502a" opacity="0.8"/>
  <text x="382" y="88" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">RRDB</text>
  <text x="382" y="100" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">Block 2</text>

  <text x="430" y="93" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="10">···</text>

  <rect x="445" y="70" width="55" height="40" rx="2" fill="#c8502a" opacity="0.8"/>
  <text x="472" y="88" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">RRDB</text>
  <text x="472" y="100" text-anchor="middle" fill="#fff" font-family="monospace" font-size="7">Block 6</text>

  <text x="400" y="145" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Residual-in-Residual Dense Blocks · learns textures &amp; details</text>

  <line x1="530" y1="100" x2="560" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Pixel Shuffle -->
  <rect x="560" y="70" width="80" height="60" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="600" y="94" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">PIXEL</text>
  <text x="600" y="107" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">SHUFFLE</text>
  <rect x="570" y="116" width="60" height="10" rx="1" fill="#2a2a2a"/>
  <text x="600" y="124" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">×4 upscale</text>

  <line x1="640" y1="100" x2="670" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Tile Merge -->
  <rect x="670" y="70" width="80" height="60" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="710" y="94" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">TILE</text>
  <text x="710" y="107" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">MERGE</text>
  <rect x="680" y="116" width="60" height="10" rx="1" fill="#2a2a2a"/>
  <text x="710" y="124" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">stitch tiles</text>

  <line x1="750" y1="100" x2="775" y2="100" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- High Res Output -->
  <rect x="775" y="60" width="105" height="80" rx="2" fill="#1a2a1a" stroke="#00cc7a" stroke-width="1.5"/>
  <text x="827" y="90" text-anchor="middle" fill="#00ff9d" font-family="monospace" font-size="9">HIGH-RES</text>
  <text x="827" y="103" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="8">Output Image</text>
  <rect x="790" y="113" width="75" height="12" rx="1" fill="#1a3a1a"/>
  <text x="827" y="123" text-anchor="middle" fill="#00cc7a" font-family="monospace" font-size="7">256×256 px</text>

  <!-- Arrow defs -->
  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#3a3a3a"/>
    </marker>
    <marker id="arr2" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/>
    </marker>
  </defs>
</svg>
"""

# ── GAN SVG DIAGRAM ─────────────────────────────────────────────────────────────
gan_diagram = """
<svg viewBox="0 0 860 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;border-radius:2px;margin-top:0.75rem;margin-bottom:1rem;">
  <rect width="860" height="180" fill="#0f0e0c"/>

  <!-- Noise/LR Input -->
  <rect x="20" y="55" width="110" height="70" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="75" y="84" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">LOW-RES</text>
  <text x="75" y="97" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">IMAGE</text>
  <text x="75" y="113" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">degraded input</text>

  <line x1="130" y1="90" x2="165" y2="90" stroke="#3a3a3a" stroke-width="1.5" marker-end="url(#a1)"/>

  <!-- Generator -->
  <rect x="165" y="40" width="130" height="100" rx="2" fill="#1a1208" stroke="#c8502a" stroke-width="1.5"/>
  <text x="230" y="65" text-anchor="middle" fill="#e8a87c" font-family="monospace" font-size="9" font-weight="bold">GENERATOR</text>
  <text x="230" y="82" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">RRDB Network</text>
  <text x="230" y="100" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Learns to create</text>
  <text x="230" y="113" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">realistic textures</text>
  <text x="230" y="126" text-anchor="middle" fill="#c8502a" font-family="monospace" font-size="7">↑ wants to fool disc.</text>

  <line x1="295" y1="90" x2="330" y2="90" stroke="#c8502a" stroke-width="1.5" marker-end="url(#a2)"/>

  <!-- Fake SR -->
  <rect x="330" y="55" width="100" height="70" rx="2" fill="#1a1a1a" stroke="#c8502a" stroke-width="1"/>
  <text x="380" y="82" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">FAKE</text>
  <text x="380" y="95" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">HIGH-RES</text>
  <text x="380" y="111" text-anchor="middle" fill="#c8502a" font-family="monospace" font-size="7">generated image</text>

  <!-- Real HR -->
  <rect x="330" y="135" width="100" height="38" rx="2" fill="#1a2a1a" stroke="#00cc7a" stroke-width="1"/>
  <text x="380" y="151" text-anchor="middle" fill="#00ff9d" font-family="monospace" font-size="8">REAL</text>
  <text x="380" y="164" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">HIGH-RES</text>

  <!-- Both go to discriminator -->
  <line x1="430" y1="90" x2="490" y2="110" stroke="#c8502a" stroke-width="1.2" marker-end="url(#a3)"/>
  <line x1="430" y1="154" x2="490" y2="130" stroke="#00cc7a" stroke-width="1.2" marker-end="url(#a4)"/>

  <!-- Discriminator -->
  <rect x="490" y="45" width="130" height="100" rx="2" fill="#0a1a1a" stroke="#00cc7a" stroke-width="1.5"/>
  <text x="555" y="68" text-anchor="middle" fill="#00ff9d" font-family="monospace" font-size="9" font-weight="bold">DISCRIMINATOR</text>
  <text x="555" y="85" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Is this image real</text>
  <text x="555" y="98" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">or AI-generated?</text>
  <text x="555" y="115" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Gives feedback to</text>
  <text x="555" y="128" text-anchor="middle" fill="#00cc7a" font-family="monospace" font-size="7">↑ train generator</text>

  <line x1="620" y1="95" x2="655" y2="95" stroke="#00cc7a" stroke-width="1.5" marker-end="url(#a4)"/>

  <!-- Output verdict -->
  <rect x="655" y="60" width="110" height="70" rx="2" fill="#1a1a1a" stroke="#3a3a3a" stroke-width="1"/>
  <text x="710" y="85" text-anchor="middle" fill="#f5f0e8" font-family="monospace" font-size="8">VERDICT</text>
  <text x="710" y="100" text-anchor="middle" fill="#c8502a" font-family="monospace" font-size="7">Real / Fake?</text>
  <text x="710" y="116" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">Loss used to update</text>
  <text x="710" y="127" text-anchor="middle" fill="#8a8070" font-family="monospace" font-size="7">both networks</text>

  <!-- Feedback loop arrow back -->
  <path d="M 710 130 Q 710 165 400 165 Q 230 165 230 140" stroke="#555" stroke-width="1" fill="none" stroke-dasharray="4,3" marker-end="url(#a5)"/>
  <text x="470" y="175" text-anchor="middle" fill="#555" font-family="monospace" font-size="7">training feedback loop</text>

  <defs>
    <marker id="a1" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3a3a3a"/></marker>
    <marker id="a2" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/></marker>
    <marker id="a3" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#c8502a"/></marker>
    <marker id="a4" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#00cc7a"/></marker>
    <marker id="a5" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker>
  </defs>
</svg>
"""

# ── UI ──────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="ArtUpscale — AI Super Resolution") as demo:

    gr.HTML("""
    <div class="site-header">
        <div class="header-left">
            <h1>Art<span>Upscale</span></h1>
            <p>AI-Powered Super Resolution · CS360 Artificial Intelligence · Semester 4</p>
            <div class="badge-row">
                <div class="badge red">Real-ESRGAN</div>
                <div class="badge">4× Upscale</div>
                <div class="badge">Artwork Optimized</div>
                <div class="badge">Deep Learning</div>
                <div class="badge">CS360</div>
            </div>
        </div>
        <div class="header-right">
            <img src="https://www.rgipt.ac.in/sites/default/files/styles/large/public/2023-01/rgipt_logo.png" 
                 onerror="this.style.display='none'"
                 alt="RGIPT Logo"/>
            <div class="inst-name">Rajiv Gandhi Institute<br>of Petroleum Technology</div>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):

        # ── TAB 1: ENHANCE ───────────────────────────────────────────────────
        with gr.Tab("✦  Enhance"):
            gr.HTML('<div class="enhance-wrap">', visible=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    input_image = gr.Image(type="pil", label="Upload Artwork", elem_classes=["gr-image"])
                    scale = gr.Slider(minimum=2, maximum=4, value=4, step=2, label="Upscale Factor")
                    with gr.Row():
                        enhance_btn = gr.Button("✦  Enhance Image", variant="primary")
                        clear_btn   = gr.Button("Clear", variant="secondary")
                    gr.HTML("""
                    <div style="padding:1rem;border:1px solid #d8d0c0;border-radius:2px;margin-top:0.5rem;">
                        <div style="font-size:0.66rem;letter-spacing:2px;text-transform:uppercase;color:#8a8070;margin-bottom:0.5rem;">Tips</div>
                        <div style="font-size:0.73rem;color:#0f0e0c;line-height:2;">
                            · Best with anime, paintings & illustrations<br>
                            · Smaller inputs process faster<br>
                            · 4× for maximum detail recovery<br>
                            · Processing takes ~10–30s on CPU
                        </div>
                    </div>
                    """)

                with gr.Column(scale=1, min_width=300):
                    output_image = gr.Image(type="pil", label="Enhanced Output", elem_classes=["gr-image"])
                    info_box = gr.Textbox(
                        label="Resolution Info",
                        interactive=False,
                        placeholder="Resolution details will appear after enhancement..."
                    )
                    gr.HTML("""
                    <div style="padding:1rem;background:#0f0e0c;border-radius:2px;margin-top:0.5rem;">
                        <div style="font-size:0.66rem;letter-spacing:2px;text-transform:uppercase;color:#555;margin-bottom:0.5rem;">How it works</div>
                        <div style="font-size:0.72rem;color:#a09888;line-height:1.9;">
                            Upload any low-res artwork → the Real-ESRGAN model analyses
                            pixel patterns and reconstructs fine detail using a deep neural
                            network trained on millions of images. Output is 2× or 4× the
                            original resolution with sharp edges and vivid colour.
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
                        Real-Enhanced Super-Resolution Generative Adversarial Network.
                        A deep learning model that doesn't just stretch pixels —
                        it <em>invents</em> the fine detail that was lost in compression or downsizing.
                        Think of it as an AI that has studied millions of high-resolution images
                        and learned exactly what sharp edges, fine textures, and crisp linework look like —
                        then applies that knowledge to reconstruct your blurry image from scratch.
                    </p>
                </div>

                <div class="stat-row">
                    <div class="stat-card"><div class="stat-val">4×</div><div class="stat-lbl">Max Upscale</div></div>
                    <div class="stat-card"><div class="stat-val">6B</div><div class="stat-lbl">RRDB Blocks</div></div>
                    <div class="stat-card"><div class="stat-val">17M</div><div class="stat-lbl">Parameters</div></div>
                    <div class="stat-card"><div class="stat-val">GAN</div><div class="stat-lbl">Architecture</div></div>
                </div>

                <div class="section-title">What is a GAN? — The Counterfeiter &amp; The Detective</div>
                {gan_diagram}
                <div class="explainer-grid">
                    <div class="explainer-card">
                        <div class="explainer-icon">🎨</div>
                        <div class="explainer-title">The Generator — The Forger</div>
                        <div class="explainer-desc">
                            Imagine a master art forger whose only job is to produce fake paintings
                            that look real. It takes a blurry input and tries to generate a crisp,
                            believable high-res version. It gets better every time the detective catches it.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🔍</div>
                        <div class="explainer-title">The Discriminator — The Detective</div>
                        <div class="explainer-desc">
                            A second network trained to spot fakes. It examines both real high-res images
                            and the generator's fakes, and tries to tell them apart.
                            Its feedback forces the generator to keep improving.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🔄</div>
                        <div class="explainer-title">Adversarial Training</div>
                        <div class="explainer-desc">
                            The two networks compete in a loop — the forger gets better at fooling,
                            the detective gets better at catching. After millions of rounds,
                            the generator produces images so convincing the discriminator can't tell them apart.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">✨</div>
                        <div class="explainer-title">Why Better Than Bicubic?</div>
                        <div class="explainer-desc">
                            Traditional upscaling (bicubic) averages nearby pixels — producing blurry results.
                            GANs learn a <em>distribution</em> of plausible sharp details,
                            so they confidently reconstruct crisp edges and textures rather than blurring them.
                        </div>
                    </div>
                </div>

                <div class="section-title">Real-ESRGAN Inference Pipeline</div>
                {esrgan_diagram}
                <div class="explainer-grid">
                    <div class="explainer-card">
                        <div class="explainer-icon">🧱</div>
                        <div class="explainer-title">RRDB Blocks</div>
                        <div class="explainer-desc">
                            Residual-in-Residual Dense Blocks are the core building unit.
                            Each block has dense skip connections that let the network learn
                            fine texture detail without losing the original image structure.
                            Think of them as layers of refinement stacked on top of each other.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🔲</div>
                        <div class="explainer-title">Tile-Based Processing</div>
                        <div class="explainer-desc">
                            Large images are split into 256×256 overlapping tiles,
                            each processed individually then stitched back together.
                            This lets the model run on a regular laptop without needing
                            a powerful GPU or large amounts of RAM.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">🎌</div>
                        <div class="explainer-title">Anime 6B Variant</div>
                        <div class="explainer-desc">
                            Fine-tuned specifically for anime and illustrated artwork.
                            Uses only 6 RRDB blocks (vs 23 in the full model) for faster inference,
                            while preserving clean linework, flat shading, and the vivid colour palette
                            typical of illustrated art.
                        </div>
                    </div>
                    <div class="explainer-card">
                        <div class="explainer-icon">📐</div>
                        <div class="explainer-title">Pixel Shuffle Upscaling</div>
                        <div class="explainer-desc">
                            Instead of resizing the image first (which causes blur),
                            the network works at the original resolution and rearranges
                            sub-pixel channels to produce the upscaled output at the very end —
                            preserving sharpness throughout.
                        </div>
                    </div>
                </div>

                <div class="dark-note">
                    <strong>The Key Insight</strong><br>
                    Traditional methods ask "what pixel value makes mathematical sense here?"
                    Real-ESRGAN asks "what would a human photographer actually have captured here?"
                    That shift — from interpolation to learned perception — is why the output
                    looks sharp and natural rather than smooth and artificial.
                </div>
            </div>
            """)

        # ── TAB 3: TEAM ─────────────────────────────────────────────────────
        with gr.Tab("◉  Our Team"):
            gr.HTML("""
            <div style="margin-top:1.5rem;">
                <div class="course-strip">
                    <div><span>Course</span><br><strong>CS360 · Artificial Intelligence</strong></div>
                    <div><span>Semester</span><br><strong>4th Semester</strong></div>
                    <div><span>Instructor</span><br><strong>Dr. Susham Biswas</strong></div>
                    <div><span>Institute</span><br><strong>RGIPT, Jais, Amethi</strong></div>
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

demo.launch()