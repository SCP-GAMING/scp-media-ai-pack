"""Tries the outfit editor at several settings on a made-up test person and saves a labelled grid,
so the app's defaults can be picked by looking. Uses the published pack (no PyTorch needed)."""
import json, os, sys, time, urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from transformers import CLIPTokenizer

TAG = sys.argv[1] if len(sys.argv) > 1 else "pack-2"
D = "packdl"; os.makedirs(D, exist_ok=True); os.makedirs("tune", exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
for f in ["manifest.json", "text_encoder.onnx", "unet.onnx", "vae_encoder.onnx", "vae_decoder.onnx", "vocab.json", "merges.txt"]:
    if not os.path.exists(f"{D}/{f}"):
        urllib.request.urlretrieve(f"https://github.com/SCP-GAMING/scp-media-ai-pack/releases/download/{TAG}/{f}", f"{D}/{f}")
log("pack downloaded")
man = json.load(open(f"{D}/manifest.json"))
lcm = man["lcm"]
tok = CLIPTokenizer(f"{D}/vocab.json", f"{D}/merges.txt")
so = ort.SessionOptions()
S = lambda n: ort.InferenceSession(f"{D}/{n}", so, providers=["CPUExecutionProvider"])
te, ve, vd, un = S("text_encoder.onnx"), S("vae_encoder.onnx"), S("vae_decoder.onnx"), S("unet.onnx")
def enc(text):
    ids = np.array([tok(text, padding="max_length", max_length=77, truncation=True).input_ids], np.int64)
    return te.run(None, {"input_ids": ids})[0]

def render(img, mask, prompt, negative="", guidance=1.0, strength=1.0, steps=6, seed=7):
    """Exactly the app's Inpainter.run, in Python."""
    rows = lcm[str(steps)]
    start = min(len(rows) - 1, int(round(len(rows) * (1.0 - strength))))
    rows = rows[start:]
    h = enc(prompt); hn = enc(negative) if guidance > 1.01 else None
    im = np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
    m = (np.asarray(mask, np.float32) / 255.0 >= 0.5).astype(np.float32)[None, None]
    lat_m = ve.run(None, {"image": (im * (m < 0.5)).astype(np.float32)})[0]
    m64 = m[:, :, ::8, ::8]
    rng = np.random.RandomState(seed)
    noise = rng.randn(1, 4, 64, 64).astype(np.float32)
    if start > 0:
        z0 = ve.run(None, {"image": im.astype(np.float32)})[0]
        x = (np.sqrt(rows[0]["a"]) * z0 + np.sqrt(1 - rows[0]["a"]) * noise).astype(np.float32)
    else:
        x = noise
    for i, r in enumerate(rows):
        def eps(hh):
            return un.run(None, {"sample": np.concatenate([x, m64, lat_m], 1).astype(np.float32), "timestep": np.array([r["t"]], np.float32), "hidden": hh})[0]
        e = eps(h)
        if hn is not None:
            en = eps(hn); e = en + guidance * (e - en)
        x0 = (x - np.sqrt(1 - r["a"]) * e) / np.sqrt(r["a"])
        den = r["c_out"] * x0 + r["c_skip"] * x
        x = (np.sqrt(r["a_prev"]) * den + np.sqrt(1 - r["a_prev"]) * rng.randn(*x.shape).astype(np.float32)) if i + 1 < len(rows) else den
    out = vd.run(None, {"latent": x.astype(np.float32)})[0]
    return Image.fromarray(np.clip((out[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8))

# A made-up test person to edit (generated with the same model, whole picture painted).
blank = Image.new("RGB", (512, 512), (128, 128, 128)); full = Image.new("L", (512, 512), 255)
base = render(blank, full, "photo of a young man standing in a living room, wearing a plain black t-shirt and blue jeans, upper body, realistic",
              negative="blurry, deformed", guidance=2.0, strength=1.0, steps=8, seed=3)
base.save("tune/base.png"); log("base person made")
# The area to change: the shirt region, including where sleeves would go.
mask = Image.new("L", (512, 512), 0); ImageDraw.Draw(mask).rectangle((110, 250, 400, 512), fill=255)
mask.save("tune/mask.png")

SAFE_LONG = ", fully clothed, modest everyday clothing, family friendly, safe for work"
SAFE_SHORT = ", fully clothed"
NEG = "nude, naked, nsfw, nipples, cleavage, underwear, lingerie, bikini, see-through clothing, sexual, explicit, blurry, deformed"
want = "a red knitted christmas sweater"
cells = []
for label, suffix, g, s in [
    ("now: g1.0 s0.65 long", SAFE_LONG, 1.0, 0.65), ("g1.0 s0.85 short", SAFE_SHORT, 1.0, 0.85), ("g1.0 s1.0 short", SAFE_SHORT, 1.0, 1.0),
    ("g1.6 s0.65 short", SAFE_SHORT, 1.6, 0.65), ("g1.6 s0.85 short", SAFE_SHORT, 1.6, 0.85), ("g1.6 s1.0 short", SAFE_SHORT, 1.6, 1.0),
    ("g2.2 s0.65 short", SAFE_SHORT, 2.2, 0.65), ("g2.2 s0.85 short", SAFE_SHORT, 2.2, 0.85), ("g2.2 s1.0 short", SAFE_SHORT, 2.2, 1.0),
]:
    t0 = time.time()
    r = render(base, mask, want + suffix, NEG, g, s, steps=6, seed=11)
    cells.append((label, r)); log(label, round(time.time() - t0), "s")

W = 384
grid = Image.new("RGB", (W * 4, (W + 24) * 3), (20, 20, 20))
d = ImageDraw.Draw(grid)
grid.paste(base.resize((W, W)), (0, 0)); d.text((4, W + 4), "original", fill=(255, 255, 255))
grid.paste(mask.convert("RGB").resize((W, W)), (0, W + 24)); d.text((4, 2 * W + 28), "area to change", fill=(255, 255, 255))
for i, (label, im) in enumerate(cells):
    x = (1 + i % 3) * W; y = (i // 3) * (W + 24)
    grid.paste(im.resize((W, W)), (x, y)); d.text((x + 4, y + W + 4), label, fill=(255, 255, 255))
grid.save("tune/grid.png"); log("grid saved")
