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

def render(img, mask, prompt, negative="", guidance=1.0, strength=1.0, steps=6, seed=7, init=None, fill=None, fill_amt=1.0, blend=False):
    """The app's Inpainter.run in Python, plus experimental options."""
    rows = lcm[str(steps)]
    start = min(len(rows) - 1, int(round(len(rows) * (1.0 - strength))))
    rows = rows[start:]
    h = enc(prompt); hn = enc(negative) if guidance > 1.01 else None
    im = np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
    m = (np.asarray(mask, np.float32) / 255.0 >= 0.5).astype(np.float32)[None, None]
    if blend:
        cond = np.zeros_like(im); m_in = np.ones((1, 1, 64, 64), np.float32)
    else:
        cond = im * (m < 0.5)
        if fill is not None:
            cond = cond + (m >= 0.5) * fill_amt * (np.array(fill, np.float32).reshape(1, 3, 1, 1) / 127.5 - 1.0)
        m_in = m[:, :, ::8, ::8]
    lat_m = ve.run(None, {"image": cond.astype(np.float32)})[0]
    # latent-size mask, grown by one cell so edges blend
    m64 = m[:, :, ::8, ::8].copy()
    from scipy.ndimage import maximum_filter
    m64 = maximum_filter(m64, size=(1, 1, 3, 3))
    rng = np.random.RandomState(seed)
    noise = rng.randn(1, 4, 64, 64).astype(np.float32)
    z_orig = ve.run(None, {"image": im.astype(np.float32)})[0]
    z_init = z_orig if init is None else ve.run(None, {"image": (np.asarray(init, np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0).astype(np.float32)})[0]
    if start > 0:
        x = (np.sqrt(rows[0]["a"]) * z_init + np.sqrt(1 - rows[0]["a"]) * noise).astype(np.float32)
    else:
        x = noise
    if blend:
        x = m64 * x + (1 - m64) * (np.sqrt(rows[0]["a"]) * z_orig + np.sqrt(1 - rows[0]["a"]) * noise)
    for i, r in enumerate(rows):
        def eps(hh):
            return un.run(None, {"sample": np.concatenate([x, m_in, lat_m], 1).astype(np.float32), "timestep": np.array([r["t"]], np.float32), "hidden": hh})[0]
        e = eps(h)
        if hn is not None:
            en = eps(hn); e = en + guidance * (e - en)
        x0 = (x - np.sqrt(1 - r["a"]) * e) / np.sqrt(r["a"])
        den = r["c_out"] * x0 + r["c_skip"] * x
        last = i + 1 == len(rows)
        if last:
            x = den
            if blend: x = m64 * x + (1 - m64) * z_orig
        else:
            n2 = rng.randn(*x.shape).astype(np.float32)
            x = np.sqrt(r["a_prev"]) * den + np.sqrt(1 - r["a_prev"]) * n2
            if blend:
                x = m64 * x + (1 - m64) * (np.sqrt(r["a_prev"]) * z_orig + np.sqrt(1 - r["a_prev"]) * n2)
    out = vd.run(None, {"latent": x.astype(np.float32)})[0]
    gen = np.clip((out[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255)
    # put back only inside the area, like the app does
    mm = (np.asarray(mask, np.float32) / 255.0)[:, :, None]
    orig = np.asarray(img, np.float32)
    return Image.fromarray((orig * (1 - mm) + gen * mm).astype(np.uint8))

from PIL import ImageFilter
def recolor(img, mask, rgb):
    a = np.asarray(img, np.float32); m = np.asarray(mask, np.float32)[:, :, None] / 255.0
    L = a @ np.array([0.299, 0.587, 0.114], np.float32)
    mean = float((L * m[:, :, 0]).sum() / max(1.0, m.sum()))
    tl = float(np.dot(rgb, [0.299, 0.587, 0.114]))
    out = np.clip(np.array(rgb, np.float32)[None, None] + (L - mean)[:, :, None], 0, 255)
    return Image.fromarray((a * (1 - m) + out * m).astype(np.uint8))

blank = Image.new("RGB", (512, 512), (128, 128, 128)); full = Image.new("L", (512, 512), 255)
people = []
# 1) man in a blue t-shirt; area = blue pixels below the chin
man = render(blank, full, "photo of a young man standing in a living room, wearing a plain blue t-shirt, upper body, realistic", "", 1.0, 1.0, 8, 3)
arr = np.asarray(man).astype(np.int32); sel = arr[:, :, 2] > arr[:, :, 0] + 40; sel[:200] = False
people.append(("shirt", man, Image.fromarray((sel * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(5))))
# 2) woman in a black dress; area = dark pixels in the lower 60%
wom = render(blank, full, "photo of a woman standing in a living room, wearing a plain black knee length dress with long sleeves, full body, realistic", "", 1.0, 1.0, 8, 5)
arr = np.asarray(wom).astype(np.int32); L = arr.mean(2); sel = L < 60; sel[:int(512 * 0.38)] = False
people.append(("dress", wom, Image.fromarray((sel * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(11)).filter(ImageFilter.MinFilter(5))))
for n, im, mk in people: im.save(f"tune/{n}.png"); mk.save(f"tune/{n}_mask.png")
log("test people made")

SAFE = ", fully clothed, modest everyday clothing, family friendly, safe for work"
want = "a red knitted christmas sweater with a white snowflake pattern"
rgb = (190, 25, 35)
rows_out = []
for name, im, mk in people:
    p = want + SAFE
    pre = recolor(im, mk, rgb)
    cells = [("original", im), ("area", mk.convert("RGB"))]
    t0 = time.time(); cells.append(("A app now: pre-colour s0.9", render(im, mk, p, "", 1.0, 0.9, 8, 11, init=pre))); log(name, "A", round(time.time() - t0))
    t0 = time.time(); r1 = render(im, mk, p, "", 1.0, 1.0, 6, 11)
    cells.append(("B two-pass", render(r1, mk, p, "", 1.0, 0.9, 8, 11, init=recolor(r1, mk, rgb)))); log(name, "B", round(time.time() - t0))
    t0 = time.time(); cells.append(("C hide old, s1.0", render(im, mk, p, "", 1.0, 1.0, 6, 11, blend=True))); log(name, "C", round(time.time() - t0))
    t0 = time.time(); cells.append(("D hide old + pre-colour s0.95", render(im, mk, p, "", 1.0, 0.95, 8, 11, init=pre, blend=True))); log(name, "D", round(time.time() - t0))
    t0 = time.time(); cells.append(("E hide old, g2.0", render(im, mk, p, "blurry, deformed, plain", 2.0, 1.0, 6, 11, blend=True))); log(name, "E", round(time.time() - t0))
    rows_out.append(cells)

W = 300
grid = Image.new("RGB", (W * 7, (W + 22) * 2), (20, 20, 20)); d = ImageDraw.Draw(grid)
for r, cells in enumerate(rows_out):
    for i, (label, im) in enumerate(cells):
        grid.paste(im.resize((W, W)), (i * W, r * (W + 22))); d.text((i * W + 4, r * (W + 22) + W + 4), label, fill=(255, 255, 255))
grid.save("tune/grid.png"); log("grid saved")
