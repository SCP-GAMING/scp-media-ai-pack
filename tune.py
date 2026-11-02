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
from scipy.ndimage import label as cc_label
def biggest(sel):
    lab, n = cc_label(sel)
    if n == 0: return sel
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()
people = []
wom = render(blank, full, "full body photo of a young woman standing in front of a white door, wearing a long sleeve emerald green knee length dress, realistic, natural light", "blurry, deformed", 2.0, 1.0, 8, 5)
arr = np.asarray(wom).astype(np.int32)
sel = biggest((arr[:, :, 1] > arr[:, :, 0] + 25) & (arr[:, :, 1] > arr[:, :, 2] + 10))
mk = Image.fromarray((sel * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(3))
mk_big = mk.filter(ImageFilter.MaxFilter(15))
wom.save("tune/dress.png"); mk.save("tune/dress_mask.png")
log("test person made, area", int(sel.sum()))

SAFE = ", fully clothed, modest everyday clothing, family friendly, safe for work"
NEG = "nude, naked, nsfw, nipples, cleavage, underwear, lingerie, bikini, see-through clothing, sexual, explicit, blurry, deformed, plain"
rows_out = []
for want in ["a santa claus costume", "a red christmas sweater and blue jeans"]:
    p = want + SAFE
    cells = [("original", wom), ("area", mk.convert("RGB"))]
    for label, kw in [
        ("A normal g2", dict(img=wom, mask=mk, negative=NEG, blend=False)),
        ("B normal g2 + 'not green'", dict(img=wom, mask=mk, negative=NEG + ", green, green dress", blend=False)),
        ("C normal g2 + not green, bigger area", dict(img=wom, mask=mk_big, negative=NEG + ", green, green dress", blend=False)),
        ("D hide old g2 (app now)", dict(img=wom, mask=mk, negative=NEG, blend=True)),
        ("E grey old + not green", dict(img=wom, mask=mk, negative=NEG + ", green, green dress", blend=False, grey=True)),
    ]:
        t0 = time.time()
        img = kw.pop("img"); mask = kw.pop("mask"); grey = kw.pop("grey", False)
        if grey:
            # show the AI the body's shape and shading inside the area, but in grey (no old colour)
            g = np.asarray(img.convert("L").convert("RGB"))
            m3 = (np.asarray(mask)[:, :, None] / 255.0)
            src = Image.fromarray((np.asarray(img) * (1 - m3) + g * m3).astype(np.uint8))
            r = render(src, mask, p, guidance=2.0, strength=0.92, steps=8, seed=11, **kw)
            r = Image.fromarray((np.asarray(img) * (1 - m3) + np.asarray(r) * m3).astype(np.uint8))
        else:
            r = render(img, mask, p, guidance=2.0, strength=1.0, steps=6, seed=11, **kw)
        cells.append((label, r)); log(want[:12], label, round(time.time() - t0))
    rows_out.append(cells)

W = 300
grid = Image.new("RGB", (W * 7, (W + 22) * 2), (20, 20, 20)); d = ImageDraw.Draw(grid)
for r, cells in enumerate(rows_out):
    for i, (label, im) in enumerate(cells):
        grid.paste(im.resize((W, W)), (i * W, r * (W + 22))); d.text((i * W + 4, r * (W + 22) + W + 4), label, fill=(255, 255, 255))
grid.save("tune/grid.png"); log("grid saved")
