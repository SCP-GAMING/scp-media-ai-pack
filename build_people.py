"""
Builds the SCP Media "People pack": find faces, tell people apart, and turn speech into text, all on the phone.

  * YuNet face detector (OpenCV Zoo, MIT)
  * SFace face recogniser (OpenCV Zoo, Apache-2.0)
  * Whisper base speech recogniser (OpenAI, MIT), exported to ONNX with the sound-to-spectrogram step built in

Every step the phone app does is re-done here in plain Python/numpy (the same maths the Kotlin code uses) and
checked against the reference (OpenCV / Hugging Face), so the app's code is known to be right before shipping.
"""
import hashlib, json, math, os, sys, time, urllib.request
import numpy as np

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
OUT = "people"; TMP = "ptmp"
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)


def fetch(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "scp-media-pack-builder"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
RAW = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"

# ----------------------------------------------------------------------------------------------- faces
def faces():
    import cv2
    import onnxruntime as ort
    yunet = fetch(ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx", f"{TMP}/yunet.onnx")
    sface = fetch(ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx", f"{TMP}/sface.onnx")
    sface8 = fetch(ZOO + "face_recognition_sface/face_recognition_sface_2021dec_int8.onnx", f"{TMP}/sface8.onnx")
    for name, folder in [("yunet", "face_detection_yunet"), ("sface", "face_recognition_sface")]:
        try:
            lic = fetch(RAW + folder + "/LICENSE", f"{OUT}/LICENSE-{name}.txt")
            log(name, "licence:", open(lic, errors="ignore").read(300).replace("\n", " ")[:200])
        except Exception as e:
            log(name, "licence file not found:", e)
    so = ort.SessionOptions()
    det = ort.InferenceSession(yunet, so, providers=["CPUExecutionProvider"])
    log("yunet inputs", [(i.name, i.shape) for i in det.get_inputs()], "outputs", [(o.name, o.shape) for o in det.get_outputs()])

    imgs = []
    for nm in ["lena.jpg", "messi5.jpg"]:
        p = fetch("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/" + nm, f"{TMP}/{nm}")
        imgs.append((nm, cv2.imread(p)))

    S = 640  # the app always feeds a 640x640 picture (letterboxed)

    def letterbox(bgr):
        h, w = bgr.shape[:2]
        k = S / max(w, h)
        nw, nh = int(round(w * k)), int(round(h * k))
        canvas = np.zeros((S, S, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, k

    def detect(bgr, thr=0.6):
        """The app's YuNet decoding (anchor-free, strides 8/16/32) + NMS. Returns [x, y, w, h, 5 points, score] in 640 space."""
        x = bgr.astype(np.float32).transpose(2, 0, 1)[None]
        outs = det.run(None, {det.get_inputs()[0].name: x})
        named = {o.name: v for o, v in zip(det.get_outputs(), outs)}
        dets = []
        for st in (8, 16, 32):
            cls = named[f"cls_{st}"].reshape(-1); obj = named[f"obj_{st}"].reshape(-1)
            box = named[f"bbox_{st}"].reshape(-1, 4); kps = named[f"kps_{st}"].reshape(-1, 10)
            cols = S // st
            for i in range(cls.size):
                sc = math.sqrt(min(max(cls[i], 0), 1) * min(max(obj[i], 0), 1))
                if sc < thr:
                    continue
                r, c = divmod(i, cols)
                cx = (c + box[i, 0]) * st; cy = (r + box[i, 1]) * st
                w = math.exp(box[i, 2]) * st; h = math.exp(box[i, 3]) * st
                pts = [((kps[i, 2 * n] + c) * st, (kps[i, 2 * n + 1] + r) * st) for n in range(5)]
                dets.append([cx - w / 2, cy - h / 2, w, h] + [v for p in pts for v in p] + [sc])
        dets.sort(key=lambda d: -d[-1])
        keep = []
        for d in dets:
            ok = True
            for k in keep:
                ax2, ay2 = d[0] + d[2], d[1] + d[3]; bx2, by2 = k[0] + k[2], k[1] + k[3]
                iw = max(0, min(ax2, bx2) - max(d[0], k[0])); ih = max(0, min(ay2, by2) - max(d[1], k[1]))
                inter = iw * ih; uni = d[2] * d[3] + k[2] * k[3] - inter
                if uni > 0 and inter / uni > 0.3:
                    ok = False; break
            if ok:
                keep.append(d)
        return keep

    TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float64)

    def similarity(src, dst):
        """Umeyama similarity (rotation + uniform scale + shift) taking src points to dst points, as a 2x3 matrix."""
        n = src.shape[0]
        mu_s = src.mean(0); mu_d = dst.mean(0)
        s0 = src - mu_s; d0 = dst - mu_d
        var_s = (s0 ** 2).sum() / n
        cov = d0.T @ s0 / n
        U, D, Vt = np.linalg.svd(cov)
        Sg = np.eye(2)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            Sg[1, 1] = -1
        R = U @ Sg @ Vt
        scale = (D * np.diag(Sg)).sum() / var_s
        t = mu_d - scale * R @ mu_s
        return np.hstack([scale * R, t[:, None]])

    def embed(sess, bgr, d):
        pts = np.array(d[4:14], np.float64).reshape(5, 2)
        M = similarity(pts, TEMPLATE)
        crop = cv2.warpAffine(bgr, M, (112, 112), flags=cv2.INTER_LINEAR)
        x = crop[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]  # RGB 0..255
        f = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
        return f / np.linalg.norm(f), M

    rec = ort.InferenceSession(sface, so, providers=["CPUExecutionProvider"])
    rec8 = ort.InferenceSession(sface8, so, providers=["CPUExecutionProvider"])
    log("sface inputs", [(i.name, i.shape) for i in rec.get_inputs()], "outputs", [(o.name, o.shape) for o in rec.get_outputs()])
    cvdet = cv2.FaceDetectorYN.create(yunet, "", (S, S), 0.6, 0.3, 5000)
    cvrec = cv2.FaceRecognizerSF.create(sface, "")
    best8 = 1.0
    feats = []
    for nm, img in imgs:
        lb, k = letterbox(img)
        mine = detect(lb)
        _, ref = cvdet.detect(lb)
        ref = [] if ref is None else ref
        log(nm, "faces mine", len(mine), "opencv", len(ref))
        for d in mine:
            r = min(ref, key=lambda q: abs(q[0] - d[0]) + abs(q[1] - d[1])) if len(ref) else None
            if r is not None:
                diff = max(abs(np.array(d[:14]) - np.array(r[:14])))
                log(f"  box/points max difference vs OpenCV: {diff:.3f} px, score {d[14]:.3f} vs {r[14]:.3f}")
                assert diff < 1.0, "YuNet decoding does not match OpenCV"
                f, M = embed(rec, lb, d)
                crop = cvrec.alignCrop(lb, r)
                fr = cvrec.feature(crop).reshape(-1); fr /= np.linalg.norm(fr)
                cs = float(f @ fr)
                f8, _ = embed(rec8, lb, d)
                best8 = min(best8, float(f8 @ f))
                log(f"  SFace: my alignment+features vs OpenCV cosine {cs:.4f}; int8 vs full {float(f8 @ f):.4f}")
                assert cs > 0.98, "SFace alignment/features do not match OpenCV"
                feats.append(f)
    if len(feats) >= 2:
        log("different people cosine (should be low):", round(float(feats[0] @ feats[-1]), 3))
    use8 = best8 > 0.97
    log("using", "int8" if use8 else "full", "SFace")
    import shutil
    shutil.copy(yunet, f"{OUT}/face_detect.onnx")
    shutil.copy(sface8 if use8 else sface, f"{OUT}/face_embed.onnx")


# ----------------------------------------------------------------------------------------------- speech
SOT, EOT, TRANSLATE, TRANSCRIBE, NOTS, TS0 = 50258, 50257, 50358, 50359, 50363, 50364
LANG0, LANG1 = 50259, 50357  # inclusive


def speech():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import onnxruntime as ort
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from onnxruntime.quantization import quantize_dynamic, QuantType

    name = "openai/whisper-base"
    m = WhisperForConditionalGeneration.from_pretrained(name, attn_implementation="eager").eval()
    proc = WhisperProcessor.from_pretrained(name)
    fe = proc.feature_extractor
    tok = proc.tokenizer
    assert tok.convert_tokens_to_ids("<|startoftranscript|>") == SOT and tok.convert_tokens_to_ids("<|notimestamps|>") == NOTS
    assert tok.convert_tokens_to_ids("<|0.00|>") == TS0 and tok.convert_tokens_to_ids("<|transcribe|>") == TRANSCRIBE
    assert tok.convert_tokens_to_ids("<|en|>") == LANG0
    langs = [tok.convert_ids_to_tokens(i)[2:-2] for i in range(LANG0, LANG1 + 1)]
    log("languages", len(langs), langs[:5], "...", langs[-3:])

    mel = torch.tensor(np.asarray(fe.mel_filters, np.float32))  # (201, 80)
    n = torch.arange(400, dtype=torch.float64); k = torch.arange(201, dtype=torch.float64)
    win = 0.5 - 0.5 * torch.cos(2 * math.pi * n / 400)  # periodic Hann
    ang = 2 * math.pi * k[:, None] * n[None, :] / 400
    wr = (torch.cos(ang) * win).float()[:, None, :]
    wi = (-torch.sin(ang) * win).float()[:, None, :]

    class Frontend(nn.Module):
        """30 s of 16 kHz sound -> Whisper's log-mel spectrogram -> encoder output."""
        def __init__(self):
            super().__init__()
            self.register_buffer("wr", wr); self.register_buffer("wi", wi); self.register_buffer("mel", mel.T.contiguous())
            self.enc = m.model.encoder
        def forward(self, audio):
            x = F.pad(audio[:, None, :], (200, 200), mode="reflect")
            re = F.conv1d(x, self.wr, stride=160); im = F.conv1d(x, self.wi, stride=160)
            p = (re * re + im * im)[:, :, :-1]
            mm = torch.matmul(self.mel, p)
            lg = torch.log10(torch.clamp(mm, min=1e-10))
            lg = torch.maximum(lg, lg.amax(dim=(1, 2), keepdim=True) - 8.0)
            lg = (lg + 4.0) / 4.0
            return self.enc(lg).last_hidden_state

    class Decoder(nn.Module):
        """Tokens so far + encoder output -> scores for the next token."""
        def __init__(self):
            super().__init__(); self.dec = m.model.decoder; self.proj = m.proj_out
        def forward(self, tokens, enc):
            h = self.dec(input_ids=tokens, encoder_hidden_states=enc, use_cache=False).last_hidden_state
            return self.proj(h[:, -1, :])

    jfk = fetch("https://github.com/openai/whisper/raw/main/tests/jfk.flac", f"{TMP}/jfk.flac")
    a, sr = sf.read(jfk, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    assert sr == 16000
    audio = np.zeros(480000, np.float32); audio[:len(a)] = a

    front = Frontend().eval(); dec = Decoder().eval()
    with torch.no_grad():
        # spectrogram check vs Hugging Face
        x = F.pad(torch.tensor(audio)[None, None, :], (200, 200), mode="reflect")
        re = F.conv1d(x, wr, stride=160); im = F.conv1d(x, wi, stride=160)
        p = (re * re + im * im)[:, :, :-1]; lg = torch.log10(torch.clamp(mel.T @ p, min=1e-10))
        lg = torch.maximum(lg, lg.max() - 8.0); lg = (lg + 4.0) / 4.0
        ref = fe(audio, sampling_rate=16000, return_tensors="np").input_features
        log("spectrogram max difference vs Hugging Face:", float(np.abs(lg.numpy() - ref).max()))
        enc_out = front(torch.tensor(audio)[None])
        torch.onnx.export(front, (torch.tensor(audio)[None],), f"{TMP}/enc.onnx", input_names=["audio"], output_names=["enc"], opset_version=17)
        t0 = torch.tensor([[SOT, LANG0, TRANSCRIBE]], dtype=torch.long)
        torch.onnx.export(dec, (t0, enc_out), f"{TMP}/dec.onnx", input_names=["tokens", "enc"], output_names=["logits"],
                          dynamic_axes={"tokens": {1: "n"}}, opset_version=17)
        hf = m.generate(torch.tensor(ref), return_timestamps=True)
        log("Hugging Face transcript:", tok.decode(hf[0], skip_special_tokens=True))
    log("exported")
    for nm in ["enc", "dec"]:
        quantize_dynamic(f"{TMP}/{nm}.onnx", f"{OUT}/speech_{nm}.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
        log(nm, "int8", os.path.getsize(f"{OUT}/speech_{nm}.onnx") // 1_000_000, "MB")

    # Vocabulary: one token per line, by id (byte-level text, as GPT-2).
    vocab = [tok.convert_ids_to_tokens(i) or "" for i in range(TS0)]
    assert all("\n" not in v for v in vocab)
    with open(f"{OUT}/speech_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))

    # ---- the app's transcription loop, in Python ----
    so = ort.SessionOptions()
    E = ort.InferenceSession(f"{OUT}/speech_enc.onnx", so, providers=["CPUExecutionProvider"])
    D = ort.InferenceSession(f"{OUT}/speech_dec.onnx", so, providers=["CPUExecutionProvider"])
    bs = bytes_to_unicode(); bd = {v: k for k, v in bs.items()}

    def text_of(ids):
        s = "".join(vocab[i] for i in ids if i < EOT)
        return bytes(bd[ch] for ch in s if ch in bd).decode("utf-8", errors="replace")

    t = time.time()
    segs, lang = transcribe(E, D, audio, len(a) / 16000.0, text_of)
    log(f"app loop ({time.time() - t:.1f}s), language {langs[lang - LANG0]}:")
    for s in segs:
        log(f"  [{s[0]:.2f} -> {s[1]:.2f}] {s[2]}")
    joined = " ".join(s[2] for s in segs).lower()
    assert "fellow americans" in joined and "country" in joined, "transcript is wrong"
    return langs


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def transcribe(E, D, audio30, speech_len, text_of):
    """Exactly what the app does for one 30 s window: language, then greedy decoding with Whisper's timestamp rules."""
    enc = E.run(None, {"audio": audio30[None].astype(np.float32)})[0]
    lg = D.run(None, {"tokens": np.array([[SOT]], np.int64), "enc": enc})[0][0]
    lang = int(np.argmax(lg[LANG0:LANG1 + 1])) + LANG0
    tokens = [SOT, lang, TRANSCRIBE]
    for _ in range(224):
        lg = D.run(None, {"tokens": np.array([tokens], np.int64), "enc": enc})[0][0].astype(np.float64)
        sampled = tokens[3:]
        lg[EOT + 1:TS0] = -np.inf  # no special tokens except end
        last_ts = len(sampled) >= 1 and sampled[-1] >= TS0
        pen_ts = len(sampled) < 2 or sampled[-2] >= TS0
        if last_ts:
            if pen_ts:
                lg[TS0:] = -np.inf
            else:
                lg[:EOT] = -np.inf
        if len(sampled) == 0:
            lg[:TS0] = -np.inf
            lg[TS0 + 50 + 1:] = -np.inf  # start within the first second
        prev = [x for x in sampled if x >= TS0]
        if prev:
            last = prev[-1] if (last_ts and not pen_ts) else prev[-1] + 1
            lg[TS0:last] = -np.inf
        mx = np.max(lg)
        lp = lg - mx - np.log(np.sum(np.exp(lg - mx)))
        if np.logaddexp.reduce(lp[TS0:]) > np.max(lp[:TS0]):
            lg[:TS0] = -np.inf
        nxt = int(np.argmax(lg))
        if nxt == EOT:
            break
        tokens.append(nxt)
    segs = []
    start = None; buf = []
    for x in tokens[3:]:
        if x >= TS0:
            ts = (x - TS0) * 0.02
            if buf and start is not None:
                segs.append((start, ts, text_of(buf).strip())); buf = []
            start = ts
        else:
            buf.append(x)
    if buf and start is not None:
        segs.append((start, min(30.0, speech_len), text_of(buf).strip()))
    return [s for s in segs if s[2]], lang


def finish(langs):
    files = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"): continue
        p = f"{OUT}/{f}"
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        files.append({"name": f, "size": os.path.getsize(p), "sha256": h})
    man = {
        "version": 1,
        "files": files,
        "speech": {"sot": SOT, "eot": EOT, "transcribe": TRANSCRIBE, "notimestamps": NOTS, "ts0": TS0,
                   "lang0": LANG0, "lang1": LANG1, "languages": langs, "window_s": 30, "rate": 16000},
        "faces": {"input": 640, "score": 0.6, "nms": 0.3, "same_person": 0.40,
                  "template": [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]}
    }
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    log("pack", sum(f["size"] for f in files) // 1_000_000, "MB:", [f["name"] for f in files])


if __name__ == "__main__":
    faces()
    langs = speech()
    finish(langs)
