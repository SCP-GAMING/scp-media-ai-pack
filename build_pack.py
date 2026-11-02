"""
Builds the SCP Media AI pack (outfit editing) from open models and publishes nothing by itself;
the workflow uploads the 'pack' folder as a GitHub release.

Models (all redistributable, licences copied into the pack):
  - Stable Diffusion 1.5 inpainting (CreativeML OpenRAIL-M)
  - LCM-LoRA for SD 1.5, merged in so 4-6 steps are enough (OpenRAIL++)
  - Falconsai/nsfw_image_detection image safety checker (Apache-2.0)  [GRAIL.MECH]

Everything is exported to ONNX for ONNX Runtime on Android. The big UNet is shrunk to 8-bit.
Test pictures are rendered three ways (PyTorch, ONNX full size, ONNX 8-bit) so quality can be checked.
"""
import hashlib, json, os, shutil, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw

OUT = "pack"
TMP = "tmp"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"

if STAGE in ("export", "all"):
    import torch
    torch.set_grad_enabled(False)
    from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
    from transformers import AutoModelForImageClassification, CLIPTextModel
    from diffusers.models.attention_processor import AttnProcessor
    from huggingface_hub import hf_hub_download

    # ---------------------------------------------------------------- load + merge LCM-LoRA
    REPOS = ["stable-diffusion-v1-5/stable-diffusion-inpainting", "runwayml/stable-diffusion-inpainting"]
    pipe = None
    for repo in REPOS:
        try:
            # "eager" attention: the ONNX converter can't handle the newer fused attention in the text encoder.
            te_model = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", attn_implementation="eager")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(repo, text_encoder=te_model, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
            BASE = repo
            log("loaded", repo)
            break
        except Exception as e:
            log("could not load", repo, e)
    if pipe is None:
        sys.exit("no inpainting model")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    SCALE = float(pipe.vae.config.scaling_factor)
    log("LCM-LoRA merged; vae scaling", SCALE)

    # ---------------------------------------------------------------- test picture
    def load_url(u):
        with urllib.request.urlopen(u, timeout=60) as r:
            return Image.open(r).convert("RGB")
    try:
        test_img = load_url("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png").resize((512, 512))
        test_mask = load_url("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png").convert("L").resize((512, 512))
    except Exception as e:
        log("test image download failed, drawing one", e)
        test_img = Image.new("RGB", (512, 512), (120, 160, 200))
        d = ImageDraw.Draw(test_img); d.ellipse((180, 60, 330, 210), fill=(230, 190, 160)); d.rectangle((150, 210, 360, 500), fill=(40, 60, 140))
        test_mask = Image.new("L", (512, 512), 0); ImageDraw.Draw(test_mask).rectangle((140, 200, 370, 512), fill=255)
    TEST_PROMPT = "a red knitted christmas sweater with white snowflakes, fully clothed, modest everyday clothing, family friendly, safe for work"
    test_img.save(f"{OUT}/test_input.png"); test_mask.save(f"{OUT}/test_mask.png")

    g = torch.Generator().manual_seed(7)
    ref = pipe(prompt=TEST_PROMPT, image=test_img, mask_image=test_mask, num_inference_steps=4, guidance_scale=1.0, generator=g).images[0]
    ref.save(f"{OUT}/test_torch.png")
    log("torch reference rendered")

    # ---------------------------------------------------------------- scheduler constants for the app
    sched = pipe.scheduler
    lcm = {}
    for steps in (4, 6, 8):
        sched.set_timesteps(steps)
        ts = [int(t) for t in sched.timesteps]
        rows = []
        for i, t in enumerate(ts):
            a = float(sched.alphas_cumprod[t])
            a_prev = float(sched.alphas_cumprod[ts[i + 1]]) if i + 1 < len(ts) else float(sched.final_alpha_cumprod)
            c_skip, c_out = sched.get_scalings_for_boundary_condition_discrete(t)
            rows.append({"t": t, "a": a, "a_prev": a_prev, "c_skip": float(c_skip), "c_out": float(c_out)})
        lcm[str(steps)] = rows
    log("lcm", json.dumps(lcm["4"]))

    # ---------------------------------------------------------------- export to ONNX
    # Plain attention for the export: same maths, written with matmul + softmax. (The usual version starts
    # from a blank tensor that the converter bakes in as huge tables of zeros: 530 MB each.)
    class PlainAttn:
        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            nd = hidden_states.ndim
            if nd == 4:
                b, c, h, w = hidden_states.shape
                hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
            bs, seq, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, seq, bs)
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            q = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            q = attn.head_to_batch_dim(q); k = attn.head_to_batch_dim(k); v = attn.head_to_batch_dim(v)
            # In slices of 1024 rows, so the phone never needs the whole (8 x 4096 x 4096) table at once (~1 GB).
            kt = k.transpose(-1, -2)
            outs = []
            for i in range(0, q.shape[1], 1024):
                sc = torch.bmm(q[:, i:i + 1024], kt) * attn.scale
                if attention_mask is not None:
                    sc = sc + attention_mask[:, i:i + 1024]
                outs.append(torch.bmm(sc.softmax(dim=-1), v))
            hidden_states = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            if nd == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            return hidden_states / attn.rescale_output_factor
    pipe.unet.set_attn_processor(PlainAttn())
    pipe.vae.set_attn_processor(PlainAttn())
    # Same answer as before? (checks the plain version against the one that made the reference picture)
    with torch.no_grad():
        _s = torch.randn(1, 9, 64, 64, generator=torch.Generator().manual_seed(1)); _h = torch.randn(1, 77, 768, generator=torch.Generator().manual_seed(2))
        _a = pipe.unet(_s, torch.tensor([500.0]), encoder_hidden_states=_h).sample
        pipe.unet.set_attn_processor(AttnProcessor())
        _b = pipe.unet(_s, torch.tensor([500.0]), encoder_hidden_states=_h).sample
        pipe.unet.set_attn_processor(PlainAttn())
        log("plain attention max difference", float((_a - _b).abs().max()))
    class TextEnc(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, ids): return s.m(ids)[0]
    class VaeEnc(torch.nn.Module):
        def __init__(s, v): super().__init__(); s.v = v
        def forward(s, x): return s.v.encode(x).latent_dist.mean * SCALE
    class VaeDec(torch.nn.Module):
        def __init__(s, v): super().__init__(); s.v = v
        def forward(s, z): return s.v.decode(z / SCALE).sample
    class Unet(torch.nn.Module):
        def __init__(s, u): super().__init__(); s.u = u
        def forward(s, sample, t, h): return s.u(sample, t, encoder_hidden_states=h).sample

    def export(mod, args, path, ins, outs, dyn=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.onnx.export(mod.eval(), args, path, input_names=ins, output_names=outs, opset_version=17, do_constant_folding=True, dynamic_axes=dyn)
        log("exported", path)

    export(TextEnc(pipe.text_encoder), (torch.zeros(1, 77, dtype=torch.int64),), f"{TMP}/te/text_encoder.onnx", ["input_ids"], ["hidden"])
    export(VaeEnc(pipe.vae), (torch.zeros(1, 3, 512, 512),), f"{TMP}/vae_enc/vae_encoder.onnx", ["image"], ["latent"])
    export(VaeDec(pipe.vae), (torch.zeros(1, 4, 64, 64),), f"{TMP}/vae_dec/vae_decoder.onnx", ["latent"], ["image"])
    export(Unet(pipe.unet), (torch.zeros(1, 9, 64, 64), torch.tensor([999.0]), torch.zeros(1, 77, 768)), f"{TMP}/unet/unet.onnx", ["sample", "timestep", "hidden"], ["eps"])

    tok = pipe.tokenizer
    tok.save_pretrained(f"{TMP}/tok")
    shutil.copy(f"{TMP}/tok/vocab.json", f"{OUT}/vocab.json")
    shutil.copy(f"{TMP}/tok/merges.txt", f"{OUT}/merges.txt")

    # ---------------------------------------------------------------- GRAIL.MECH: image safety checker (export)
    clf = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection", attn_implementation="eager").eval()
    labels = {str(k): v for k, v in clf.config.id2label.items()}
    class Clf(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m(pixel_values=x).logits
    export(Clf(clf), (torch.zeros(1, 3, 224, 224),), f"{TMP}/clf/safety.onnx", ["pixel_values"], ["logits"])
    json.dump(labels, open(f"{OUT}/safety_labels.json", "w"))
    # ---------------------------------------------------------------- tokenizer test vectors (to check the app's tokenizer)
    prompts = ["a red knitted christmas sweater", "Santa outfit, with a hat!", "denim jacket & jeans", "it's a 1990s hoodie",
               "Halloween witch costume    (purple)", "tank-top", "café uniform", TEST_PROMPT]
    vectors = [{"text": p, "ids": tok(p, padding="max_length", max_length=77, truncation=True).input_ids} for p in prompts]
    json.dump(vectors, open(f"{OUT}/testvectors.json", "w"))

    # ---------------------------------------------------------------- licences
    lic = ["# AI pack licences\n", "This pack contains models converted for on-device use by SCP Media.\n"]
    for repo, name in ((BASE, "Stable Diffusion 1.5 inpainting — CreativeML OpenRAIL-M"),
                       ("latent-consistency/lcm-lora-sdv1-5", "LCM-LoRA SD 1.5 — OpenRAIL++"),
                       ("Falconsai/nsfw_image_detection", "Falconsai nsfw_image_detection — Apache-2.0")):
        lic.append(f"\n## {name}\nSource: https://huggingface.co/{repo}\n")
        for fn in ("LICENSE", "LICENSE.md", "license.txt"):
            try:
                lic.append(open(hf_hub_download(repo, fn)).read()); break
            except Exception:
                pass
    open(f"{OUT}/LICENSES.md", "w").write("\n".join(lic))


    ids = [tok(TEST_PROMPT, padding="max_length", max_length=77, truncation=True).input_ids]
    json.dump({"base": BASE, "scale": SCALE, "lcm": lcm, "ids": ids, "labels": labels}, open(f"{TMP}/state.json", "w"))
    log("stage 1 done")


if STAGE in ("finish", "all"):
    st = json.load(open(f"{TMP}/state.json"))
    BASE, SCALE, lcm, labels = st["base"], st["scale"], st["lcm"], st["labels"]
    ids = np.array(st["ids"], np.int64)
    test_img = Image.open(f"{OUT}/test_input.png").convert("RGB")
    test_mask = Image.open(f"{OUT}/test_mask.png").convert("L")
    # ---------------------------------------------------------------- shrink to 8-bit
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    import subprocess
    def quant(src, dst, ops):
        # In a fresh process, so it has the memory to itself.
        subprocess.run([sys.executable, "quant_one.py", src, dst, ",".join(ops)], check=True)
        log("quantized", dst, os.path.getsize(dst) // 1_000_000, "MB")

    quant(f"{TMP}/te/text_encoder.onnx", f"{TMP}/text_encoder_q.onnx", ["MatMul", "Gemm"])
    quant(f"{TMP}/unet/unet.onnx", f"{OUT}/unet.onnx", ["MatMul", "Gemm", "Conv"])
    if os.environ.get("STOP_AFTER_UNET"):
        sys.exit("stopping after the UNet shrink (diagnostic run)")
    shutil.copy(f"{TMP}/vae_enc/vae_encoder.onnx", f"{OUT}/vae_encoder.onnx")
    shutil.copy(f"{TMP}/vae_dec/vae_decoder.onnx", f"{OUT}/vae_decoder.onnx")

    so = ort.SessionOptions()
    def sess(p): return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

    # The 8-bit text encoder is only used if it gives (nearly) the same answer as the full one.
    h32 = sess(f"{TMP}/te/text_encoder.onnx").run(None, {"input_ids": ids})[0]
    h8 = sess(f"{TMP}/text_encoder_q.onnx").run(None, {"input_ids": ids})[0]
    cos = float((h32 * h8).sum() / np.sqrt((h32 * h32).sum() * (h8 * h8).sum()))
    log("text encoder 8-bit similarity", cos)
    shutil.copy(f"{TMP}/text_encoder_q.onnx" if cos > 0.995 else f"{TMP}/te/text_encoder.onnx", f"{OUT}/text_encoder.onnx")

    # ---------------------------------------------------------------- the app's pipeline, in Python (the Kotlin code mirrors this)
    def run_pipeline(unet_path, seed=7, steps=4, strength=1.0):
        te, ve, vd, un = sess(f"{OUT}/text_encoder.onnx"), sess(f"{OUT}/vae_encoder.onnx"), sess(f"{OUT}/vae_decoder.onnx"), sess(unet_path)
        hidden = te.run(None, {"input_ids": ids})[0]
        img = np.asarray(test_img, np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
        m = (np.asarray(test_mask, np.float32) / 255.0 >= 0.5).astype(np.float32)[None, None]
        masked = img * (m < 0.5)
        lat_m = ve.run(None, {"image": masked.astype(np.float32)})[0]
        m64 = m[:, :, ::8, ::8]  # nearest, exactly like torch interpolate (pixel 8*i)
        rng = np.random.RandomState(seed)
        rows = lcm[str(steps)]
        # Keep-shape edits start part-way, from the original picture with some noise added (like the app).
        start = min(len(rows) - 1, int(round(len(rows) * (1.0 - strength))))
        rows = rows[start:]
        noise = rng.randn(1, 4, 64, 64).astype(np.float32)
        if start > 0:
            z0 = ve.run(None, {"image": img.astype(np.float32)})[0]
            x = (np.sqrt(rows[0]["a"]) * z0 + np.sqrt(1 - rows[0]["a"]) * noise).astype(np.float32)
        else:
            x = noise
        for i, r in enumerate(rows):
            inp = np.concatenate([x, m64, lat_m], 1).astype(np.float32)
            eps = un.run(None, {"sample": inp, "timestep": np.array([r["t"]], np.float32), "hidden": hidden})[0]
            x0 = (x - np.sqrt(1 - r["a"]) * eps) / np.sqrt(r["a"])
            den = r["c_out"] * x0 + r["c_skip"] * x
            x = (np.sqrt(r["a_prev"]) * den + np.sqrt(1 - r["a_prev"]) * rng.randn(*x.shape).astype(np.float32)) if i + 1 < len(rows) else den
        out = vd.run(None, {"latent": x.astype(np.float32)})[0]
        out = np.clip((out[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    t0 = time.time(); run_pipeline(f"{TMP}/unet/unet.onnx").save(f"{OUT}/test_ort_full.png"); log("onnx full render", round(time.time() - t0), "s")
    t0 = time.time(); run_pipeline(f"{OUT}/unet.onnx").save(f"{OUT}/test_ort_8bit.png"); log("onnx 8-bit render", round(time.time() - t0), "s")
    t0 = time.time(); run_pipeline(f"{OUT}/unet.onnx", steps=8, strength=0.6).save(f"{OUT}/test_ort_8bit_keepshape.png"); log("onnx 8-bit keep-shape render", round(time.time() - t0), "s")
    shutil.rmtree(f"{TMP}/unet", ignore_errors=True)

    # ---------------------------------------------------------------- GRAIL.MECH: image safety checker (shrink + check)
    quant(f"{TMP}/clf/safety.onnx", f"{OUT}/safety_classifier.onnx", ["MatMul", "Gemm"])
    def clf_probs(im):
        a = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
        l = sess(f"{OUT}/safety_classifier.onnx").run(None, {"pixel_values": a})[0][0]
        e = np.exp(l - l.max()); return dict(zip([labels[str(i)] for i in range(len(l))], (e / e.sum()).round(4).tolist()))
    log("safety labels", labels, "test result", clf_probs(Image.open(f"{OUT}/test_ort_8bit.png")))

    # ---------------------------------------------------------------- manifest
    APP_FILES = ["text_encoder.onnx", "unet.onnx", "vae_encoder.onnx", "vae_decoder.onnx", "safety_classifier.onnx",
                 "safety_labels.json", "vocab.json", "merges.txt", "LICENSES.md"]
    # Big models keep their weights in a side file next to them; the app downloads those too.
    APP_FILES += [n + ".data" for n in list(APP_FILES) if os.path.exists(f"{OUT}/{n}.data")]
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
        return h.hexdigest()
    manifest = {
        "pack": 1, "kind": "outfit-inpaint", "base": BASE, "vae_scaling": SCALE, "size": 512, "latent": 64,
        "files": [{"name": n, "size": os.path.getsize(f"{OUT}/{n}"), "sha256": sha(f"{OUT}/{n}")} for n in APP_FILES],
        "lcm": lcm,
    }
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    log("pack ready", sum(f["size"] for f in manifest["files"]) // 1_000_000, "MB")
    for n in sorted(os.listdir(OUT)): log(" ", n, os.path.getsize(f"{OUT}/{n}") // 1000, "KB")
