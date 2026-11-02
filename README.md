# SCP Media AI pack

Open AI models used by SCP Media's on-phone **AI outfit editor**. The app downloads these files
itself (from this repo's Releases) when you tap **Download AI pack**. Nothing else is stored here.

| Model | Licence |
|---|---|
| Stable Diffusion 1.5 inpainting | CreativeML Open RAIL-M (use restrictions apply) |
| LCM-LoRA for SD 1.5 (merged in, for fast 4–8 step renders) | Open RAIL++ |
| Falconsai nsfw_image_detection (GRAIL.MECH safety check) | Apache-2.0 |
| CLIP tokenizer vocabulary | MIT |

Full licence texts are in `LICENSES.md` inside each release.

## Building a new pack

Actions → **Build AI pack** → Run workflow. It downloads the models, converts them to ONNX,
shrinks the big one to 8-bit, renders test pictures (`test_torch.png`, `test_ort_full.png`,
`test_ort_8bit.png`) to check quality, and publishes everything as a release.
