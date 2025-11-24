# ITOPipeline: SDXL with Information-Theoretic Guidance

This repo provides a thin, production‑oriented wrapper around
[`StableDiffusionXLPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
that adds:

- Information-Theoretic Optimization (ITO) guidance
  – adaptive per‑step guidance strength based on a KL budget
- Stable FP32 decoding
  – VAE runs in float32 with tiling/slicing to reduce NaNs and OOMs
- Standard SDXL path with fp32 decode
  – a `generate_fixed` helper for "vanilla" SDXL images
- Negative prompts & SDXL dual‑prompt support
- A small `make_grid` utility to tile images with labels

The goal is to make it easy to A/B test ITO guidance vs. vanilla SDXL in a
single, clean interface.

---

## 1. Installation

You'll need:

- Python 3.9+
- PyTorch with CUDA (recommended) or CPU
- `diffusers`, `transformers`, `accelerate`, `safetensors`, `Pillow`, `numpy`

### Quick Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/leochlon/ITO.git
cd ITO
pip install -r requirements.txt
```

The `requirements.txt` file includes all necessary dependencies with pinned versions for reproducibility.

> **Note:** Make sure your PyTorch build matches your CUDA version if you're on GPU.

By default the pipeline loads:

- `stabilityai/stable-diffusion-xl-base-1.0`

You may need a Hugging Face token configured to download the model.

---

## 2. What is ITO guidance?

Instead of using a fixed CFG scale (e.g. 7.5) for all diffusion steps,
ITO guidance:

- Treats each diffusion step as spending part of a KL budget.
- Measures how different the conditional and unconditional noise predictions are.
- Chooses a per‑step guidance strength `lambda_k` so that:

$$
\sum_k \tfrac{1}{2} \lambda_k^2 q_k \approx \text{budget}
$$

where `q_k` is a scalar measure of how strong the text signal is at step `k`.

This means:

- Early steps (high noise, strong text signal) tend to get higher λ.
- Later steps (low noise, weaker text signal) tend to get lower λ.
- You control how "hard" the model is pushed overall via a single `budget`
  (instead of guessing a fixed CFG scale).

On top of that, there's an `alpha` soft‑rescale that stabilizes guidance by
matching the variance of the guided noise to the "pure text" direction.

---

## 3. Quickstart

### 3.1 Running ITO.py with CLI

An easy way to try ITO is to run the module directly with command-line arguments:

```bash
bash# Use default prompt
python ITO.py

# Custom prompt
python ITO.py --prompt "cozy cabin in the mountains"
python ITO.py -p "futuristic cityscape at sunset"

# Full customization
python ITO.py -p "a serene lake" -n "blurry, ugly" --budget 50 --steps 30 --alpha 0.5

# Skip baseline comparison (only generate ITO)
python ITO.py -p "forest path" --no-baseline

# Custom output filename
python ITO.py -p "mountain vista" -o my_mountain

# View all options
python ITO.py --help
```
This will generate comparison images using the specified prompt and save them to the current directory, demonstrating the difference between ITO guidance and standard fixed CFG.

**Available CLI options:**

```
--prompt, -p: Main text prompt (default: "a photo of an astronaut riding a horse on mars")
--negative, -n: Negative prompt (default: "blurry, low resolution, ugly")
--budget, -b: KL budget for ITO guidance (default: 40.0)
--lambda-max: Maximum guidance strength per step (default: 7.5)
--alpha, -a: Rescale factor, 0=aggressive, 1=stable (default: 0.3)
--steps, -s: Number of diffusion steps (default: 40)
--cfg: Fixed CFG scale for baseline comparison (default: 7.5)
--seed: Random seed for reproducibility (default: 42)
--height: Image height in pixels (default: 1024)
--width: Image width in pixels (default: 1024)
--output, -o: Output filename prefix (default: "output")
--no-baseline: Skip baseline CFG generation
--no-grid: Skip comparison grid generation
--quiet, -q: Suppress verbos
```

### 3.2 Running the Web Application

For an interactive web interface:

**1.** Try it online: 🤗 [Our Hugging Face Space for ITO](https://huggingface.co/spaces/sirine16/ITO)

**2.** Or run it locally with the Gradio app:

```bash
python app.py
```

This will launch a local web server where you can:
  - Enter custom prompts and negative prompts
  - Adjust ITO parameters (budget, lambda_max, alpha)
  - **Compare ITO vs fixed CFG side-by-side**
  - Download generated images

The app will be accessible at `http://localhost:7860` (or another port if 7860 is in use).

---

## 4. ITOPipeline API

### 4.1 Initialization

```python
ito = ITOPipeline(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",  # default
)
```

What happens under the hood:

- Loads `StableDiffusionXLPipeline.from_pretrained(model_id, ...)`
  - Uses `torch.float16` on GPU / MPS, `torch.float32` on CPU
  - Enables `safetensors`
- Replaces the scheduler with DPM-Solver++ (Karras sigmas).
- Moves the VAE to float32 and enables:
  - `enable_vae_tiling()`
  - `enable_vae_slicing()`
- Enables TF32 matmul on CUDA for a performance/precision sweet spot.
- Disables the diffusers progress bar (less noisy logs).

---

### 4.2 `generate_ito(...)`

```python
image, total_kl, lambdas = ito.generate_ito(
    prompt: str,
    prompt_2: Optional[str] = None,
    budget: float = 40.0,
    lambda_max: Optional[float] = 7.5,
    num_steps: int = 40,
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
    alpha: float = 0.3,
    verbose: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
)
```

Key parameters:

- `prompt`: main SDXL text prompt.
- `prompt_2`: optional second SDXL text encoder prompt (e.g. style / extra info).
- `negative_prompt`, `negative_prompt_2`: standard SDXL negatives for each encoder.
- `budget`: total KL budget. Higher = stronger overall guidance.
- `lambda_max`: hard per‑step cap on guidance strength. `None` disables the cap.
- `num_steps`: diffusion steps (e.g. 30–50).
- `height`, `width`: resolution in pixels (multiples of 64 recommended).
- `alpha`:
  - `0.0` → minimal rescale (more vivid / aggressive guidance)
  - `1.0` → fully rescaled (more stable, closer to standard CFG behavior)
- `seed`: for reproducibility.
- `verbose`: print per‑10‑step stats, final KL and latent/decoded stats.

Returns:

- `image` — `PIL.Image.Image`, decoded from fp32 VAE.
- `total_kl` — accumulated KL usage, approx. `<= budget`.
- `lambdas` — list of per‑step guidance strengths used.

---

### 4.3 `generate_fixed(...)`

```python
image = ito.generate_fixed(
    prompt: str,
    prompt_2: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_steps: int = 40,
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
)
```

This is a convenience wrapper for standard SDXL generation, but:

- Asks diffusers to output latents (`output_type="latent"`).
- Decodes those latents using the fp32 VAE in this pipeline.
- Clamps VAE output to `[-1, 1]` and runs `image_processor` to get a PIL image.

Use this to:

- Get a "baseline" SDXL image for comparison with the ITO version.
- Enjoy more stable VAE decoding than the default fp16 path.

---

### 4.4 `make_grid(images, labels)`

```python
from ITO import make_grid

grid = make_grid(
    images=[image_ito, image_fixed],
    labels=["ITO (budget 40)", "Baseline CFG 7.5"],
)
grid.save("comparison_grid.png")
```

- Creates a horizontal grid of images with 40px of padding at the top.
- Draws the corresponding label above each image (if provided).

---

## 5. Repository Structure

```
ITO/
├── ITO.py              # Main pipeline implementation
├── app.py              # Gradio web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 6. Tips & Notes

- **GPU strongly recommended.** SDXL at 1024×1024 on CPU is extremely slow.
- If you hit NaNs / infs in latents, the code already:
  - Detects non‑finite values.
  - Sanitizes them with `torch.nan_to_num`.
- If you get OOMs, try:
  - Lowering `height`/`width` (e.g. to 768×768).
  - Reducing `num_steps`.
- `budget`, `lambda_max`, and `alpha` are meant to be tunable knobs:
  - Start with `budget=30–50`, `lambda_max=7.5`, `alpha=0.3`.
  - Increase `budget` for more "locked‑in" prompt adherence (at risk of artifacts).
  - Increase `alpha` for more stability / less aggressive guidance.

---

## 7. License / Attribution

This repo wraps the `StableDiffusionXLPipeline` and SDXL model weights
provided by Stability AI via Hugging Face. Please make sure your usage complies
with:

- The license terms of the SDXL model you load (e.g. `stabilityai/stable-diffusion-xl-base-1.0`).
- Hugging Face's and Stability AI's usage policies.

Based on our upcoming preprint "Information-Budgeted Inference-Time Optimization for Diffusion Models: A Theoretical Foundation with Event-Level Guarantees".

This code is released under the MIT License by Hassana Labs Ltd.

---

MIT License

Copyright (c) 2025 Hassana Labs Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
