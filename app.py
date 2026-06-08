from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

_local_appdata = os.environ.get("LOCALAPPDATA") or str(Path.home())
_GRADIO_TMP = Path(_local_appdata) / "demo_app_gradio_tmp"
_GRADIO_TMP.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_GRADIO_TMP)

import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
_orig_j2p = _gc_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


def _safe_j2p(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_j2p(schema, defs)


_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_j2p

import gradio as gr
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"
SD_TARGET_SIZE = 512


ALGORITHMS = [
    "OpenCV Telea (FMM)",
    "OpenCV Navier-Stokes",
    "LaMa",
    "Stable Diffusion Inpainting"
]

def _to_pil_rgb(arr) -> Image.Image:
    """Konvertuje numpy/PIL u PIL RGB sliku."""
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    if isinstance(arr, np.ndarray):
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr[:, :, :3]).convert("RGB")
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Nepodrzan tip slike: {type(arr)}")


def _extract_mask(editor_value) -> Image.Image | None:
    """Iz Gradio ImageEditor vrednosti izvlaci binarnu masku regiona."""
    if editor_value is None:
        return None

    layers = editor_value.get("layers") if isinstance(editor_value, dict) else None
    if not layers:
        return None

    combined = None
    for layer in layers:
        if layer is None:
            continue
        layer_arr = np.asarray(layer)
        if layer_arr.ndim == 3 and layer_arr.shape[2] == 4:
            alpha = layer_arr[:, :, 3]
        else:
            alpha = layer_arr if layer_arr.ndim == 2 else layer_arr.mean(axis=2)
        if combined is None:
            combined = alpha.astype(np.uint8)
        else:
            combined = np.maximum(combined, alpha.astype(np.uint8))

    if combined is None or combined.max() == 0:
        return None

    mask = (combined > 10).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def _resize_pad(img: Image.Image, size: int, mode: str = "RGB"):
    img = img.copy()
    interp = Image.LANCZOS if mode == "RGB" else Image.NEAREST
    img.thumbnail((size, size), interp)
    canvas_color = (0, 0, 0) if mode == "RGB" else 0
    canvas = Image.new(mode, (size, size), canvas_color)
    offset = ((size - img.size[0]) // 2, (size - img.size[1]) // 2)
    canvas.paste(img, offset)
    return canvas, offset, img.size


def inpaint_opencv(image: Image.Image, mask: Image.Image,
                   method: str, radius: int = 3) -> Image.Image:
 
    import cv2

    img_arr = np.array(image.convert("RGB"))
    mask_arr = np.array(mask.convert("L"))

    # OpenCV ocekuje BGR
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    result_bgr = cv2.inpaint(img_bgr, mask_arr, radius, flag)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


_lama_model = None


def get_lama():
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama
        print("[info] Ucitavanje LaMa modela...")
        _lama_model = SimpleLama()
        print("[info] LaMa ucitan.")
    return _lama_model


def inpaint_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    lama = get_lama()
    return lama(image.convert("RGB"), mask.convert("L"))

_sd_pipeline = None


def get_sd_pipeline():
    global _sd_pipeline
    if _sd_pipeline is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[info] Ucitavanje '{SD_MODEL_ID}' na '{device}'...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            SD_MODEL_ID, torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        _sd_pipeline = pipe
        print("[info] Stable Diffusion Inpainting ucitan.")
    return _sd_pipeline


def inpaint_stable_diffusion(image: Image.Image, mask: Image.Image,
                             num_steps: int, guidance: float) -> Image.Image:
  
    pipe = get_sd_pipeline()
    image_rgb = image.convert("RGB")
    mask_l = mask.convert("L")
    original_size = image_rgb.size  # (W, H)

    img_resized, offset, inner_size = _resize_pad(image_rgb, SD_TARGET_SIZE, mode="RGB")
    mask_resized, _, _ = _resize_pad(mask_l, SD_TARGET_SIZE, mode="L")

    result = pipe(
        prompt="",
        image=img_resized,
        mask_image=mask_resized,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance),
        strength=1.0,
    ).images[0]

    # skini padding i vrati rezultat na originalnu dimenziju.
    ox, oy = offset
    iw, ih = inner_size
    result_cropped = result.crop((ox, oy, ox + iw, oy + ih))
    result_full = result_cropped.resize(original_size, Image.LANCZOS)


    import cv2
    mask_arr = np.array(mask_l)
    mask_blur = cv2.GaussianBlur(mask_arr, (5, 5), 0)
    mask_blur_pil = Image.fromarray(mask_blur, mode="L")

    final = Image.composite(result_full, image_rgb, mask_blur_pil)
    return final


def inpaint(editor_value, algorithm: str,
            num_steps: int, guidance: float, cv_radius: int,
            rp_model: str, rp_steps: int, rp_jump_length: int,
            rp_jump_n_sample: int, rp_eta: float):
    """Pokrece izabrani algoritam popune nad slikom + maskom."""
    if editor_value is None or editor_value.get("background") is None:
        raise gr.Error("Ucitajte sliku pre pokretanja popune.")

    image = _to_pil_rgb(editor_value["background"])
    mask = _extract_mask(editor_value)
    if mask is None:
        raise gr.Error(
            "Nije detektovana maska. Cetkicom oznacite region za brisanje."
        )

    # Maska mora biti iste dimenzije kao slika za OpenCV/LaMa.
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    print(f"[info] Pokretanje algoritma: {algorithm}")
    if algorithm == "OpenCV Telea (FMM)":
        result = inpaint_opencv(image, mask, method="telea", radius=int(cv_radius))
    elif algorithm == "OpenCV Navier-Stokes":
        result = inpaint_opencv(image, mask, method="ns", radius=int(cv_radius))
    elif algorithm == "LaMa":
        result = inpaint_lama(image, mask)
    elif algorithm == "Stable Diffusion Inpainting":
        result = inpaint_stable_diffusion(image, mask, num_steps, guidance)
    else:
        raise gr.Error(f"Nepoznat algoritam: {algorithm}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_alg = algorithm.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = OUTPUT_DIR / f"inpaint_{safe_alg}_{timestamp}.png"
    result.save(out_path)
    print(f"[info] Sacuvano: {out_path}")

    return result, str(out_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Algoritmi za popunu slike", analytics_enabled=False) as demo:


        with gr.Row():
            with gr.Column():
                editor = gr.ImageEditor(
                    label="1. Ucitajte sliku  |  2. Cetkicom oznacite region za brisanje",
                    type="numpy",
                    sources=("upload",),
                    brush=gr.Brush(colors=["#ffffff"], default_size=30),
                    layers=False,
                    transforms=(),
                    height=512,
                )

                algorithm = gr.Dropdown(
                    choices=ALGORITHMS,
                    value=ALGORITHMS[0],
                    label="3. Izaberite algoritam popune",
                )

                with gr.Group(visible=True) as cv_group:
                    cv_radius = gr.Slider(
                        minimum=1, maximum=15, value=3, step=1,
                        label="OpenCV: poluprecnik regiona za interpolaciju",
                    )

                with gr.Group(visible=False) as sd_group:
                    steps = gr.Slider(
                        minimum=10, maximum=50, value=25, step=1,
                        label="Broj koraka denosinga",
                    )
                    guidance = gr.Slider(
                        minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                        label="Jacina vodjenja (guidance scale)",
                    )

                run_btn = gr.Button("4. Pokreni popunu", variant="primary")

            with gr.Column():
                output_img = gr.Image(label="5. Rezultat popune", height=512)
                saved_path = gr.Textbox(label="Sacuvano u", interactive=False)


        def _on_alg_change(alg):
            is_cv = alg.startswith("OpenCV")
            is_sd = alg == "Stable Diffusion Inpainting"
            return (
                gr.update(visible=is_cv),
                gr.update(visible=is_sd),
            )

        algorithm.change(
            fn=_on_alg_change,
            inputs=algorithm,
            outputs=[cv_group, sd_group],
        )

        run_btn.click(
            fn=inpaint,
            inputs=[
                editor, algorithm, steps, guidance, cv_radius
            ],
            outputs=[output_img, saved_path],
        )


    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0")
