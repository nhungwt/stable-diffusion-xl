import gradio as gr
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import os
import gc
import datetime

# Create new folfer
new_directory_path = 'imagesSDXL'
os.makedirs(new_directory_path, exist_ok=True)

from helper import UNetDataParallel

model_dir = os.getenv("SDXL_MODEL_DIR")

if model_dir:
    # Use local model
    model_key_base = os.path.join(model_dir, "stable-diffusion-xl-base-1.0")
    model_key_refiner = os.path.join(model_dir, "stable-diffusion-xl-refiner-1.0")
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-1.0"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"


offload_base = os.getenv("OFFLOAD_BASE", "true").lower() == "true"

# Generate how many images by default
default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "1"))

pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
multi_gpu = os.getenv("MULTI_GPU", "false").lower() == "true"
checkpoint_path =  os.getenv("CHECKPOINT_PATH")

if multi_gpu:
    if checkpoint_path != "NONE":
        print("Loading checkpoint...")
        pipe.load_lora_weights(checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
        print("Loading checkpoint success.")
    pipe.unet = UNetDataParallel(pipe.unet)
    pipe.unet.config, pipe.unet.dtype, pipe.unet.add_embedding = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.add_embedding
    pipe.to("cuda")
else:
    if offload_base:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")


def infer(prompt, negative, scale, samples=4, steps=50, seed=-1):
    prompt, negative = [prompt] * samples, [negative] * samples

    g = torch.Generator(device="cuda")
    if seed != -1:
        g.manual_seed(seed) # create CỐ ĐỊNH random number if pass the same argument - right here is 'seed'
    else:
        g.seed()

    images_b64_list = []

    images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale, num_inference_steps=steps, generator=g).images
    
    gc.collect()
    torch.cuda.empty_cache()

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)
    
        # Save image in to folder
        current_time = datetime.datetime.now()
        file_name = current_time.strftime("%m-%d_%H-%M")
        status = 'pos' if negative[0] == '' else 'nev'
        seed_ = g.initial_seed()
        file_path = f"{new_directory_path}/{file_name}_arg_{steps}_{scale}_{seed_}_{status}.jpg"
        image.save(file_path, format="JPEG")

    return images_b64_list
    
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(css=css)

examples = [
    [
        'nscm_Elsa princess is riding a horse in the pine forest in the autumn',
        '',
        9
    ],
    [
        'nscm_Olaf snowman wearing a witch outfit',
        '',
        9
    ],
    [
        'nscm_cup cup has yellow pattern of dancing cats on the bottom edge of the cup',
        '',
        9
    ],
    [
        'An insect robot preparing a delicious meal',
        'low quality, illustration',
        9
    ],
    [
        "nscm_msTaylor woman is playing with cat",
        'duplicate cat',
        9
    ],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Stable Diffusion XL
                </h1>
              </div>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                with gr.Column():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Accordion("Advanced settings", open=False):
            samples = gr.Slider(label="Images", minimum=1, maximum=max(2, default_num_images), value=default_num_images, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=50, step=1)
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=9, step=0.1
            )

            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale], outputs=[gallery], cache_examples=False)
        ex.dataset.headers = [""]
        negative.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, seed], outputs=[gallery], postprocess=False)
        text.submit(infer, inputs=[text, negative, guidance_scale, samples, steps, seed], outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[text, negative, guidance_scale, samples, steps, seed], outputs=[gallery], postprocess=False)
        
block.queue().launch(share=True)
