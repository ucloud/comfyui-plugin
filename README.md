# UCloud ModelVerse API for ComfyUI

## Supported models

Now we do support following models:

    - Flux Dev
    - Flux Kontext Pro Text2Image
    - Flux Kontext Pro 
    - Flux Kontext Pro (Multi-inputs)
    - Flux Kontext Max Text2Image
    - Flux Kontext Max 
    - Flux Kontext Max (Multi-inputs)
    - Step1X-Edit

You can find the corresponding nodes in our node set.

Note: (Multi-inputs) uses same node as normal (single input). Please check out the example workflow hidden in the following images.

## Instruction

1. Install ComfyUI, please refer to [ComfyUI official github](https://github.com/comfyanonymous/ComfyUI).

2. Change to ComfyUI/custom_nodes directory, and clone this repo.

    ```
    cd ComfyUI/custom_nodes
    git clone https://github.com/ucloud/ucloud-comfyui.git
    ```

3. Start ComfyUI service, use browser to open `localhost:8188`.

    ```
    cd ../ComfyUI
    python main.py
    ```

4. Build your own workflow with Modelverse nodes. You can check the following images, drag them into ComfyUI GUI and autoload example workflows.

    | Flux Dev | Flux Kontext Pro | Flux Kontext Max (Multi-inputs) |
    |:-:|:-:|:-:|
    | <img src="assets/flux_dev.png"  width="200" height="200"> | <img src="assets/flux_kontext_pro_single.png"  width="200" height="200"> | <img src="assets/flux_kontext_max_multi.png"  width="200" height="200"> |