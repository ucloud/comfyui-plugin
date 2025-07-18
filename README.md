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

Note: (Multi-inputs) uses same node as normal (single input). Please check out the example workflow hidden in the images below.

## Instruction

1. Install ComfyUI, please refer to [ComfyUI official github](https://github.com/comfyanonymous/ComfyUI).

2. Change to `ComfyUI/custom_nodes` directory, and clone this repo, install required python packages.

    ```
    cd ～/ComfyUI/custom_nodes
    git clone https://github.com/ucloud/ucloud-comfyui.git
    cd ./ucloud-comfyui
    pip install -r requirements.txt
    ```

3. Back to `ComfyUI` directory, start ComfyUI service, use web browser to open `localhost:8188`.

    ```
    cd ～/ComfyUI
    python main.py
    ```

4. Build your own workflow with Modelverse nodes. You can check the following images, drag them into ComfyUI GUI and autoload example workflows. Don't forget to enter your `API_KEY` in the `Modelverse Client` node.

    | Flux Dev | Flux Kontext Pro | Flux Kontext Max (Multi-inputs) |
    |:-:|:-:|:-:|
    | <img src="assets/flux_dev.png"  width="200" height="200"> | <img src="assets/flux_kontext_pro_single.png"  width="200" height="200"> | <img src="assets/flux_kontext_max_multi.png"  width="200" height="200"> |

    In case the above images does not work for you, check the screenshot below:

    Text-to-Image with Flux Dev
    ![Text-to-Image](assets/screenshot-flux_dev.png)

    Single image editting with Flux Kontext Pro
    ![single-input](assets/screenshot-flux_kontext_pro.png)

    Multi image editting with Flux Kontext Max
    ![Multi-inputs](assets/screenshot-flux_kontext_max.png)