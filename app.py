from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


def process_image(image, prompt, threhsold, alpha_value, draw_rectangles):
    inputs = processor(
        text=prompt, images=image, padding="max_length", return_tensors="pt"
    )

    # predict
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits

    pred = torch.sigmoid(preds)
    mat = pred.cpu().numpy()
    mask = Image.fromarray(np.uint8(mat * 255), "L")
    mask = mask.convert("RGB")
    mask = mask.resize(image.size)
    mask = np.array(mask)[:, :, 0]

    # normalize the mask
    mask_min = mask.min()
    mask_max = mask.max()
    mask = (mask - mask_min) / (mask_max - mask_min)

    # threshold the mask
    bmask = mask > threhsold
    # zero out values below the threshold
    mask[mask < threhsold] = 0

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(mask, alpha=alpha_value, cmap="jet")

    if draw_rectangles:
        contours, hierarchy = cv2.findContours(
            bmask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect = plt.Rectangle(
                (x, y), w, h, fill=False, edgecolor="yellow", linewidth=2
            )
            ax.add_patch(rect)

    ax.axis("off")
    plt.tight_layout()

    return fig, mask


title = "Interactive demo: zero-shot image segmentation with CLIPSeg"
description = "Demo for using CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. To use it, simply upload an image and add a text to mask (identify in the image), or use one of the examples below and click 'submit'. Results will show up in a few seconds."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"


with gr.Blocks() as demo:
    gr.Markdown("# CLIPSeg: Image Segmentation Using Text and Image Prompts")
    gr.Markdown(article)
    gr.Markdown(description)
    gr.Markdown(
        "*Example images are taken from the [ImageNet-A](https://paperswithcode.com/dataset/imagenet-a) dataset*"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil")
            input_prompt = gr.Textbox(label="Please describe what you want to identify")
            input_slider_T = gr.Slider(
                minimum=0, maximum=1, value=0.4, label="Threshold"
            )
            input_slider_A = gr.Slider(minimum=0, maximum=1, value=0.5, label="Alpha")
            draw_rectangles = gr.Checkbox(label="Draw rectangles")
            btn_process = gr.Button(label="Process")

        with gr.Column():
            output_plot = gr.Plot(label="Segmentation Result")
            output_image = gr.Image(label="Mask")

    btn_process.click(
        process_image,
        inputs=[
            input_image,
            input_prompt,
            input_slider_T,
            input_slider_A,
            draw_rectangles,
        ],
        outputs=[output_plot, output_image],
    )

    gr.Examples(
        [
            ["0.003473_cliff _ cliff_0.51112.jpg", "dog", 0.5, 0.5, True],
            ["0.001861_submarine _ submarine_0.9862991.jpg", "beacon", 0.55, 0.4, True],
            ["0.004658_spatula _ spatula_0.35416836.jpg", "banana", 0.4, 0.5, True],
        ],
        inputs=[
            input_image,
            input_prompt,
            input_slider_T,
            input_slider_A,
            draw_rectangles,
        ],
    )

demo.launch()
