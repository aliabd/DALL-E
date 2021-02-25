import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dall_e import map_pixels, unmap_pixels, load_model
import gradio as gr
import torch.nn.functional as F

target_image_size = 256
dev = torch.device('cpu')
# For faster load times, download these files locally and use the local paths instead.
enc = load_model("https://cdn.openai.com/dall-e/encoder.pkl", dev)
dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", dev)


def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


def generate(img):
    x = preprocess(img)
    T.ToPILImage(mode='RGB')(x[0])
    z_logits = enc(x)
    z = torch.argmax(z_logits, axis=1)
    z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()
    x_stats = dec(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
    return x_rec


inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(label="Reconstructed Image")

examples = [
    ["penguin.png"],
    ["kangaroo.png"]
]
title = "Discrete VAE for DALL-E"
description = "This is a demo of the discrete VAE used for OpenAI's DALL-E. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. "
article = "<p style='text-align: center'><a href='https://openai.com/blog/dall-e/'>DALL·E: Creating Images from Text</a> | <a href='https://github.com/openai/DALL-E'>Github Repo</a></p>"

gr.Interface(generate, inputs, outputs, title=title, description=description, examples=examples, article=article).launch()