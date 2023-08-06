from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
from controlnet_aux import OpenposeDetector
import torch
import os
import pickle

def canny_extractor(img_file):
    canny_image = load_image(img_file)
    canny_image = np.array(canny_image)
    
    low_threshold = 100
    high_threshold = 200
    
    canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)
    
    # zero out middle columns of image where pose will be overlayed
    zero_start = canny_image.shape[1] // 4
    zero_end = zero_start + canny_image.shape[1] // 2
    canny_image[:, zero_start:zero_end] = 0
    
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    return canny_image

def openpose_extractor(img_file, openpose_model):
    if os.path.exists(openpose_model):
        with open(openpose_model, "rb") as fo:
            openpose = pickle.load(fo)
    else:
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        with open(openpose_model, "wb") as fo:
            pickle.dump(openpose, fo)
    i_image = load_image(img_file)
    openpose_image = openpose(i_image)

    return openpose_image


model_pth = "/home/gs/aigc/modelzoo"
model_name = {
    "openpose": os.path.join(model_pth, "controlnet/openpose.pkl"),
    "ctl_openpose": os.path.join(model_pth, "controlnet/sd-controlnet-openpose"),
    "ctl_canny": os.path.join(model_pth, "controlnet/sd-controlnet-canny"),
    "sd": os.path.join(model_pth, "stable-diffusion-v1-5"),
}

is_gpu = torch.cuda.is_available()
if is_gpu:
    device = "cuda"
    md_dtype = torch.float16
else:
    device = "cpu"
    md_dtype = torch.float32

canny_image = canny_extractor(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)

openpose_image = openpose_extractor(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png",
    model_name["openpose"]
)

controlnet = [
    ControlNetModel.from_pretrained(model_name["ctl_openpose"], torch_dtype=md_dtype),
    ControlNetModel.from_pretrained(model_name["ctl_canny"], torch_dtype=md_dtype),
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_name["sd"], controlnet=controlnet, torch_dtype=md_dtype
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if is_gpu:
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.Generator(device="cpu").manual_seed(1)

images = [openpose_image, canny_image]

image = pipe(
    prompt,
    images,
    num_inference_steps=2,
    generator=generator,
    negative_prompt=negative_prompt,
    controlnet_conditioning_scale=[1.0, 0.8],
).images[0]

image.save("./multi_controlnet_output.png")
