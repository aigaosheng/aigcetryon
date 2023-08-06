from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import os
import pickle

model_pth = "/home/gs/aigc/modelzoo"
model_name = {
    "openpose": os.path.join(model_pth, "openpose.pkl"),
    "ctl_openpose": os.path.join(model_pth, "sd-controlnet-openpose"),
    "ctl_canny": os.path.join(model_pth, "sd-controlnet-canny"),
    "sd": os.path.join(model_pth, "stable-diffusion-v1-5"),
}

if os.path.exists(model_name["openpose"]):
    with open(model_name["openpose"], "rb") as fo:
        openpose = pickle.load(fo)
else:
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    with open(model_name["openpose"], "wb") as fo:
        pickle.dump(openpose, fo)
ctl_openpose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
# ctl_openpose.save_pretrained(model_name["ctl_openpose"])

ctl_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny"),
# ctl_canny.save_pretrained(model_name["ctl_canny"])
controlnet = [
    ctl_openpose,
    ctl_canny
]
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
pipe.save_pretrained(model_name["sd"])
print(f"*** {model_name['sd']} saved")

