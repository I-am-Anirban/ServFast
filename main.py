from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

#SDXL Turbo
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo")
pipe.to("mps")
pipe.enable_attention_slicing()

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

_ = pipe(prompt, num_inference_steps=1)

image = pipe(prompt).images[0]
# @app.post('/generate-image')
# async def generate_image(text: str):
#     image = pipe(prompt).images[0]
#     return {'image': image}



