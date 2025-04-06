import torch
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
pretrained_model_name_or_path = "HiDream-ai/HiDream-I1-Full"
scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_fast=False)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16).to("cuda")

transformer = HiDreamImageTransformer2DModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="transformer", 
    torch_dtype=torch.bfloat16).to("cuda")

pipe = HiDreamImagePipeline.from_pretrained(
    pretrained_model_name_or_path, 
    scheduler=scheduler,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer = transformer

prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."
images = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=50,
    num_images_per_prompt=1,
    generator=torch.Generator("cuda").manual_seed(42)
).images
for i, image in enumerate(images):
   image.save(f"{i}.jpg")