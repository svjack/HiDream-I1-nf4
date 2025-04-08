import torch
import argparse
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="dev")
args = parser.parse_args()
model_type = args.model_type


MODEL_PREFIX = "azaneko"
LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

def log_vram(msg: str):
    print(msg)
    print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Load models
def load_models(model_type: str):
    config = MODEL_CONFIGS[model_type]
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME)
    log_vram("Tokenizer loaded!")
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        return_dict_in_generate=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    log_vram("Text encoder loaded!")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        config["path"],
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    log_vram("Transformer loaded!")
    
    pipe = HiDreamImagePipeline.from_pretrained(
        config["path"],
        scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False),
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer = transformer
    log_vram("Pipeline loaded!")
    pipe.enable_sequential_cpu_offload()
    
    return pipe, config

# Generate image function
@torch.inference_mode()
def generate_image(pipe, model_type, prompt, resolution, seed):
    # Get configuration for current model
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = resolution
 
    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(seed)
    
    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator
    ).images
    
    return images[0], seed

# Initialize with default model
print("Loading default model (full)...")
pipe, _ = load_models(model_type)
print("Model loaded successfully!")
prompt = "A cat holding a sign that says \"I1 nf4\"." 
# Possible values: 1024x1024, 768x1360, 1360x768, 880x1168, 1168x880, 1248x832, 832x1248
resolution = (1024, 1024)
seed = -1
image, seed = generate_image(pipe, model_type, prompt, resolution, seed)
image.save("output.png")
