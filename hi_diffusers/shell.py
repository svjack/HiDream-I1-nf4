from .nf4 import *

import argparse
import time
import IPython
import logging

from IPython.display import Image, display


def gen(prompt: str, seed: int = -1, res: str = "1024x1024", output="output.png"):
    """Generate and display an image from the prompt."""
    resolution = tuple(map(int, res.strip().split("x")))
    
    st = time.time()
    image, final_seed = generate_image(pipe, args.model, prompt, resolution, seed)
    image.save(output)
    print(f"Image saved to {output}")
    print(f"Seed used: {final_seed}, Time: {time.time() - st:.2f} seconds")
    
    # Display the image
    display(Image(filename=output))


if __name__ == "__main__":
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="dev",
                        help="Model to use",
                        choices=["dev", "full", "fast"])
    args = parser.parse_args()

    # Load model 
    print(f"Loading model {args.model}...")
    pipe, _ = load_models(args.model)
    print()
    print("✅ Model loaded successfully!")
    print("Try gen('your prompt here') to generate an image.")
    print()
    
    # Set up IPython shell
    banner = f"""
HiDream-I1-nf4 Shell

Model: {args.model} NF4 Quantized
"""
    IPython.start_ipython(argv=[], user_ns={"gen": gen}, banner=banner, display_banner=True)
