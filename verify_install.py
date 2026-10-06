"""Check imports, rasterization and optional CUDA guidance in an installed environment."""
import argparse
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

import torch
import pydiffvg
from wire4d import StableDiffusion, render_sample_dreamwire


def main():
    parser = argparse.ArgumentParser(description="Verify the 4Dwire installation")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if no CUDA GPU is visible")
    parser.add_argument("--load-guidance", action="store_true", help="Load the Stable Diffusion weights and encode the food prompts")
    parser.add_argument("--surface", action="store_true", help="Render a procedural surface filling curve")
    args = parser.parse_args()
    print(f"Python package versions: torch {torch.__version__}, torchvision {metadata.version('torchvision')}, diffusers {metadata.version('diffusers')}, diffvg {metadata.version('diffvg')}")
    print(f"PyTorch CUDA build: {torch.version.cuda}; GPU visible: {torch.cuda.is_available()}")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA is not available in this shell; training requires a GPU-enabled session")
    device = 'cuda' if args.require_cuda else 'cpu'
    path = render_sample_dreamwire(size=64, device=device, output=Path(__file__).resolve().parent / 'outputs' / f'verify_{device}.png')
    print(f"pydiffvg render passed: {path}")
    if args.surface:
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "surface_filling.py"),
                        "--example", "sphere", "--steps", "0", "--size", "64",
                        "--points", "24", "--device", device,
                        "--output", str(Path(__file__).resolve().parent / "outputs" / f"verify_surface_{device}")], check=True)
        print("Surface filling render passed")
    if args.load_guidance:
        if not torch.cuda.is_available():
            parser.error("--load-guidance requires CUDA")
        import json
        prompts = json.loads((Path(__file__).resolve().parent / 'inputs' / 'prompt.json').read_text())['food']
        guidance = StableDiffusion('cuda')
        embeddings = guidance.get_text_embeds(prompts)
        assert embeddings.shape[0] == 2 * len(prompts)
        print(f"Stable Diffusion prompt encoding passed: {tuple(embeddings.shape)}")


if __name__ == '__main__':
    main()
