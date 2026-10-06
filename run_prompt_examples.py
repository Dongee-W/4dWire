"""Run selected prompt sets and save separate preview images/checkpoints."""
import argparse
import json
import shutil
import time
from pathlib import Path

import pydiffvg
import torch
import torchvision

from wire4d import render_batch, train_bspline

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Test 4Dwire prompt sets')
    parser.add_argument('prompt_sets', nargs='+', help='Keys from inputs/prompt.json')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=365)
    parser.add_argument('--schedule-steps', type=int, default=1801, help='Noise schedule length; the original notebook uses 1801')
    args = parser.parse_args()
    prompt_sets = json.loads((ROOT / 'inputs' / 'prompt.json').read_text())
    unknown = set(args.prompt_sets) - set(prompt_sets)
    if unknown:
        parser.error(f'Unknown prompt sets: {sorted(unknown)}. Available: {sorted(prompt_sets)}')
    if args.steps < 1 or args.schedule_steps < 1:
        parser.error('--steps and --schedule-steps must be at least 1')
    for prompt_set in args.prompt_sets:
        started = time.monotonic()
        model = train_bspline(steps=args.steps, prompt_set=prompt_set, seed=args.seed, schedule_steps=args.schedule_steps)
        output = ROOT / 'outputs' / 'prompt_tests' / prompt_set
        output.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            views = render_batch(model, device='cuda')
            grid = torchvision.utils.make_grid(views, nrow=3, padding=8, pad_value=1.0)
        pydiffvg.imwrite(grid.permute(1, 2, 0).cpu(), str(output / 'final_views.png'), gamma=1.0)
        shutil.copy2(ROOT / 'outputs' / 'model_final.pth', output / 'model_final.pth')
        shutil.copy2(ROOT / 'outputs' / 'debug_tile_0000.png', output / 'initial_views.png')
        record = {'prompt_set': prompt_set, 'prompts': prompt_sets[prompt_set],
                  'steps': args.steps, 'schedule_steps': args.schedule_steps, 'seed': args.seed,
                  'elapsed_seconds': round(time.monotonic() - started, 1)}
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        print(output / 'final_views.png', flush=True)
        del model, views, grid
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
