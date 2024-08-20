import gradio as gr
import imageio
import math
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import re
import sys
import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms
from torch.nn import functional as F
import modules.scripts as scripts
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state, sd_upscalers
from modules.images import resize_image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rife.ssim import ssim_matlab
from rife.RIFE_HDv3 import Model

# Ustawienia domyślne
DEFAULT_UPSCALE_METH = opts.data.get('customscript/seed_travel.py/txt2img/Upscaler/value', 'Lanczos')
DEFAULT_UPSCALE_RATIO = opts.data.get('customscript/seed_travel.py/txt2img/Upscale ratio/value', 1.0)
CHOICES_UPSCALER = [x.name for x in sd_upscalers]

class Script(scripts.Script):
    def title(self):
        return "Seed travel"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        seed_travel_extra = []

        # Ustawienia interfejsu użytkownika
        return [
            gr.Textbox(label='Destination seeds', lines=1),
            gr.Checkbox(label='Use random seeds', value=False),
            gr.Number(label='Number of random seeds', value=4),
            gr.Number(label='Steps', value=10),
            gr.Checkbox(label='Loop back to initial seed', value=False),
            gr.Number(label='FPS', value=30),
            gr.Number(label='Lead in/out', value=0),
            gr.Slider(label='SSIM threshold', info='0 to disable', value=0.0, minimum=0.0, maximum=1.0, step=0.01),
            gr.Slider(label='SSIM CenterCrop%', info='0 to disable', value=0, minimum=0, maximum=100, step=1),
            gr.Number(label='RIFE passes', value=0),
            gr.Checkbox(label='Drop original frames', value=False),
            gr.Dropdown(label='Interpolation curve', value='Linear', choices=[
                'Linear', 'Hug-the-middle', 'Hug-the-nodes', 'Slow start', 'Quick start', 'Easy ease in', 'Partial', 'Random'
            ]),
            gr.Slider(label='Curve strength', value=3, minimum=0.0, maximum=10.0, step=0.1),
            gr.Accordion(label='Seed Travel Extras...', open=False),
            gr.HTML(value='Seed Travel links: <a href=http://github.com/yownas/seed_travel/>Github</a>'),
            gr.Row([
                gr.Dropdown(label='Upscaler', value=DEFAULT_UPSCALE_METH, choices=CHOICES_UPSCALER),
                gr.Slider(label='Upscale ratio', value=DEFAULT_UPSCALE_RATIO, minimum=0.0, maximum=8.0, step=0.1)
            ]),
            gr.Row([
                gr.Checkbox(label='Use cache', value=True),
                gr.Checkbox(label='Show generated images in ui', value=True),
            ]),
            gr.Checkbox(label='Allow default sampler', value=False),
            gr.Checkbox(label='Compare paths', value=False),
            gr.Slider(label='Bump seed', value=0.0, minimum=0, maximum=0.5, step=0.001),
            gr.Number(label='SSIM min substep', value=0.001),
            gr.Slider(label='SSIM min threshold', value=75, minimum=0, maximum=100, step=1),
            gr.Checkbox(label='Save extra status information', value=True),
        ]

    @staticmethod
    def get_next_sequence_number(path):
        from pathlib import Path
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                result = max(result, num)
            except ValueError:
                pass
        return result + 1

    def run(self, p, rnd_seed, seed_count, dest_seed, steps, curve, curvestr, loopback, video_fps,
            show_images, compare_paths, allowdefsampler, bump_seed, lead_inout, upscale_meth, upscale_ratio,
            use_cache, ssim_diff, ssim_ccrop, substep_min, ssim_diff_min, rife_passes, rife_drop, save_stats):

        def initialize_paths():
            travel_path = os.path.join(p.outpath_samples, "travels")
            os.makedirs(travel_path, exist_ok=True)
            travel_number = Script.get_next_sequence_number(travel_path)
            travel_path = os.path.join(travel_path, f"{travel_number:05}")
            os.makedirs(travel_path, exist_ok=True)
            return travel_path, travel_number

        def validate_parameters():
            if bump_seed > 0:
                return False, False, 1
            if not rnd_seed and not dest_seed:
                print(f"No destination seeds were set.")
                return False, False, 0
            if rnd_seed and (not seed_count or int(seed_count) < 2):
                print(f"You need at least 2 random seeds.")
                return False, False, 0
            return True, save_video, video_fps

        def generate_seeds():
            seeds = []
            if rnd_seed:
                if (compare_paths or bump_seed) and p.seed is not None:
                    seeds.append(p.seed)
                seeds.extend(random.randint(0, 2147483647) for _ in range(seed_count))
            else:
                seeds = [int(p.seed)] if p.seed else []
                seeds.extend(int(x.strip()) for x in dest_seed.split(","))
            return seeds

        def process_images_with_seed(p, seed, subseed, strength, cache_key, use_cache, image_cache):
            p.seed, p.subseed, p.subseed_strength = seed, subseed, strength
            if use_cache and cache_key in image_cache:
                return image_cache[cache_key]
            proc = process_images(p)
            image = [resize_image(0, proc.images[0], tgt_w, tgt_h, upscaler_name=upscale_meth)] if upscale_meth != 'None' and upscale_ratio not in {1.0, 0.0} else [proc.images[0]]
            if use_cache:
                image_cache[cache_key] = image
            return image

        # Inicjalizacja
        travel_path, travel_number = initialize_paths()
        valid_params, save_video, video_fps = validate_parameters()
        if not valid_params:
            return Processed(p, [], p.seed)
        
        seeds = generate_seeds()
        p.seed = seeds[0]
        initial_prompt, initial_negative_prompt = p.prompt, p.negative_prompt
        
        travel_queue = [[seeds[0], seeds[i+1]] for i in range(len(seeds) - 1)] if compare_paths else [[seed for seed in seeds]]

        generation_queues = []
        for travel in travel_queue:
            generation_queue = []
            for s in range(len(travel) - (0 if loopback else 1)):
                seed, subseed = travel[s], travel[s + 1] if s + 1 < len(travel) else travel[0]
                numsteps = int(steps) + (1 if s + 1 == len(travel) else 0)
                for i in range(numsteps):
                    strength = self.calculate_strength(i, numsteps, curve, curvestr)
                    generation_queue.append((seed, subseed, strength))
            if not loopback:
                generation_queue.append((subseed, subseed, 0.0))
            generation_queues.append(generation_queue)

        total_images = len(set(key for queue in generation_queues for key in queue))
        print(f"Generating {total_images} images.")
        state.job_count = total_images
        image_cache = {}
        images = []

        for queue in generation_queues:
            step_images = []
            step_keys = []
            for idx, (seed, subseed, strength) in enumerate(queue):
                cache_key = f"{seed}_{subseed}_{strength:.2f}"
                img = process_images_with_seed(p, seed, subseed, strength, cache_key, use_cache, image_cache)[0]
                step_images.append(img)
                step_keys.append(cache_key)
                
                if show_images:
                    img.show(title=f"Image {idx + 1}/{len(queue)}")

            images.extend(step_images)
            state.current_image = step_images[-1]
            state.job_count -= len(step_images)
            if not save_stats:
                continue
            for img_key, img in zip(step_keys, step_images):
                ssim = ssim_matlab(images[0], img)
                print(f"SSIM for image {img_key}: {ssim}")

        if save_video:
            self.create_video(travel_path, images, video_fps, save_stats, rife_passes, rife_drop)

        return Processed(p, images, p.seed)

    @staticmethod
    def calculate_strength(i, numsteps, curve, curvestr):
        """Calculate the strength based on the interpolation curve."""
        t = i / (numsteps - 1)
        if curve == 'Linear':
            return t
        elif curve == 'Hug-the-middle':
            return 2 * t - t ** 2
        elif curve == 'Hug-the-nodes':
            return 1 - (1 - t) ** 2
        elif curve == 'Slow start':
            return t ** 2
        elif curve == 'Quick start':
            return 1 - (1 - t) ** 2
        elif curve == 'Easy ease in':
            return t ** 3
        elif curve == 'Partial':
            return t
        elif curve == 'Random':
            return random.random()
        else:
            return t

    @staticmethod
    def create_video(travel_path, images, video_fps, save_stats, rife_passes, rife_drop):
        """Create a video from images."""
        video_file = os.path.join(travel_path, "output.mp4")
        with imageio.get_writer(video_file, fps=video_fps) as writer:
            for img in images:
                writer.append_data(np.array(img))
        if save_stats:
            with open(os.path.join(travel_path, "stats.txt"), 'w') as f:
                f.write(f"Video saved to: {video_file}\n")

