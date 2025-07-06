# Seed Transition/Traveling README

## Overview

The **Seed Transition/Travel** script facilitates generating a sequence of images transitioning from one seed to another, optionally creating a video from the sequence. 
It allows various configurations including random seed generation, interpolation curves, and video settings.

## Features

- **Destination Seeds:** Specify a list of seeds or generate random seeds.
- **Interpolation Curves:** Customize image transitions with different curves (e.g., Linear, Hug-the-middle).
- **Video Creation:** Create a video from the generated images, adjustable FPS.
- **Upscaling:** Optionally upscale images using available upscalers.
- **SSIM Comparison:** Optionally compare images using Structural Similarity Index (SSIM).

## Installation

Ensure you have the following Python libraries installed:
- Gradio
- ImageIO
- Matplotlib
- NumPy
- Torch
- Torchmetrics
- torchvision

## Usage

1. **Set Parameters:**
   - **Destination Seeds:** Enter seeds or leave blank to use random seeds.
   - **Number of Random Seeds:** Define how many random seeds to generate.
   - **Steps:** Number of interpolation steps between seeds.
   - **FPS:** Frames per second for the video output.
   - **SSIM Threshold & CenterCrop:** Adjust to compare image quality.
   - **Upscaler & Ratio:** Select an upscaling method and ratio if desired.
   - **Interpolation Curve & Strength:** Choose and configure the interpolation curve.

2. **Run Script:** Execute the script with your configured parameters.

3. **View Output:**
   - Images are displayed if the option is selected.
   - Video is saved in the specified path if enabled.

