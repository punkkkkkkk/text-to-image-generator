# text-to-image

## THERE'RE ARE 2 PROJECTS INIT 1ST IS FAST AND RELIABLE WHEREAS IN 2ND IT'S HIGHLY DETAILED ####

                                    WORKS IN GOOGLE COLAB ONLY 

                         ### IF TRIED TO RUN LOCALLY IT'LL TAKE TOO LONG ###


                            
This project shows how to generate images from text descriptions using the Stable Diffusion model. The implementation is designed to run in **Google Colab** for easy setup and execution. The following images are genrated using 2nd Project

<p align="center">
  <img src="https://github.com/user-attachments/assets/05d726f3-d29f-4199-8e27-85773ef3e1e1" alt="Top Left Image" width="400" style="margin: 10px;">
  <img src="https://github.com/user-attachments/assets/240f53ab-f5a3-4afa-9319-76d9b454bd82" alt="Top Right Image" width="400" style="margin: 10px;">
  <br>
  <img src="https://github.com/user-attachments/assets/6916d0cd-5820-4559-8023-f24e526fe786" alt="Bottom Left Image" width="400" style="margin: 10px;">
  <img src="https://github.com/user-attachments/assets/2f22ecf1-33bc-47e0-b470-bc9b6e07afce" alt="Bottom Right Image" width="400" style="margin: 10px;">
</p>



## Running the Code

- Open a new Google Colab notebook.
- Copy and paste the code snippets above into the notebook cells.
- Execute each cell sequentially to set up the environment and generate images.

## Notes

- Ensure you select a GPU runtime in Colab for faster image generation. Go to **Runtime** > **Change runtime type** and select **GPU**.
- If you encounter any issues, double-check that the packages are installed and the runtime is configured correctly.

## Acknowledgments

- This project uses the Stable Diffusion model from [CompVis](https://github.com/CompVis/stable-diffusion).
- Thanks to Hugging Face for providing the `diffusers` library for easy model integration.

## Libraries Used

1. torch

Purpose: PyTorch is an open-source deep learning framework used for building and training neural networks. In this project, it provides GPU support and helps in loading and managing the model.
Key Functionality: Efficient tensor computation and GPU acceleration.

3. diffusers

Purpose: The diffusers library by Hugging Face provides tools and pipelines for using diffusion models, like Stable Diffusion, for text-to-image tasks.
Key Functionality: Simplifies the integration and use of large pre-trained models, such as Stable Diffusion, for generating images from text.

5. accelerate
   
Purpose: This library by Hugging Face helps optimize the model's performance on various hardware setups, particularly on GPUs, to ensure efficient computation.
Key Functionality: Simplifies distributed training and model acceleration in a user-friendly way.

7. PIL (Pillow)
   
Purpose: A Python library for image processing. It is used here to handle and display the generated images.
Key Functionality: Converting and manipulating image data in Python.

9. matplotlib

Purpose: A plotting library used for visualizing data in Python. Here, it is used to display the generated images in the Colab notebook.
Key Functionality: Rendering images and hiding axes for a cleaner display.



========================================================================================================================================================================

# 1st PROJECT


## Project Overview

This project uses Stable Diffusion, a powerful text-to-image generation model, to create images from text descriptions. The model, developed by CompVis and made easily accessible through Hugging Face’s diffusers library, is capable of generating high-quality, detailed images based on user-provided prompts. The implementation is designed to run efficiently in Google Colab, where you can leverage the power of GPUs for faster performance.

## How It Works

1) Text Prompt Input: The user provides a text prompt that describes the image they want to generate.

2) Stable Diffusion Model: The StableDiffusionPipeline from the diffusers library processes the text input and generates an image. The model uses a process called latent diffusion to iteratively create an image that matches the given description.

3) Image Display: The generated image is displayed using matplotlib for easy visualization in the Colab notebook.

The process involves deep learning techniques that use pre-trained neural networks to transform textual information into visual content, relying heavily on GPU acceleration for efficient computation.


========================================================================================================================================================================

# 2nd PROJECT

## Project Overview

This project uses Stable Diffusion XL (SDXL), a more advanced version of the Stable Diffusion model, for generating high-quality images from text descriptions. The implementation employs two main stages to create refined images: an initial generation step and a refinement step, leveraging models provided by Stability AI. The code is optimized for running in Google Colab, taking advantage of GPU acceleration to speed up the generation process. It uses the diffusers library from Hugging Face, which provides convenient tools for working with diffusion models, along with the transformers and accelerate libraries for efficient model execution.

## How It Works

1. Setup and Libraries
   
The project starts by installing necessary libraries: diffusers, transformers, and accelerate. These libraries streamline the process of loading and running the models efficiently.
Warnings from the libraries are suppressed for a cleaner output using Python's warnings module.

2. Loading the Models

Base Model: The Stable Diffusion XL Base Model generates an initial image from a text prompt. It is loaded using the DiffusionPipeline class and set to run on the GPU (cuda).
Refiner Model: The SDXL Refiner Model enhances the initial image to produce more refined and detailed results. It shares some components (e.g., the text encoder and VAE) with the base model to optimize performance.

3. Image Generation Process

The generate() function takes a text prompt as input and follows a two-step process:
Initial Image Generation: The base model generates a preliminary image using the given text prompt, with a specified number of inference steps (n_steps) and noise settings (high_noise_frac) to control the generation process.

Image Refinement: The refiner model improves the initial image, making it more detailed and visually appealing.
The final image is then displayed directly in the Colab notebook using IPython.display.

4. User Interaction Loop
   
The code includes a loop that allows users to input text prompts continuously to generate images. Users can type 'q' to quit the loop.
The process is interactive, making it suitable for experimentation and creativity.
