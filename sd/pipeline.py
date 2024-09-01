#!/usr/bin/env python3
# Copyright (c) 2024 Shaojie Li (shaojieli.nlp@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from ddpm import DDPMSampler
from torch import nn
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 64
LATENT_HEIGHT = 64


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sample_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be in (0, 1]")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize the random number generator
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        clip = models["clip"]
        clip = clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], max_length=77, padding="max_length"
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], max_length=77, padding="max_length"
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], max_length=77, padding="max_length"
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)

        if sample_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sample_name: {sample_name}")

        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder = encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(
                latent_shape, generator=generator, device=device
            )
            latents = encoder(input_image_tensor, encoder_noise)

            # add noise to latents
            sampler.set_strenth(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusions = models["diffusions"]
        diffusions = diffusions.to(device)

        timesteps = tqdm(diffusions.timesteps)
        for i, step in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(step).to(device)
            model_input = latents
            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusions(model_input, context, time_embedding)

            if do_cfg:
                out_cond, out_uncond = torch.chunk(model_output, 2, dim=0)
                model_output = cfg_scale * (out_cond - out_uncond) + out_uncond

            # denoise
            latents = sampler.step(timesteps, latents, model_output)
        to_idle(diffusions)

        decoder = models["decoder"]
        decoder = decoder.to(device)
        # (1, 4, Latents_Height, Latents_Width) -> (1, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, input_range, output_range, clamp=False):
    input_min, input_max = input_range
    output_min, output_max = output_range
    y = (x - input_min) / (input_max - input_min) * (
        output_max - output_min
    ) + output_min
    if clamp:
        y = torch.clamp(y, output_min, output_max)
    return y


def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
