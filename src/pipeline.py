import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 512 // 8
LATENT_HEIGHT = 512 // 8

def generate(prompt: str, 
             uncond_prompt: str, # Unconditional prompt for no cfg generation, also called negative prompt (can be empty)
             input_image=None, # Image to condition the generation of a new image
             strength=0.8, # How much the model to pay attention to the input image (i.e how strong the noise)
             do_cfg=True, # Do classifier-free guidance (i.e. the model gives 2 outputs, one conditioned with the prompt and the other unconditioned, and then we mix them according to thr cfg_scale)
             cfg_scale=7.5, # How much to mix the two prompts
             sampler_name='ddpm', # Scheduler
             n_inference_steps=50,
             models={},
             seed=None,
             device=None,
             idle_device=None,
             tokenizer=None):
    
    with torch.no_grad():
        if not (0 < strength <=1):
            raise ValueError("Strength must be between 0 and 1")
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is None:
            seed = torch.seed()
        else:
            seed = generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        # In classifier-free guidance, we go through the UNET twice, once with the prompt and once without it and 
        # then we mix the two outputs according to the cfg scale
        if do_cfg:
            # convert prompt into tokes using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_lenght=77).input_ids
            # Convert tokens to tensor (B, T)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (B, T) -> (B, T, C) or (B, T, 768)
            cond_tokens = clip(cond_tokens).to(device)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_lenght=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (B, T) -> (B, T, C) or (B, T, 768)
            uncond_tokens = clip(uncond_tokens).to(device)

            # (2, T, C) or (2, 77, 768)
            context = torch.cat((cond_tokens, uncond_tokens))
        else:
            # Conver it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_lenght=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens).to(device)

        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        # If we have an input image, we need to run it through the encoder
        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize(WIDTH, HEIGHT)

            # (H, W, C)
            input_image_tensor = torch.tensor(np.array(input_image_tensor), dtype=torch.float32, device=device)

            # channels from 0-255 to -1 to 1, as the UNet model expects input in this range
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            # (H, W, C) -> (1, H, W, C) -> (1, C, H, W) 
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            enconder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # Run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, enconder_noise)

            sampler.set_strengh(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        # If we don't have an input image, we start with random noise
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
            sampler.set_strengh(strength=strength)

        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            # (B,4,LATENT_HEIGHT,LATENT_WIDTH)
            model_input = latents
            if do_cfg:
                # (B,4,LATENT_HEIGHT,LATENT_WIDTH) -> (2 * B,4,LATENT_HEIGHT,LATENT_WIDTH)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            # Combine the conditioned output and the unconditioned output
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove the noise predicted by the UNET from the latents
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1,1), (0,255), clamp=True)
        # (B, C, H, W) -> (B, H, W, C)
        images.permute(0, 2, 3, 1)
        images = images.cpu().numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = x - old_min
    x = x * (new_max - new_min) / (old_max - old_min) + new_min

    if clamp:
        x = torch.clamp(x, new_min, new_max)
    
    return x

# Time embedding is a vector that represents the current timestep
# Take the timestep and return a vector of size 320 that represents it
# TODO - Rework this function to make it more readable
def get_time_embedding(timestep):
    # (160)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1,160) -> (1,320)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


