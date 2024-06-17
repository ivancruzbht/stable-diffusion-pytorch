import torch
import numpy as np

# Denoising diffusion probabilistic models (DDPM) are a class of generative models that are trained to denoise images.
# The model (UNET) is trained to predict the noise that was added to the image, and then subtract that noise from the image.
# This sampler is used to remove the noise from the image given the noise that was predicted by the UNET.
# The originial paper can be found here: https://arxiv.org/pdf/2006.11239

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120) -> None:
        # The beta_start and beta_end are the values that come from the original stable diffusion paper and were 
        # decided by the author.

        # Linear Beta scheduler as defined in the original stable diffusion code
        self.betas =  torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0) # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.one - torch.tensor(1.0, dtype=torch.float32)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy()) # numpy slicing operator: [start:stop:step], -1 means reverse the array 1 step at a time

    def set_inference_steps(self, n_inference_steps: int = 50) -> None:
        # Divide the total amount of steps (999) by n_inference_steps and space them out evenly to cover 0-999
        self.n_inference_steps = n_inference_steps
        step_ratio = self.num_training_steps // self.n_inference_steps
        timesteps = (np.arange(0, self.n_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.n_inference_steps)
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.FloatTensor:
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_bar[timestep]
        alpha_prod_t_prev = self.alpha_bar[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t/alpha_prod_t_prev

        # Compute the variance with eq (7) of the DDPM paper
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    # This function is to modify the number of steps the scheduler will do given a noise strenght
    # E.g. if the strength is 0.5, the scheduler will only do half of the steps
    # This is because when we condition on an image with a certain strength noise, we want to remove the noise from the image
    # but not all the way of the number of steps in the scheduler
    def set_strength(self, strength=1) -> None:
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    # This function describes how we can remove the noise given a latent, based on the noise that was predicted by the UNET
    # In the paper, the algorithm 2 and the eq (11) describes how to remove the noise once the UNET has predicted it. and go from X_t to X_t-1
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.FloatTensor:
        t = timestep
        prev_t = self._get_previous_timestep(t)
        alpha_prod_t = self.alpha_bar[t]
        alpha_prod_t_prev = self.alpha_bar[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # According to the equation (15) of the DDPM paper, to compute the predicted original sample X0:
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Compute the coefficient for the pred_original_sample and for current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / alpha_prod_t ** 0.5
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        std = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            std = (self._get_variance(t) ** 0.5) * noise

        # Z = μ + σ * ε where ε ~ N(0, 1), this is to convert the mean and std to a normal distribution
        pred_prev_sample = pred_prev_sample + std

        return pred_prev_sample



    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        # Add noise to the original samples
        # original_samples: (B, 4, H, W)
        # timesteps: (B)
        alpha_bar = self.alpha_bar.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_bar = torch.sqrt(alpha_bar[timesteps])
        sqrt_alpha_bar = sqrt_alpha_bar.flatten()
        # add dimensions to the tensor to match the original_samples shape
        while len(sqrt_alpha_bar.shape) < len(original_samples.shape):
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)

        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar[timesteps]) # standard deviation
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.flatten()
        # add dimensions to the tensor to match the original_samples shape
        while len(sqrt_one_minus_alpha_bar.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

        # Sample noise from a normal distribution according to the equation (4) of the DDPM paper
        # Sample from the normal distribution:
        # Z = μ + σ * ε where ε ~ N(0, 1), this is to convert the mean and variance(std) to a normal distribution
        noise = torch.rand(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_bar * original_samples) + (sqrt_one_minus_alpha_bar * noise)
        return noisy_samples