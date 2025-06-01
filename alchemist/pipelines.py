import torch
from diffusers.pipelines.sana import SanaPipeline


class AlchemistSanaPipeline(SanaPipeline):

    @torch.no_grad()
    def __call__(self, image, prompt: str, t: float):
        device = self._execution_device

        pixel_values = self.image_processor.preprocess(image).to(device)

        vae_output = self.vae.encode(pixel_values)
        latents = vae_output.latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=1,
            device=device,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            clean_caption=False,
            max_sequence_length=300,
            complex_human_instruction=None,
            lora_scale=None,
        )

        num_train_timesteps = self.scheduler.num_train_timesteps
        t_clamped = float(t)
        t_clamped = max(0.0, min(1.0, t_clamped))
        timestep_idx = int(t_clamped * (num_train_timesteps - 1))

        noise = torch.randn_like(latents, device=device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timestep_idx)
        self.noisy_latent = noisy_latents

        latent_model_input = noisy_latents.to(self.transformer.dtype)

        timestep_tensor = torch.tensor([timestep_idx], device=device, dtype=latent_model_input.dtype)
        timestep_tensor = timestep_tensor * self.transformer.config.timestep_scale

        noise_pred = self.transformer(
            latent_model_input,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep_tensor,
            return_dict=False,
            attention_kwargs=None
        )[0]

        self.noise_pred = noise_pred
