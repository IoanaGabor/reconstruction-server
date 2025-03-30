from reconstruction_service.abstract_image_embedding_reconstructor import AbstractImageReconstructor
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL


class StableDiffusionReconstructor(AbstractImageReconstructor):

    def __init__(self):
        base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_model_path = "stabilityai/sd-vae-ft-mse"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        torch.cuda.empty_cache()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            device="cuda"
        )
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        images = self.pipe(
            prompt_embeds=torch.stack([torch.tensor(embeddings[0:16])]),
            negative_prompt_embeds=torch.stack([torch.tensor(embeddings[16:32])]),
            num_inference_steps=100 
        ).images

        return images[0]
