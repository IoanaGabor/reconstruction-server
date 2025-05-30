from reconstruction_service.abstract_image_embedding_reconstructor import AbstractImageReconstructor
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus



class StableDiffusionReconstructor(AbstractImageReconstructor):

    def __init__(self):
        #base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        #vae_model_path = "stabilityai/sd-vae-ft-mse"
        base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        image_encoder_path = "models/image_encoder"
        ip_ckpt = "models/ip-adapter-plus_sd15.bin"
        device = "cuda"

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
        #generator = torch.Generator(device="cuda").manual_seed(0)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )
        #generator = torch.Generator(device="cuda").manual_seed(0)
        ip_model = IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
        generator = torch.Generator(device="cuda").manual_seed(0)
        #self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        generator = torch.Generator(device="cuda").manual_seed(0)

        images = self.pipe(
            prompt_embeds=torch.stack([torch.tensor(embeddings[0:16])]),
            negative_prompt_embeds=torch.stack([torch.tensor(embeddings[16:32])]),
            num_inference_steps=100 
        ).images
        print(len(images))
        return images[-1]
