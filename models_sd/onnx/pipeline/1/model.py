import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import DPMSolverMultistepScheduler
from torch.utils.dlpack import from_dlpack
from transformers import CLIPTokenizer


class TritonPythonModel:
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded. Implementing `initialize` function is
        optional. This function allows the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.device = torch.device(
            "cpu" if args["model_instance_kind"] == "CPU" else "cuda"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained("weights_sd/tokenizer")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "weights_sd/scheduler"
        )

        model_config = json.loads(args["model_config"])
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(model_config, "image")["data_type"]
        )

        model_hparams = model_config["parameters"]
        self.unet_inchannels = int(model_hparams["unet_inchannels"]["string_value"])
        self.vae_scale_factor = int(model_hparams["vae_scale_factor"]["string_value"])

    def predict_text_encoder(self, input_ids):
        input_ids = pb_utils.Tensor("input_ids", input_ids.numpy().astype(np.int32))
        encoding_request = pb_utils.InferenceRequest(
            model_name="text_encoder",
            requested_output_names=["last_hidden_state"],
            inputs=[input_ids],
        )
        response = encoding_request.exec()
        text_embeddings = pb_utils.get_output_tensor_by_name(
            response, "last_hidden_state"
        )
        text_embeddings = from_dlpack(text_embeddings.to_dlpack()).clone()
        # TODO: Check if this is required
        text_embeddings = text_embeddings.to(self.device)
        return text_embeddings

    def predict_unet(self):
        pass

    def predict_vae(self, latents):
        latents = 1 / 0.18215 * latents
        latents = pb_utils.Tensor("latents", latents.numpy())
        encoding_request = pb_utils.InferenceRequest(
            model_name="vae",
            requested_output_names=["image"],
            inputs=[latents],
        )
        response = encoding_request.exec()
        image = pb_utils.get_output_tensor_by_name(response, "image")
        image = from_dlpack(image.to_dlpack()).clone()
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        layout=torch.strided,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, dtype=dtype, layout=layout)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def execute(self, requests):
        responses = []
        for request in requests:
            prompts = [
                p.decode()
                for p in pb_utils.get_input_tensor_by_name(request, "prompt")
                .as_numpy()
                .tolist()
            ]
            height = pb_utils.get_input_tensor_by_name(request, "height").as_numpy()[0]
            width = pb_utils.get_input_tensor_by_name(request, "width").as_numpy()[0]
            num_inference_steps = pb_utils.get_input_tensor_by_name(
                request, "inference_steps"
            ).as_numpy()[0]

            batch_size = len(prompts)
            text_input_ids = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            uncond_tokens = [""] * batch_size
            uncond_input_ids = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=text_input_ids.shape[1],
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = self.predict_text_encoder(text_input_ids)
            negative_prompt_embeds = self.predict_text_encoder(uncond_input_ids)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            self.scheduler.set_timesteps(num_inference_steps)

            latents = self.prepare_latents(
                batch_size,
                self.unet_inchannels,
                height,
                width,
                prompt_embeds.dtype,
            )
            """
            timesteps = self.scheduler.timesteps
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            for i, t in tqdm(enumerate(timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                # UNET (TODO)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            """
            image = self.predict_vae(latents)

            # Sending results
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "image",
                        np.array(image, dtype=self.output0_dtype),
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("Cleaning up...")
        del self.tokenizer
        del self.scheduler
