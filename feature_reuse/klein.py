from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2Klein
from mflux.utils.image_util import ImageUtil

from feature_reuse.reuse import reuse_blocks
from feature_reuse.samplers import make_sampler

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "klein-base-4b-q4"
SCHEDULER = "flow_match_euler_discrete"
SIZE = 512
GUIDANCE = 3.5


@dataclass(frozen=True)
class Denoised:
    latents: mx.array
    seconds: float
    full_steps: int
    rejected_steps: int
    cache_bytes: int
    checked_steps: int = 0


def load_klein() -> Flux2Klein:
    model = Flux2Klein(
        model_config=ModelConfig.flux2_klein_base_4b(),
        model_path=str(MODEL_PATH),
        compute_precision=mx.float16,
    )
    mx.set_cache_limit(2 * 1000**3)
    if TilingConfig.may_tile_implicitly(model):
        model.tiling_config = TilingConfig()
    return model


def encode_prompt(model: Flux2Klein, prompt: str) -> tuple[mx.array, ...]:
    embeds = model._encode_prompt_pair(prompt=prompt, negative_prompt=" ", guidance=GUIDANCE)
    mx.eval(*embeds)
    return embeds


def denoise(
    model: Flux2Klein, embeds: tuple[mx.array, ...], seed: int, steps: int, policy=None, sampler: str = "euler"
) -> Denoised:
    prompt, text_ids, negative_prompt, negative_text_ids = embeds
    config = Config(
        model_config=model.model_config,
        num_inference_steps=steps,
        height=SIZE,
        width=SIZE,
        guidance=GUIDANCE,
        scheduler=SCHEDULER,
    )
    scheduler = config.scheduler
    solver = make_sampler(sampler, scheduler.sigmas)
    latents, latent_ids, _, _ = Flux2LatentCreator.prepare_packed_latents(
        seed=seed, height=SIZE, width=SIZE, batch_size=1
    )
    mx.eval(latents, latent_ids)
    seconds = 0.0

    with reuse_blocks(model.transformer, policy) as reuse:

        def run(branch: str, encoder_hidden_states: mx.array, ids: mx.array) -> mx.array:
            reuse.branch = branch
            return model.transformer(
                hidden_states=latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=scheduler.timesteps[step],
                img_ids=latent_ids,
                txt_ids=ids,
                guidance=None,
            )

        def predict() -> mx.array:
            noise = run("cond", prompt, text_ids)
            negative_noise = run("uncond", negative_prompt, negative_text_ids)
            return negative_noise + GUIDANCE * (noise - negative_noise)

        for step in range(steps):
            start = perf_counter()
            reuse.begin_step(step)
            noise = predict()
            if not reuse.confirm_step():
                noise = predict()
            latents = solver.step(noise, step, latents)
            mx.eval(latents, *reuse.arrays())
            seconds += perf_counter() - start

        cache_bytes = sum(array.nbytes for array in reuse.arrays())
        return Denoised(latents, seconds, reuse.full_steps, reuse.rejected_steps, cache_bytes, reuse.checked_steps)


def decode(model: Flux2Klein, latents: mx.array):
    size = SIZE // 16
    packed = latents.reshape(latents.shape[0], size, size, latents.shape[-1]).transpose(0, 3, 1, 2)
    return ImageUtil.to_pil(model.vae.decode_packed_latents(packed, tiling_config=model.tiling_config))
