# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import importlib
import os
import traceback

import torch
from omegaconf import OmegaConf

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)

###############################################################################
#                              CUSTOM START
###############################################################################

DEFAULT_INSTRUCT_IP2P_CLASS = "StableDiffusionInstructPix2PixPipeline"

###############################################################################
#                              CUSTOM END
###############################################################################


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the InstructPix2Pix .ckpt to convert.",
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "Path to the YAML config file for InstructPix2Pix. "
            "It should define unet_config with 'in_channels=8' or similar."
        ),
    )
    parser.add_argument(
        "--config_files",
        default=None,
        type=str,
        help=(
            "Additional config files if needed. Generally not required for InstructPix2Pix, "
            "unless you have separate partial configs."
        ),
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help=(
            "Force the number of input channels. For InstructPix2Pix it's often 8. "
            "If None, the script tries to detect from the config."
        ),
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="One of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm'].",
    )
    parser.add_argument(
        "--pipeline_type",
        default=None,
        type=str,
        help=(
            "If needed, specify 'FrozenCLIPEmbedder' or similar. Often not strictly required for IP2P. "
            "If None, script attempts to infer automatically."
        ),
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help="Image resolution used in training. Usually 512 for older IP2P, but confirm your config.",
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help="Typically 'epsilon' for SD1.x-based models (including InstructPix2Pix). If v2-based, might be 'v_prediction'.",
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help="Use EMA weights if present in the ckpt. If not present, script might fail or ignore.",
    )
    parser.add_argument(
        "--upcast_attention",
        action="store_true",
        help="For certain SD2.1 or above. Usually not needed for IP2P 1.x, but optional.",
    )
    parser.add_argument(
        "--from_safetensors",
        action="store_true",
        help="Set if your ckpt is in safetensors format.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Save output pipeline in safetensors format.",
    )
    parser.add_argument(
        "--dump_path",
        default=None,
        type=str,
        required=True,
        help="Output directory for the converted Diffusers pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load checkpoint onto (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--stable_unclip",
        type=str,
        default=None,
        help="Ignore unless you're dealing with stable unCLIP. Typically not used for IP2P.",
    )
    parser.add_argument(
        "--stable_unclip_prior",
        type=str,
        default=None,
        help="Ignore unless stable unCLIP txt2img. Typically not used for IP2P.",
    )
    parser.add_argument(
        "--clip_stats_path",
        type=str,
        default=None,
        help="Ignore for IP2P. For unCLIP only.",
    )
    parser.add_argument(
        "--controlnet",
        action="store_true",
        default=None,
        help="For ControlNet checkpoints, not typical for InstructPix2Pix.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Convert weights to float16 for smaller file size & GPU usage.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="If you have a pre-converted VAE or want a custom VAE path, specify here. Otherwise ignored.",
    )
    parser.add_argument(
        "--pipeline_class_name",
        type=str,
        default=DEFAULT_INSTRUCT_IP2P_CLASS,  # <--- Default to IP2P Pipeline
        required=False,
        help=(
            "Which pipeline class to use. For InstructPix2Pix, typically 'StableDiffusionInstructPix2PixPipeline'. "
            "If not specified, we default to that."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1) load config from your original YAML to detect in_channels=8 if not provided
    # ------------------------------------------------------------------
    in_channels_final = args.num_in_channels

    if in_channels_final is None:
        if os.path.isfile(args.original_config_file):
            config_ = OmegaConf.load(args.original_config_file)
            try:
                in_channels_final = config_.model.params.unet_config.params.in_channels
                print(f"[INFO] Detected in_channels={in_channels_final} from {args.original_config_file}")
            except Exception:
                print("[WARN] Could not find `in_channels` in config. Default to 8.")
                in_channels_final = 8
        else:
            in_channels_final = 8
            print("[WARN] original_config_file not found. Setting in_channels=8 by default for IP2P.")

    # ------------------------------------------------------------------
    # 2) Load pipeline_class by name
    # ------------------------------------------------------------------
    library = importlib.import_module("diffusers")
    class_obj = getattr(library, args.pipeline_class_name, None)
    if class_obj is None:
        raise ValueError(
            f"Cannot find pipeline class `{args.pipeline_class_name}` in diffusers. Make sure the name is correct."
        )
    pipeline_class = class_obj
    print(f"[INFO] Using pipeline_class={pipeline_class.__name__}")
    print(f"[INFO] pipeline_type={args.pipeline_type}, in_channels={in_channels_final}")

    # ------------------------------------------------------------------
    # 3) Actually call the original function with debug
    # ------------------------------------------------------------------
    try:
        print("[DEBUG] Starting download_from_original_stable_diffusion_ckpt...")
        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=args.checkpoint_path,
            original_config_file=args.original_config_file,
            config_files=args.config_files,
            image_size=args.image_size,
            prediction_type=args.prediction_type,
            model_type=args.pipeline_type,
            extract_ema=args.extract_ema,
            scheduler_type=args.scheduler_type,
            num_in_channels=in_channels_final,  # <--- KEY PART: pass our forced in_channels=8
            upcast_attention=args.upcast_attention,
            from_safetensors=args.from_safetensors,
            device=args.device,
            stable_unclip=args.stable_unclip,
            stable_unclip_prior=args.stable_unclip_prior,
            clip_stats_path=args.clip_stats_path,
            controlnet=args.controlnet,
            vae_path=args.vae_path,
            pipeline_class=pipeline_class,
        )
        print("[DEBUG] Finished download_from_original_stable_diffusion_ckpt.")
    except Exception as e:
        print("[ERROR] Exception during download_from_original_stable_diffusion_ckpt:")
        traceback.print_exc()
        return  # Stop script here

    # ------------------------------------------------------------------
    # 4) Optional: check if unet, vae, text_encoder REALLY have data
    # ------------------------------------------------------------------
    if not hasattr(pipe, "unet"):
        print("[ERROR] `pipe` has no unet. Conversion likely failed.")
    else:
        try:
            unet_weight = pipe.unet.conv_in.weight
            print("[DEBUG] unet.conv_in.weight shape:", unet_weight.shape, "device:", unet_weight.device)
            # We can also check sum or something to see if it's not a meta tensor
            print("[DEBUG] unet.conv_in.weight sum:", unet_weight.sum().item())
        except Exception as e:
            print("[ERROR] Could not access unet weights. Possibly a meta tensor or mismatch.")
            traceback.print_exc()

    if not hasattr(pipe, "vae"):
        print("[WARN] `pipe` has no vae. Maybe it wasn't in the checkpoint?")
    else:
        try:
            vae_weight = pipe.vae.encoder.conv_in.weight
            print("[DEBUG] vae.encoder.conv_in.weight shape:", vae_weight.shape, "device:", vae_weight.device)
            print("[DEBUG] vae.encoder.conv_in.weight sum:", vae_weight.sum().item())
        except Exception as e:
            print("[WARN] Could not access vae weights.")
            traceback.print_exc()

    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        try:
            some_text_enc_param = next(pipe.text_encoder.parameters())
            print("[DEBUG] text_encoder param shape:", some_text_enc_param.shape)
            print("[DEBUG] text_encoder param sum:", some_text_enc_param.sum().item())
        except Exception as e:
            print("[WARN] Could not read text_encoder weights.")
            traceback.print_exc()
    else:
        print("[WARN] No text_encoder in pipeline, or it's None.")

    # ------------------------------------------------------------------
    # 5) If half-precision requested
    # ------------------------------------------------------------------
    if args.half:
        print("[DEBUG] Casting pipeline to float16...")
        pipe.to(dtype=torch.float16)

    # ------------------------------------------------------------------
    # 6) Save the pipeline
    # ------------------------------------------------------------------
    try:
        if args.controlnet:
            pipe.controlnet.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
            print(f"[INFO] ControlNet weights saved at: {args.dump_path}")
        else:
            pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
            print(f"[INFO] InstructPix2Pix pipeline saved at: {args.dump_path}")
    except Exception as e:
        print("[ERROR] Exception during pipe.save_pretrained:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
