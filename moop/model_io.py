"""Model loading and the inference pieces the importance and evaluation code share.

`build_inputs` / `run_rollout` replicate the release inference path of `alpamayo1_5` for one
sample and one trajectory (b=1, n=1); `denoise_with_cache` runs the expert's 10-step flow-matching
sampler on an existing VLM KV cache.
"""

import torch
from alpamayo1_5 import helper
from alpamayo1_5.models.token_utils import to_special_token

from moop import MODEL_ID, MODEL_REV


def load_base_model(device="cuda"):
    """Dense Alpamayo 1.5 at the pinned revision, frozen, with its processor."""
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = Alpamayo1_5.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, helper.get_processor(model.tokenizer)


def set_attn_impl(model, impl="sdpa"):
    """Attention implementation for the VLM (text + vision) and the expert."""
    model.vlm.config._attn_implementation = impl
    model.vlm.config.text_config._attn_implementation = impl
    model.vlm.config.vision_config._attn_implementation = impl
    model.expert.config._attn_implementation = impl


def build_inputs(model, processor, data, device):
    """Fused input_ids (camera frames + ego-history trajectory tokens) and tokenized data."""
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)
    tokenized = model_inputs["tokenized_data"]
    input_ids = tokenized.pop("input_ids")  # (1, T_prompt)
    traj = {"ego_history_xyz": model_inputs["ego_history_xyz"],
            "ego_history_rot": model_inputs["ego_history_rot"]}
    input_ids = model.fuse_traj_tokens(input_ids, traj)  # (1, T_prompt)
    return {
        "input_ids": input_ids,
        "tokenized_data": dict(tokenized),  # pixel_values, image_grid_thw, attention_mask
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }


@torch.no_grad()
def run_rollout(model, inputs, temperature=0.6, top_p=0.98, max_generation_length=256):
    """Autoregressive Chain-of-Causation rollout (one sample). Returns sequences and the EOS position."""
    from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
    from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    input_ids = inputs["input_ids"]
    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    stopping = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
    processors = LogitsProcessorList([ExpertLogitsProcessor(
        traj_token_offset=model.config.traj_token_start_idx,
        traj_vocab_size=model.config.traj_vocab_size)])
    out = model.vlm.generate(input_ids=input_ids, generation_config=generation_config,
                             stopping_criteria=stopping, logits_processor=processors,
                             **inputs["tokenized_data"])
    sequences = replace_padding_after_eos(token_ids=out.sequences, eos_token_id=eos_token_id,
                                          pad_token_id=model.tokenizer.pad_token_id)  # (1, T_total)
    eos_mask = sequences[0] == eos_token_id
    eos_pos = int(eos_mask.int().argmax().item()) if eos_mask.any() else sequences.shape[1] - 1
    return {
        "sequences": sequences,
        "eos_pos": eos_pos,  # position of <|traj_future_start|>
        "prompt_len": input_ids.shape[1],
        "past_key_values": out.past_key_values,
        "rope_deltas": model.vlm.model.rope_deltas,
    }


def denoise_with_cache(model, prompt_cache, rope_deltas, offset, prefix_mask, seed=42):
    """10-step flow-matching sampling on a given VLM cache (b=1). Returns the action (1, 64, 2)."""
    device = prefix_mask.device
    n_tok = model.action_space.get_action_space_dims()[0]  # 64
    prefill = prompt_cache.get_seq_length()
    position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
        offset=offset, rope_deltas=rope_deltas, kv_cache_seq_len=prefill,
        n_diffusion_tokens=n_tok, b_star=1, device=device, prefix_mask=prefix_mask,
    )
    forward_kwargs = {"is_causal": False} if model.config.expert_non_causal_attention else {}

    def step_fn(x, t):
        # x: (1, 64, 2), t: (1, 1, 1)
        embeds = model.action_in_proj(x, t)  # (1, 64, 2048)
        if embeds.dim() == 2:
            embeds = embeds.view(1, n_tok, -1)
        out = model.expert(inputs_embeds=embeds, position_ids=position_ids, past_key_values=prompt_cache,
                           attention_mask=attention_mask, use_cache=True, **forward_kwargs)
        prompt_cache.crop(prefill)
        last = out.last_hidden_state[:, -n_tok:]  # (1, 64, 2048)
        return model.action_out_proj(last).view(-1, *model.action_space.get_action_space_dims())

    torch.cuda.manual_seed_all(seed)
    return model.diffusion.sample(batch_size=1, step_fn=step_fn, device=device,
                                  return_all_steps=False)  # (1, 64, 2)


def cache_layer_kv(cache, i):
    """(keys, values) of cache layer i."""
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def set_cache_layer_kv(cache, i, k, v):
    if hasattr(cache, "layers"):
        cache.layers[i].keys = k
        cache.layers[i].values = v
    else:
        cache.key_cache[i] = k
        cache.value_cache[i] = v


def gt_actions(model, data, device):
    """Ground-truth future trajectory in the model's normalized action space. (1, 64, 2)."""
    hx = data["ego_history_xyz"][:, 0].to(device)  # (1, 16, 3)
    hr = data["ego_history_rot"][:, 0].to(device)  # (1, 16, 3, 3)
    fx = data["ego_future_xyz"][:, 0].to(device)  # (1, 64, 3)
    fr = data["ego_future_rot"][:, 0].to(device)  # (1, 64, 3, 3)
    return model.action_space.traj_to_action(hx.float(), hr.float(), fx.float(), fr.float())
