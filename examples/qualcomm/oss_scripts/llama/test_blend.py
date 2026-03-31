
# usage:
# export PYTHONUNBUFFERED=1
# export SOC_MODEL=8650
# python examples/qualcomm/oss_scripts/llama/test_blend.py -b build-android -c -m ${SOC_MODEL} --temperature 0 --model_mode blender --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen3-1_7b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1 > debug.txt 2>&1

import argparse
import inspect
import json
import logging
import types

from functools import partial
from typing import Any, Dict, List

import torch

from executorch.backends.qualcomm._passes import FoldQDQ, I64toI32, TagQuantIO
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_prefill_kv_output,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    BLENDER_DECODER_GRAPH_NAMES,
    DECODER_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    TEXT_EMBEDDING_GRAPH_NAMES,
    TEXT_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    graph_module_inference,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import TextEmbedding
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    get_model_specific_kwargs,
    is_node_src_start_with_name,
    log_info,
    Mode,
    process_model_args,
    Processor,
    Request,
)
from executorch.examples.qualcomm.utils import make_quantizer
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from torchao.prototype.spinquant import apply_spinquant
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoModel

import json
import logging
import os
import sys
from multiprocessing.connection import Client
from typing import Dict

import torch

from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLMModelConfig,
    SUPPORTED_LLM_MODELS,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset import DatasetBuilder
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    ATTENTION_SINK_EVICTOR,
    AUDIO_ENCODER,
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    EVAL_MODE,
    PROMPT_EVAL,
    SQNR_EVAL,
    TASKS_EVAL,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    TEXT_EMBEDDING_GRAPH_NAMES,
    TEXT_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_runtime_evaluator import (
    DefaultEval,
    SqnrEval,
    TaskEval,
)

from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import (
    HybridAttentionSinkEvictor,
    is_attention_sink_config_equal,
    MultiModalManager,
    next_power_of_two,
)
from executorch.examples.qualcomm.utils import setup_common_args_and_variables
from torchao.quantization.utils import compute_error

from executorch.examples.qualcomm.oss_scripts.llama.llama import _build_parser
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import TextDecoder

from pytorch_tokenizers.hf_tokenizer import HuggingFaceTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer
from pytorch_tokenizers.tiktoken import TiktokenTokenizer

from executorch.examples.qualcomm.oss_scripts.llama.model import (
    FeedForward_REGISTRY,
    NORM_REGISTRY,
    ROTARY_EMB_REGISTRY,
)

import scipy
import math

kv_store = []
reuse_thres = 6
inter_compute_num = 8

k_caches_total = []
v_caches_total = []


def quant(self, tokenizer, datasets):
    if self.quant_recipe is None:
        return

    if self.decoder is None or (
        self.apply_embedding and self.tok_embedding is None
    ):
        return

    # check bit width graph io
    fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
    if self.quant_recipe.get_kv_io_bit_width() == 8:
        fixed_point_type["kv_type"] = torch.uint8
    elif self.quant_recipe.get_kv_io_bit_width() == 16:
        fixed_point_type["kv_type"] = torch.uint16
    else:
        raise RuntimeError(
            f"unknown kv io bit width {self.quant_recipe.get_kv_io_bit_width()}"
        )

    if self.quant_recipe.get_logits_output_bit_width() == 16:
        fixed_point_type["io_type"] = torch.uint16
    else:
        raise RuntimeError(
            f"unknown logits io bit width {self.quant_recipe.get_logits_output_bit_width()}"
        )

    quantizer = make_quantizer()
    custom_annotation = tuple()
    for custom_annotation in custom_annotation:
        self.quant_recipe.recipe.custom_quant_annotations.append(custom_annotation)
    quantizer.recipe = self.quant_recipe

    text_embedding_quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a8w,
        per_channel_conv=True,
        per_channel_linear=True,
        act_observer=MinMaxObserver,
    )

    with torch.no_grad():
        # prepare tok embedding model for ptq
        if self.apply_embedding:
            self.tok_embedding = torch.export.export(
                self.tok_embedding,
                self.tok_embedding.get_example_input(),
                strict=True,
            ).module()

        # prepare decoder model for ptq
        self.decoder = torch.export.export(
            self.decoder, self.export_input, strict=True
        ).module()
        self.decoder = prepare_pt2e(self.decoder, quantizer)
        if self.apply_embedding:
            self.tok_embedding = prepare_pt2e(
                self.tok_embedding, text_embedding_quantizer
            )

        # start calibration
        self._calibrate(
            model=self.decoder,
            tokenizer=tokenizer,
            event="prepare_pt2e",
            user_calibration_data=datasets,
            tok_embedding=self.tok_embedding,
            intermediate_outputs=[],
        )
        self.decoder = convert_pt2e(self.decoder)

        # Saving Decode QDQ Model EP for SQNR evaluation
        if self.mode == Mode.DECODE:
            qdq_ep = torch.export.export(
                self.decoder, self.export_input, strict=True
            )
            qdq_ep_path = f"{self.control_args.artifact}/{DECODE_QDQ_FILENAME}"
            torch.export.save(qdq_ep, qdq_ep_path)
            logging.info(f"QDQ EP saved to {qdq_ep_path}")

        if self.apply_embedding:
            self.tok_embedding = convert_pt2e(self.tok_embedding)

        if self.control_args.verbose:
            if self.apply_embedding:
                image_embedding = []
            self._calibrate(
                model=self.decoder,
                tokenizer=tokenizer,
                event="convert_pt2e",
                user_calibration_data=datasets,
                tok_embedding=self.tok_embedding,
                intermediate_outputs=image_embedding,
            )

    
def store_kv(tokens, start_pos, k_caches, v_caches):
    kv_store.append({
        "tokens": tokens,
        "ori_pos": start_pos,
        "k_caches": [k[:,:,:,:len(tokens)] for k in k_caches],
        "v_caches": [v[:,:,:len(tokens),:] for v in v_caches]
    })

def rerotate_kv(ori_pos, new_pos, 
                freqs_cos, freqs_sin,
                k_caches, v_caches,
                matched_len, enable_r3=True,
                partial_rotary_factor=1.0):
    if partial_rotary_factor < 1:
        apply_rope_emb = ROTARY_EMB_REGISTRY["partial"]
    else:
        apply_rope_emb = ROTARY_EMB_REGISTRY["default"]
    # compute rerotation matrices
    original_freqs_cos = freqs_cos.narrow(0, ori_pos, matched_len)
    original_freqs_sin = freqs_sin.narrow(0, ori_pos, matched_len)
    new_freqs_cos = freqs_cos.narrow(0, new_pos, matched_len)
    new_freqs_sin = freqs_sin.narrow(0, new_pos, matched_len)
    rerotation_cos = (
        new_freqs_cos * original_freqs_cos + new_freqs_sin * original_freqs_sin
    )
    rerotation_sin = (
        new_freqs_sin * original_freqs_cos - new_freqs_cos * original_freqs_sin
    )
    # rerotate kv, resolve spinquant
    rerotated_k = []
    if enable_r3:
        head_dim = k_caches[0].shape[2]
        r3_weight = torch.tensor(
            scipy.linalg.hadamard(head_dim, dtype=float)
            / math.sqrt(head_dim),
            dtype=torch.float32,
            device="cpu",
        )
    for k in k_caches:
        k = k.transpose(2, 3)
        if enable_r3:
            k = torch.matmul(k, r3_weight.T)
        k = apply_rope_emb(k, rerotation_cos, rerotation_sin)
        if enable_r3:
            k = torch.matmul(k, r3_weight)
        rerotated_k.append(k.transpose(2,3))
    rerotated_v = v_caches
    return rerotated_k, rerotated_v

def lcs_pos(a, b):
    """
    output: (start_in_a, start_in_b, length)
    """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    max_len = 0
    end_a = 0
    end_b = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_a = i
                    end_b = j
    return end_a-max_len, end_b-max_len, max_len

def load_kv(tokens, start_pos, 
            freqs_cos, freqs_sin):
    tokens_list = [tokens]
    pos_list = [start_pos]
    matched_kv = []
    for kv in kv_store:
        if len(tokens_list) == 0:
            break
        matched_len = 0
        for i, (tokens, start_pos) in enumerate(zip(tokens_list, pos_list)):
            # Currently, only longest substring is matched, fairly enough.
            ori_start_offset, new_start_offset, matched_len = lcs_pos(kv["tokens"], tokens)
            # ori_start_pos and new_start_pos are absolute pos, used only in rerotation.
            ori_start_pos = kv["ori_pos"] + ori_start_offset
            new_start_pos = start_pos + new_start_offset
            if matched_len > reuse_thres:
                break # matched
        if matched_len > reuse_thres:
            rerotated_k, rerotated_v = rerotate_kv(ori_start_pos, new_start_pos,
                        freqs_cos, freqs_sin,
                        [k.narrow(3, ori_start_offset, matched_len) for k in kv["k_caches"]],
                        [v.narrow(2, ori_start_offset, matched_len) for v in kv["v_caches"]], 
                        matched_len)    
            matched_kv.append({"tokens": tokens[new_start_offset:new_start_offset+matched_len],
                               "ori_start_pos": ori_start_pos, "new_start_pos": new_start_pos, 
                               "k_caches": rerotated_k, "v_caches": rerotated_v})
            tokens_list.pop(i)
            pos_list.pop(i)
            if matched_len == len(tokens):
                continue
            pre_tokens = tokens[:new_start_offset]
            pre_start_pos = start_pos
            suf_tokens = tokens[new_start_offset+matched_len:]
            suf_start_pos = start_pos + new_start_offset + matched_len
            # insert suf first, so that suf would be behind pre.
            if len(suf_tokens) > 0:
                tokens_list.insert(i, suf_tokens)
                pos_list.insert(i, suf_start_pos)
            if len(pre_tokens) > 0:
                tokens_list.insert(i, pre_tokens)
                pos_list.insert(i, pre_start_pos)
    matched_kv = sorted(matched_kv, key=lambda x: x["new_start_pos"])
    return matched_kv

def blend_kv(tokens, start_pos, 
             freqs_cos, freqs_sin,
             use_blend=False,
             merge_blend=False):
    if not use_blend:
        return [{
            "tokens": tokens, 
            "type": Mode.PREFILL,
            "start_pos": start_pos,
            "k_caches": None,
            "v_caches": None,
        }]
    # 对于tokens的前缀，能拼进去多少就拼多少进去！
    matched_kv = load_kv(tokens, start_pos, 
                         freqs_cos, freqs_sin)
    # add a dummy node for the following iteration
    matched_kv.append({"tokens": torch.tensor([]), "ori_start_pos": 0, "new_start_pos": len(tokens)})
    inputs_kv = []

    # 不考虑完整命中.
    # 成果：tokens, type (prefill/blend), start_pos, k_caches, v_caches, valid_mask
    # Version 0: no inter-reuse prefill
    if not merge_blend:
        # put t together
        for matched in matched_kv:
            # only both ori and new being prefix can be prefill, otherwise make it blend.
            # 1. ori_start_pos==0 and new_start_pos==0, prefix caching: direct reuse, merge to subsequent blend/prefill.
            # 2. last_end_pos != next_start_pos, missed tokens: type prefill.
            # 3. otherwise, type blend.
            pre_inputs = None
            if len(inputs_kv) > 0:
                pre_inputs = inputs_kv[-1]
            if matched["ori_start_pos"]==0 and matched["new_start_pos"]==0:
                inputs = {
                    "tokens": matched["tokens"], 
                    "type": None, # preload
                    "start_pos": matched["new_start_pos"],
                    "k_caches": matched["k_caches"],
                    "v_caches": matched["v_caches"],
                }                 
                inputs_kv.append(inputs)
                continue         
            else:
                # determine type
                last_end_pos = pre_inputs["start_pos"] + len(pre_inputs["tokens"])
                next_start_pos = matched["new_start_pos"]
                if last_end_pos < next_start_pos:
                    # missed tokens
                    inputs = {
                        "tokens": tokens[last_end_pos:next_start_pos], 
                        "type": Mode.PREFILL,
                        "start_pos": last_end_pos,
                        "k_caches": None,
                        "v_caches": None,
                    }
                    inputs_kv.append(inputs)
                # update previous element
                if len(inputs_kv) > 0:
                    pre_inputs = inputs_kv[-1]
                if len(matched["tokens"]) == 0 and matched["new_start_pos"] == len(tokens):
                    # dummy node
                    break
                # blend
                # can be merged with previous blend.
                if pre_inputs["type"] is not None and pre_inputs["type"] == Mode.BLENDER:
                    inputs = {
                        "tokens": torch.cat([pre_inputs["tokens"], matched["tokens"]]), 
                        "type": Mode.BLENDER,
                        "start_pos": matched["new_start_pos"],
                        "k_caches": [torch.cat([pre_k, k], dim=3)
                                     for pre_k, k in zip(pre_inputs["k_caches"], matched["k_caches"])],
                        "v_caches": [torch.cat([pre_v, v], dim=2)
                                     for pre_v, v in zip(pre_inputs["v_caches"], matched["v_caches"])],
                        "valid_mask": torch.ones([len(matched["tokens"])])
                    }
                    inputs_kv[-1] = inputs
                else:
                    inputs = {
                        "tokens": matched["tokens"], 
                        "type": Mode.BLENDER,
                        "start_pos": matched["new_start_pos"],
                        "k_caches": matched["k_caches"],
                        "v_caches": matched["v_caches"],
                        "valid_mask": torch.ones([len(matched["tokens"])])
                    }
                    inputs_kv.append(inputs)
    else:
        raise NotImplementedError("MergeBlend not supported yet!")
    return inputs_kv

def print_inputs(inputs_kv):
    print([{
        "tokens": inputs["tokens"],
        "type": inputs["type"],
        "start_pos": inputs["start_pos"]
    } for inputs in inputs_kv])

def _prefill(prefiller,
             blender,
             prompt,
             tokenizer,
             ar_len=128,
             blend_len=None,
             start_pos=0,
             k_caches=None,
             v_caches=None
             ):
    prefill_module = prefiller.decoder
    blend_module = blender.decoder
    if isinstance(prompt, str):
        # Llama2 tokenizer has no special tokens
        if isinstance(tokenizer, (SentencePieceTokenizer, HuggingFaceTokenizer)):
            prompt_token_list = tokenizer.encode(prompt, bos=True, eos=False)
        elif isinstance(tokenizer, TiktokenTokenizer):
            prompt_token_list = tokenizer.encode(
                prompt, bos=True, eos=False, allowed_special="all"
            )

    # record total input tokens and generated tokens
    total_token_list = prompt_token_list
    context_len = prefiller.meta["get_max_context_len"]
    
    # 3. prepare decoder inputs
    use_blend = (blend_len is not None)
    freqs_cos, freqs_sin = prefill_module.freqs_cos, prefill_module.freqs_sin
    inputs_kv = blend_kv(total_token_list, start_pos, freqs_cos, freqs_sin, use_blend=use_blend)
    if start_pos != 0:
        assert (k_caches is not None) and (v_caches is not None)
        for i in range(len(inputs_kv)):
            inputs[i]["start_pos"] += start_pos
        pre_k_caches = k_caches
        pre_v_caches = v_caches

    if k_caches is None:
        if blend_len is None:
            tokens, _, _, k_caches, v_caches = prefill_module.get_example_inputs()
        else:
            tokens, _, _, k_caches, v_caches, _ = blend_module.get_example_inputs()   
 
    if start_pos != 0:
        for layer in len(k_caches):
            k_caches[layer][:,:,:,:start_pos] = pre_k_caches[layer][:,:,:,:start_pos]
            v_caches[layer][:,:,:start_pos,:] = pre_v_caches[layer][:,:,:start_pos,:]
    print("input_kv: ")
    print_inputs(inputs_kv)

    # iteratively input
    for inputs in inputs_kv:
        start_pos = inputs["start_pos"]
        end_pos = start_pos + len(inputs["tokens"])
        if inputs["type"] is None:
            # preload
            for layer in range(len(k_caches)):
                k_caches[layer][:,:,:,start_pos:end_pos] = inputs["k_caches"][layer]
                v_caches[layer][:,:,start_pos:end_pos,:] = inputs["v_caches"][layer]
            continue
        # if inputs["type"] == Mode.PREFILL:
        if inputs["type"] == Mode.PREFILL:
            # prefill: make a copy of rearranged kv_caches
            # rearrange kv
            _, _, _, prefill_k_caches, prefill_v_caches = prefill_module.get_example_inputs()
            for layer in range(len(k_caches)):
                prefill_k_caches[layer][:,:,:,:start_pos] = k_caches[layer][:,:,:,:start_pos]
                prefill_v_caches[layer][:,:,:start_pos,:] = v_caches[layer][:,:,:start_pos,:]
            # else:
            #     prefill_k_caches = k_caches
            #     prefill_v_caches = v_caches
            # compute
            _, next_token, prefill_k_caches, prefill_v_caches = \
                _compute_prefill(prefill_module, tokens, inputs["tokens"],
                                 start_pos, end_pos, ar_len, context_len, 
                                 prefill_k_caches, prefill_v_caches)
            # rearrange kv
            for layer in range(len(k_caches)):
                k_caches[layer][:,:,:,:end_pos] = prefill_k_caches[layer][:,:,:,:end_pos]
                v_caches[layer][:,:,:end_pos,:] = prefill_v_caches[layer][:,:,:end_pos,:]
            continue
        if inputs["type"] == Mode.BLENDER:
            # blend: concatenate precomputed kv to the last few tokens of kv_caches
            # reuse kv
            # recompute kv
            _, next_token, k_caches, v_caches = \
                _compute_prefill(blend_module, tokens, inputs["tokens"],
                                 start_pos, end_pos, ar_len, context_len, 
                                 k_caches, v_caches, 
                                 inputs["k_caches"], inputs["v_caches"],
                                 blend_len, inputs["valid_mask"])
            continue
    return total_token_list, next_token, k_caches, v_caches

def _compute_prefill(module,
                     tokens, total_token_list,
                     prompt_start_pos, prompt_end_pos,
                     ar_len, context_len, 
                     k_caches, v_caches,
                     precomputed_k=None, precomputed_v=None,
                     blend_len=None,  valid_mask_raw=None):
    start_pos = prompt_start_pos
    end_pos = min(prompt_end_pos, start_pos+ar_len)
    while(start_pos < prompt_end_pos):
        input_ids = total_token_list[(start_pos - prompt_start_pos):(end_pos - prompt_start_pos)]
        tokens[:, :(end_pos-start_pos)] = torch.tensor(input_ids)
        dtype = torch.int32
        all_pos = torch.arange(ar_len, dtype=dtype).reshape(1, -1) + start_pos
        pos_val = 0.
        neg_val = -255.
        """
cols:   0    1    2  |   3    4    5  |   6    7    8 
      ------------------------------------------------
        0    0    0  | -255 -255 -255 |   0  -255 -255
        0    0    0  | -255 -255 -255 |   0    0  -255
      -255 -255 -255 | -255 -255 -255 | -255 -255 -255
        """
        mask = torch.full((1,ar_len,context_len), neg_val)
        mask[:,:(end_pos-start_pos),:start_pos] = pos_val
        causal = torch.triu(torch.full((ar_len,ar_len), neg_val), 1)
        mask[:,:,context_len-ar_len:] = causal
        mask[:,(end_pos-start_pos):,:] = neg_val
        valid_mask = torch.zeros((1, ar_len), dtype=torch.int32)
        if valid_mask_raw is not None:
            valid_mask[:, :len(valid_mask_raw)] = valid_mask_raw
        else:
            valid_mask[:] = (torch.arange(ar_len) < (end_pos-start_pos))
        if precomputed_k is not None and precomputed_v is not None:
            precomputed_start = start_pos - prompt_start_pos
            precomputed_end = end_pos - prompt_start_pos
            kv_load_start = -ar_len
            kv_load_end = -ar_len+(end_pos-start_pos)
            for layer in range(len(k_caches)):
                k_caches[layer][:,:,:,kv_load_start:kv_load_end] = precomputed_k[layer][:,:,:,precomputed_start:precomputed_end]
                v_caches[layer][:,:,kv_load_start:kv_load_end,:] = precomputed_v[layer][:,:,precomputed_start:precomputed_end,:]
        
        # 4. decoder forward
        with torch.no_grad():
            results = module(tokens,
                        mask,
                        all_pos,
                        *k_caches,
                        *v_caches,
                        valid_mask)
            if blend_len is None:
                logits, new_k_caches, new_v_caches = results
            else:
                logits, new_k_caches, new_v_caches, imp_indices = results
                print("imp_indices:", imp_indices)
    
        # update
        for layer in range(len(k_caches)):
            if blend_len is None:
                k_caches[layer][:,:,:,start_pos:start_pos+ar_len] = new_k_caches[layer]
                v_caches[layer][:,:,start_pos:start_pos+ar_len,:] = new_v_caches[layer]
            else:
                # update HKVD only
                k_caches[layer][:,:,:,start_pos:start_pos+ar_len] = k_caches[layer][:,:,:,-ar_len:]
                v_caches[layer][:,:,start_pos:start_pos+ar_len,:] = v_caches[layer][:,:,-ar_len:,:]
                _, n_kv_heads, head_dim, _ = k_caches[layer].shape
                cache_indices = imp_indices.unsqueeze(1).unsqueeze(1).expand(-1, n_kv_heads, head_dim, -1) + start_pos
                k_caches[layer] = k_caches[layer].scatter(dim=3, src=new_k_caches[layer], index=cache_indices)
                v_caches[layer] = v_caches[layer].scatter(dim=2, src=new_v_caches[layer], index=cache_indices.transpose(2, 3))
        # blender shall always blend the last token, so that the generation can be executed smoothly!
        next_token_pos = (end_pos-start_pos - 1) if blend_len is None else -1
        next_token = torch.argmax(logits[:, next_token_pos], dim=-1).item()
        start_pos += ar_len
        end_pos = min(prompt_end_pos, start_pos+ar_len)
        # if (blend_config is None and ar_len > 1):
        #     pickle.dump({"mask": list(*inputs.atten_mask),
        #                  "k": list(*k_caches),
        #                  "v": list(*v_caches),}, debug_fd)
        # debug_fd.close()       
    return total_token_list, next_token, k_caches, v_caches
    
def _generate(decoder, start_pos, next_token, 
              prefill_k, prefill_v, tokenizer):
    module = decoder.decoder
    context_len = decoder.meta["get_max_context_len"]
    generated = [next_token]
    print(tokenizer.decode([next_token]), end="")
    # rearrange kv
    _, _, _, k_caches, v_caches = decoder.get_example_inputs()
    for layer in range(len(k_caches)):
        k_caches[layer][:,:,:,:start_pos] = prefill_k[layer][:,:,:,:start_pos]
        v_caches[layer][:,:,:start_pos,:] = prefill_v[layer][:,:,:start_pos,:]

    while (start_pos < context_len 
        and next_token != tokenizer.eos_id):
        dtype = torch.int32
        tokens = torch.tensor([next_token], dtype=dtype).reshape(1, -1)
        all_pos = torch.tensor([0], dtype=dtype).reshape(1, -1) + start_pos
        pos_val = 0.
        neg_val = -255.
        """
cols:     0  1  2  |  3  4   5  |   6    7   8 
        ---------------------------------------
          0  0  0  |  0  0 -255 | -255 -255  0
        """
        mask = torch.full((1,1,context_len), neg_val)
        mask[:,:,:start_pos] = pos_val
        mask[:,:,-1] = pos_val
        
        # 4. decoder forward
        with torch.no_grad():
            results = module(tokens,
                        mask,
                        all_pos,
                        *k_caches,
                        *v_caches)
            logits, new_k_caches, new_v_caches = results    
    
        # update
        for layer in range(len(k_caches)):
            k_caches[layer][:,:,:,start_pos:start_pos+1] = new_k_caches[layer]
            v_caches[layer][:,:,start_pos:start_pos+1,:] = new_v_caches[layer]
        next_token = torch.argmax(logits.flatten(), dim=-1).item()
        start_pos += 1
        generated.append(next_token)
        print(tokenizer.decode([next_token]), end="")
    return generated

def apply_chat_template(prompt, tokenizer_wrapper,
                        chat_template, system_prompt, no_think=None):
    prompt = (
        tokenizer_wrapper.apply_prompt_template(
            chat_template, prompt, system_prompt
        )
        if chat_template is not None
        else prompt
    )
    if no_think is not None:
        prompt += no_think
    return prompt

# def compare_kv_caches(k_caches, v_caches, compare_len, atol=1e-5):
#     diffs = []
#     for layer, (tk, tv, k, v) in enumerate(zip(k_caches_total, v_caches_total, k_caches, v_caches)):
#         # ----- K cache -----
#         # shape: [B, H, D, S]
#         k_diff = (tk[..., :compare_len] - k[..., :compare_len]).abs() > atol
#         # reduce over H, D -> [B, S]
#         k_diff = k_diff.any(dim=(1, 2))
#         # ----- V cache -----
#         # shape: [B, H, S, D]
#         v_diff = (tv[:, :, :compare_len, :] - v[:, :, :compare_len, :]).abs() > atol
#         # reduce over H, D -> [B, S]
#         v_diff = v_diff.any(dim=(1, 3))
#         # combine K and V
#         total_diff = k_diff | v_diff
#         B_ids, S_ids = torch.where(total_diff)
#         for b, s in zip(B_ids.tolist(), S_ids.tolist()):
#             diffs.append((layer, s))
#             print(diffs[-1])
#     return diffs

def compare_kv_caches(k_caches, v_caches, compare_len, atol=1e-4):
    diffs = []
    for layer, (tk, tv, k, v) in enumerate(zip(k_caches_total, v_caches_total, k_caches, v_caches)):
        # ---- K cache ----
        # shape: [B, H, D, S]
        k_diff = (tk[..., :compare_len] - k[..., :compare_len]).abs() > atol
        idx = torch.where(k_diff)
        for b, h, d, s in zip(*idx):
            diffs.append(("K", layer, h.item(), d.item(), s.item()))
            print(diffs[-1], tk[0, h.item(), d.item(), s.item()], k[0, h.item(), d.item(), s.item()], tk[0, h.item(), d.item(), s.item()] - k[0, h.item(), d.item(), s.item()])
        # ---- V cache ----
        # shape: [B, H, S, D]
        v_diff = (tv[:, :, :compare_len, :] - v[:, :, :compare_len, :]).abs() > atol
        idx = torch.where(v_diff)
        for b, h, s, d in zip(*idx):
            diffs.append(("V", layer, h.item(), d.item(), s.item()))
            print(diffs[-1])
    return diffs


def compile(args, config, 
            tokenizer,
            tokenizer_wrapper,
            chat_template, 
            prompt):
    apply_embedding = False
    os.makedirs(args.artifact, exist_ok=True)
    decode = TextDecoder(args,
            config,
            Mode.DECODE,
            apply_embedding=apply_embedding)
    prefill = TextDecoder(args,
            config,
            Mode.PREFILL,
            apply_embedding=apply_embedding)
    blend = TextDecoder(args,
            config,
            Mode.BLENDER,
            apply_embedding=apply_embedding)
    # quant(prefill, tokenizer, args.prompt)
    # quant(blend, tokenizer, args.prompt)
    # print("prefill graph: ")
    # for node in prefill.decoder.graph.nodes:
    #     print(node.name)
    #     print("  op:", node.op)
    #     print("  target:", node.target)
    #     print("  meta:", node.meta)
    #     print("\n\n\n")
    # print("blender graph: ")
    # for node in blend.decoder.graph.nodes:
    #     print(node.name)
    #     print("  op:", node.op)
    #     print("  target:", node.target)
    #     print("  meta:", node.meta)
    #     print("\n\n\n")

    if True:
        chat_prompt = apply_chat_template(prompt, tokenizer_wrapper,
                                        chat_template, args.system_prompt, no_think="<think></think>")
        p1, p2, p3 = prompt.strip().split("\n")
        chat_p = []
        for p in [p1, p2, p3]:
            chat_p.append(apply_chat_template(p, tokenizer_wrapper,
                                            chat_template, args.system_prompt, no_think="<think></think>"))
        
        # test run prefill and decode
        tokens, next_token, k_caches, v_caches = _prefill(prefill, blend, chat_prompt, tokenizer)
        # k_caches_total.extend(k_caches)
        # v_caches_total.extend(v_caches)
        # store_kv(tokens, 0, k_caches, v_caches)
        _generate(decode, len(tokens), next_token, k_caches, v_caches, tokenizer)
        print()

        # prefill each chunks for blender test
        tokens, next_token, k_caches, v_caches = _prefill(prefill, blend, chat_p[0], tokenizer)
        store_kv(tokens, 0, k_caches, v_caches)
        tokens, next_token, k_caches, v_caches = _prefill(prefill, blend, chat_p[2], tokenizer)
        store_kv(tokens, 0, k_caches, v_caches)
        tokens, next_token, k_caches, v_caches = _prefill(prefill, blend, chat_p[1], tokenizer)
        store_kv(tokens, 0, k_caches, v_caches)
        # test run blender
        tokens, next_token, k_caches, v_caches = _prefill(prefill, blend, chat_prompt, tokenizer, blend_len=32)
        # compare_kv_caches(k_caches, v_caches, len(tokens))
        _generate(decode, len(tokens), next_token, k_caches, v_caches, tokenizer)
        print()
    

def test(args):
    decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
    if args.max_context_len is None:
        args.max_context_len = args.max_seq_len
    print(decoder_model_config)
    print(args.decoder_model)
    # Prepare tokenizer
    tokenizer_wrapper = TokenizerWrapper(
        args,
        decoder_model_config,
    )
    runtime_tokenizer_path, tokenizer, chat_template = (
        tokenizer_wrapper.get_runtime_tokenizer(
            args.tokenizer_model, args.tokenizer_bin
        )
    )

#     prompt = """Lionel Messi scored 13 goals at FIFA World Cups.
# Cristiano scored 8 goals at FIFA World Cups.
# Who scored more goals at FIFA World Cups, Messi or Ronaldo?"""    
# 
    prompt = """某图书馆为了鼓励阅读，推出了一项“阅读积分”活动。活动规则规定：每借阅 一本普通图书可以获得 4 积分，如果是科普类或历史类图书，则每本可以获得 6 积分。活动开始后的第一个月，小李一共借阅了 9 本普通图书，同时还借阅了一些科普类或历史类图书。图书馆系统会自动记录借阅种类并计算积分，读者可以在手机应用中随时查看自己的积分变化。小李希望通过这次活动积累更多积分，因为积分可以在年底兑换文创产品或阅读券。
根据图书馆月底公布的个人阅读统计，小李在第一个月的 阅读积分总共达到了 60 分。统计报告还说明，积分只来自于本月借阅图书的数量与类别，不包含其他奖励活动或额外加分。也就是说，小李本月获得的全部积分，都是通过借阅普通图书以及科普或历史类图书计算得到的。
根据两段信息，小李在这个月借阅了多少本科普类或历史类图书？（需要结合两段中的规则和数字进行计算。）"""

    compile(
        args,
        decoder_model_config,
        tokenizer, tokenizer_wrapper,
        chat_template, prompt,
    )
    
if __name__ == "__main__":
    # only support: 
    sliding_window = None
    partial_rotary_factor = 1.0
    if True:
        parser = _build_parser()
        args = parser.parse_args()
        test(args)
        
    if False:
        # blend_kv unit test
        B, H, S, D = 1, 8, 1024, 128
        store_kv(torch.arange(12), 0, [torch.ones(B,H,D,S)], [torch.ones(B,H,S,D)])
        store_kv(torch.tensor([0,1,2,17,18,19,20,21,8,9,11]), 0, [torch.ones(B,H,D,S)], [torch.ones(B,H,S,D)])
        store_kv(torch.tensor([0,1,2,12,13,14,15,16,8,9,11]), 0, [torch.ones(B,H,D,S)], [torch.ones(B,H,S,D)])
        freqs_cos, freqs_sin = torch.ones(S, D//2)*0.6, torch.ones(S, D//2)*0.8
        mock_inputs = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,8,9,17,18,19,20,21,8,9,11])
        
        print("\nUnit Test: load_kv...")
        matched_kv = load_kv(mock_inputs, 0, freqs_cos, freqs_sin)
        print(matched_kv)

        print("\nUnit Test: load_kv...")
        inputs_kv = blend_kv(mock_inputs, 0, freqs_cos, freqs_sin, use_blend=True)
        print(inputs_kv)