# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import inspect
import json
import logging
import os
import re
import struct
import types

from functools import partial
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

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
    annotate_blend
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
    SEPARATE_EMBED_INFO_FILENAME,
    SEPARATE_EMBED_MATRIX_FILENAME,
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
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.debugging import _store_graph_quant_attrs, get_graph_name

class TextDecoder(Component):

    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
        apply_embedding: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.meta = {}
        self.quant_recipe: StaticLLMQuantRecipe = (
            self.config.quant_recipe(True) if self.config.quant_recipe else None
        )
        self.is_embedding = getattr(self.config, "is_embedding", False)
        self.blend_config = None

        # For multimodal embedding
        self._modality_placeholder_token_id = None
        self.apply_embedding = apply_embedding
        self.tok_embedding_passes_job = (
            get_capture_program_passes() if apply_embedding else None
        )
        self.tok_embedding_dep_table = (
            get_passes_dependency_for_capture_program() if apply_embedding else None
        )

        # load static llama model args
        params_path = (
            config.params_path if control_args.params is None else control_args.params
        )
        with open(params_path) as f:
            self.model_args = process_model_args(
                control_args, ModelArgs(**json.load(f)), self.quant_recipe, config, mode
            )
        # prepare instance
        self.tok_embedding, self.decoder = self._prepare_model()

        # check if sharding required
        if self.decoder and self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.meta["get_n_layers"],
                shares=self.config.num_sharding,
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

    def _prepare_model(self):  # noqa: C901
        if (instance := self._get_model_instance()) is None:
            return None, None
        tok_embedding, decoder = instance
        # load parameters for HF models
        if self.control_args.checkpoint is None:
            checkpoint = download_and_convert_hf_checkpoint(
                self.config.repo_id,
                self.config.convert_weights.__func__,
            )
            state_dict = torch.load(
                checkpoint, weights_only=True, map_location="cpu", mmap=True
            )
            if self.control_args.decoder_model in {
                "gemma-2b",
                "gemma2-2b",
                "gemma3-1b",
            }:
                for k, v in state_dict.items():
                    if "norm" not in k:
                        continue
                    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                    # See https://github.com/huggingface/transformers/pull/29402
                    state_dict[k] = v.float() + torch.ones(v.shape, dtype=torch.float32)
        else:
            state_dict = torch.load(
                self.control_args.checkpoint,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if self.control_args.decoder_model == "stories260k":
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }

        # change to HF weight to improve the performance of RoPE in HTP backend.
        if self.config.transform_weight:

            def permute(w, heads, partial_rotary_dim):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                transformed_weight = (
                    w.view(
                        heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1
                    )
                    .transpose(2, 3)
                    .reshape(dim_0, dim_1)
                )
                return transformed_weight

            # TODO: handle cases where input size isn't divisible.
            partial_rotary_dim = int(1 // self.model_args.partial_rotary_factor)
            for layer_i in range(decoder.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"],
                    decoder.n_heads,
                    partial_rotary_dim,
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"],
                    decoder.n_kv_heads,
                    partial_rotary_dim,
                )

        if self.control_args.num_layers is not None:
            trimmed_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("layers."):
                    trimmed_state_dict[key] = value
                    continue

                parts = key.split(".")
                layer_idx = int(parts[1])
                if layer_idx < decoder.n_layers:
                    trimmed_state_dict[key] = value
            state_dict = trimmed_state_dict

        decoder.load_state_dict(state_dict, strict=True, assign=True)

        # apply spin quant if required
        if any([self.config.r1, self.config.r2]):
            decoder.config = types.SimpleNamespace(
                dim=decoder.dim,
                head_dim=decoder.dim // decoder.n_heads,
                n_local_heads=decoder.n_heads,
                intermediate_size=4 * decoder.dim,
            )
            apply_spinquant(
                decoder,
                use_r1=self.config.r1,
                use_r2=self.config.r2,
                use_r4=False,
                pretrained_rotation_path=None,
                qkv_split=True,
            )

        # perform model transformation
        for layer in decoder.layers:
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

        decoder = convert_linear_to_conv2d(decoder)

        # check dtype override
        if self.control_args.dtype_override is not None:
            dtype_override = DType[self.control_args.dtype_override]
            decoder = decoder.to(dtype_override.to_torch_dtype())

        # check embedding fallback
        if self.control_args.embedding_quantize:
            decoder = get_quant_embedding_transform(
                embedding_quantize=self.control_args.embedding_quantize
            )(decoder)
            self.passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "skip_node"
            ] = {"tokens"}
            if self.apply_embedding:
                tok_embedding = get_quant_embedding_transform(
                    embedding_quantize=self.control_args.embedding_quantize
                )(tok_embedding)
                self.tok_embedding_passes_job[I64toI32][
                    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
                ]["skip_node"] = {"tokens"}

        if tok_embedding is not None:
            tok_embedding = tok_embedding.eval()

        return tok_embedding, decoder.eval()

    def _get_model_instance(self) -> LlamaModel:
        if self.mode == Mode.PREFILL and self.control_args.model_mode == "kv":
            return None
        use_i64_token = self.control_args.embedding_quantize is not None

        # get embedding model
        tok_embedding = None
        if self.apply_embedding:
            auto_model = AutoModel.from_pretrained(
                self.config.repo_id, _attn_implementation="eager"
            )
            tok_embedding = TextEmbedding(
                auto_model.get_input_embeddings().to(torch.float32),
                self.model_args.max_batch_size,
                self.model_args.ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )
        # get decoder model
        if self.control_args.decoder_model in {"gemma-2b", "gemma3-1b"}:
            # For gemma, we have preprocessed the weight of rmsnorm
            self.model_args.norm_type = "rmsnorm"

        model_specific_kwargs = get_model_specific_kwargs(
            self.control_args, self.config
        )
        model_specific_kwargs["is_embedding"] = self.is_embedding
        decoder: LlamaModel = LLM_VARIANT_ARCHS.get(
            self.control_args.decoder_model, LlamaModel
        )(
            self.model_args,
            ar_len=self.model_args.ar_len,
            output_new_cache_only=True,
            output_cache=True,
            use_i64_token=use_i64_token,
            **model_specific_kwargs,
        )

        self.meta = decoder.get_metadata()
        # get example input
        self.example_input = decoder.get_example_inputs()
        self.get_example_inputs = decoder.get_example_inputs
        self.export_input = (
            self.example_input[0],  # tokens or hidden_states
            *self.example_input[1],  # attn_mask
            *((self.example_input[2],) if decoder.use_kv_cache else []),  # pos_ids
            *(self.example_input[3] if decoder.use_kv_cache else []),  # k_caches
            *(self.example_input[4] if decoder.use_kv_cache else []),  # v_caches
            *((self.example_input[5],) if decoder.use_blend else []),  # valid_mask
        )
        output_dim = decoder.dim if self.is_embedding else decoder.vocab_size
        self.io_shape = {
            (
                decoder.max_batch_size,
                decoder.ar_len,
                output_dim,
            ),
            (
                decoder.max_batch_size,
                decoder.blend_len,
                output_dim,
            ),
        }
        # shape of k caches and v caches
        self.kv_cache_shape = {
            # single head, kv input
            (self.meta["get_head_dim"], self.meta["get_max_context_len"]),
            (self.meta["get_max_context_len"], self.meta["get_head_dim"]),
            # single head, kv output
            (self.meta["get_head_dim"], self.meta["get_ar_len"]),
            (self.meta["get_ar_len"], self.meta["get_head_dim"]),
            # single head, kv output
            (self.meta["get_head_dim"], self.meta["get_blend_len"]),
            (self.meta["get_blend_len"], self.meta["get_head_dim"]),
        }
        if decoder.use_blend:
            self.blend_config = {
                "blend_len": decoder.blend_len
            }

        if self.apply_embedding:
            self.tok_embedding_export_input = (
                tok_embedding.get_example_input()
            )  # tokens

        return tok_embedding, decoder

    def _save_logits_quant_attrs(self):
        for node in self.decoder.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.meta["get_logits_scale"] = output_node.args[1]
                            self.meta["get_logits_zero_point"] = output_node.args[2]
                            break

    def _extract_layer_idx(self, node: torch.fx.Node):
        pattern = r"layers[._](\d+)"
        candidates = [node.name]
        nn_stack = node.meta.get("nn_module_stack", None)
        if nn_stack is not None:
            candidates.append(str(nn_stack.values()))

        def gather_node_sources(node_source):
            if node_source is None:
                return
            for src in node_source:
                name = getattr(src, "name", None)
                if isinstance(name, str):
                    candidates.append(name)
                gather_node_sources(getattr(src, "from_node", None))

        gather_node_sources(node.meta.get("from_node", None))
        for candidate in candidates:
            match = re.search(pattern, candidate)
            if match:
                return int(match.group(1))
        return None

    def _collect_layer_hidden_states(
        self, graph_module: torch.fx.GraphModule
    ) -> Dict[int, torch.Tensor]:
        layer_outputs = {}
        expected_dim = self.meta["get_dim"]

        class LayerOutputCollector(torch.fx.Interpreter):
            def __init__(self, gm, parent):
                super().__init__(gm)
                self.parent = parent

            def run_node(self, n):
                out = super().run_node(n)
                if (
                    isinstance(out, torch.Tensor)
                    and out.dim() == 3
                    and out.shape[-1] == expected_dim
                ):
                    layer_idx = self.parent._extract_layer_idx(n)
                    if layer_idx is not None:
                        layer_outputs[layer_idx] = out.detach().float().cpu()
                return out

        LayerOutputCollector(graph_module, self).run(*self.export_input)
        return layer_outputs

    def _compare_prepared_vs_converted_by_layer(
        self,
        prepared_outputs: Dict[int, torch.Tensor],
        converted_outputs: Dict[int, torch.Tensor],
    ):
        common_layers = sorted(
            set(prepared_outputs.keys()) & set(converted_outputs.keys())
        )
        if not common_layers:
            logging.warning(
                "Layer-wise compare skipped: no matched layer hidden states were found."
            )
            return

        for layer_idx in common_layers:
            fp_out = prepared_outputs[layer_idx]
            qdq_out = converted_outputs[layer_idx]
            if fp_out.shape != qdq_out.shape:
                logging.warning(
                    "[LayerCompare][%s] layer=%d shape mismatch: %s vs %s",
                    self.mode.name,
                    layer_idx,
                    tuple(fp_out.shape),
                    tuple(qdq_out.shape),
                )
                continue
            mse = torch.mean((fp_out - qdq_out) ** 2).item()
            fp_energy = torch.mean(fp_out**2).item() + 1e-12
            nmse = mse / fp_energy
            cos = (
                F.cosine_similarity(
                    fp_out.reshape(-1, fp_out.shape[-1]),
                    qdq_out.reshape(-1, qdq_out.shape[-1]),
                    dim=-1,
                )
                .mean()
                .item()
            )
            logging.info(
                "[LayerCompare][%s] layer=%d mse=%.6e nmse=%.6e cos=%.6f",
                self.mode.name,
                layer_idx,
                mse,
                nmse,
                cos,
            )

    def _pick_first_node_arg(self, args):
        if isinstance(args, torch.fx.Node):
            return args
        if isinstance(args, (tuple, list)):
            for item in args:
                picked = self._pick_first_node_arg(item)
                if picked is not None:
                    return picked
        return None

    def _get_output_predecessor_chain(
        self, graph_module: torch.fx.GraphModule, num_predecessors: int = 5
    ) -> List[torch.fx.Node]:
        output_node = None
        for node in graph_module.graph.nodes:
            if node.op == "output":
                output_node = node
                break
        if output_node is None:
            return []

        logits_node = self._pick_first_node_arg(output_node.args)
        if logits_node is None:
            return []

        chain = [logits_node]
        current = logits_node
        for _ in range(num_predecessors):
            prev = self._pick_first_node_arg(current.args)
            if prev is None:
                break
            chain.append(prev)
            current = prev
        return chain

    def _collect_output_predecessor_tensors(
        self, graph_module: torch.fx.GraphModule, num_predecessors: int = 5
    ) -> List[Tuple[str, torch.Tensor]]:
        chain_nodes = self._get_output_predecessor_chain(
            graph_module, num_predecessors=num_predecessors
        )
        if not chain_nodes:
            return []
        target_names = {node.name for node in chain_nodes}
        captured = {}

        class OutputPredCollector(torch.fx.Interpreter):
            def run_node(self, n):
                out = super().run_node(n)
                if n.name in target_names and isinstance(out, torch.Tensor):
                    captured[n.name] = out.detach().float().cpu()
                return out

        OutputPredCollector(graph_module).run(*self.export_input)
        ordered = []
        for node in chain_nodes:
            if node.name in captured:
                ordered.append((node.name, captured[node.name]))
        return ordered

    def _compare_output_and_predecessors(
        self,
        prepared_tensors: List[Tuple[str, torch.Tensor]],
        converted_tensors: List[Tuple[str, torch.Tensor]],
    ):
        n = min(len(prepared_tensors), len(converted_tensors))
        if n == 0:
            logging.warning(
                "[OutPredCompare][%s] skipped: no output/predecessor tensors found.",
                self.mode.name,
            )
            return

        for idx in range(n):
            fp_name, fp_out = prepared_tensors[idx]
            qdq_name, qdq_out = converted_tensors[idx]
            if fp_out.shape != qdq_out.shape:
                logging.warning(
                    "[OutPredCompare][%s] depth=%d fp=%s qdq=%s shape mismatch: %s vs %s",
                    self.mode.name,
                    idx,
                    fp_name,
                    qdq_name,
                    tuple(fp_out.shape),
                    tuple(qdq_out.shape),
                )
                continue
            mse = torch.mean((fp_out - qdq_out) ** 2).item()
            fp_energy = torch.mean(fp_out**2).item() + 1e-12
            nmse = mse / fp_energy
            cos = F.cosine_similarity(
                fp_out.reshape(1, -1), qdq_out.reshape(1, -1), dim=-1
            ).item()
            logging.info(
                "[OutPredCompare][%s] depth=%d fp=%s qdq=%s mse=%.6e nmse=%.6e cos=%.6f",
                self.mode.name,
                idx,
                fp_name,
                qdq_name,
                mse,
                nmse,
                cos,
            )

    def _get_embedding_reference_model_path(self) -> str:
        default_path = os.path.join(
            self.control_args.artifact, "qwen3_embed_limit10_reference.pt2"
        )
        return os.getenv("LLAMA_EMBED_REF_MODEL_PATH", default_path)

    def _get_embedding_replaced_model_save_path(self) -> str:
        default_path = os.path.join(
            self.control_args.artifact, "qwen3_embed_replaced_from_saved_ref.pt2"
        )
        return os.getenv("LLAMA_EMBED_REPLACED_MODEL_PATH", default_path)

    def _compare_with_saved_embedding_reference(self):
        ref_path = self._get_embedding_reference_model_path()
        if not os.path.exists(ref_path):
            logging.warning(
                "[SavedModelCompare][%s] skipped: reference model not found at %s",
                self.mode.name,
                ref_path,
            )
            return
        try:
            ref_decoder = torch.export.load(ref_path).module()
            ref_tensors = self._collect_output_predecessor_tensors(
                ref_decoder, num_predecessors=0
            )
            cur_tensors = self._collect_output_predecessor_tensors(
                self.decoder, num_predecessors=0
            )
            logging.info(
                "[SavedModelCompare][%s] compare current convert_pt2e model against %s",
                self.mode.name,
                ref_path,
            )
            self._compare_output_and_predecessors(ref_tensors, cur_tensors)
        except Exception as e:
            logging.warning(
                "[SavedModelCompare][%s] skipped: failed to load/compare reference model %s due to %s",
                self.mode.name,
                ref_path,
                str(e),
            )

    def _get_quant_arg_indices(
        self, node: torch.fx.Node
    ) -> Tuple[int, int, int, int] | None:
        if node.op != "call_function" or not hasattr(node.target, "_schema"):
            return None
        schema = node.target._schema
        arg_names = [arg.name for arg in schema.arguments]

        def find_idx(candidates: List[str]):
            for name in candidates:
                if name in arg_names:
                    return arg_names.index(name)
            return None

        scale_idx = find_idx(["scale", "scales"])
        zero_point_idx = find_idx(["zero_point", "zero_points"])
        quant_min_idx = find_idx(["quant_min", "min"])
        quant_max_idx = find_idx(["quant_max", "max"])
        if (
            scale_idx is None
            or zero_point_idx is None
            or quant_min_idx is None
            or quant_max_idx is None
        ):
            return None
        return scale_idx, zero_point_idx, quant_min_idx, quant_max_idx

    def _resolve_quant_arg_value(self, graph_module: torch.fx.GraphModule, arg):
        if isinstance(arg, torch.fx.Node):
            if arg.op != "get_attr":
                raise RuntimeError(
                    f"Unsupported quant arg producer {arg.op} for node {arg.name}"
                )
            return getattr(graph_module, arg.target)
        return arg

    def _to_scalar(self, value, desc: str):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RuntimeError(
                    f"{desc} should be scalar, got tensor shape {tuple(value.shape)}"
                )
            return value.item()
        return value

    def _assign_quant_arg_value(
        self, graph_module: torch.fx.GraphModule, node: torch.fx.Node, idx: int, value
    ) -> None:
        current_arg = node.args[idx]
        if isinstance(current_arg, torch.fx.Node):
            if current_arg.op != "get_attr":
                raise RuntimeError(
                    f"Unsupported quant arg consumer {current_arg.op} for node {node.name}"
                )
            if isinstance(value, torch.Tensor):
                setattr(graph_module, current_arg.target, value.detach().clone())
            else:
                setattr(graph_module, current_arg.target, value)
            return

        new_args = list(node.args)
        if isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"Expected scalar quant arg for node {node.name} index {idx}, got tensor"
            )
        new_args[idx] = value
        node.args = tuple(new_args)

    def _replace_quant_scale_zero_point_with_saved_embedding_reference(self):
        ref_path = self._get_embedding_reference_model_path()
        if not os.path.exists(ref_path):
            logging.warning(
                "[SavedModelReplace][%s] skipped: reference model not found at %s",
                self.mode.name,
                ref_path,
            )
            return

        ref_decoder = torch.export.load(ref_path).module()
        ref_nodes = []
        cur_nodes = []
        for node in ref_decoder.graph.nodes:
            if self._get_quant_arg_indices(node) is not None:
                ref_nodes.append(node)
        for node in self.decoder.graph.nodes:
            if self._get_quant_arg_indices(node) is not None:
                cur_nodes.append(node)

        if len(ref_nodes) != len(cur_nodes):
            raise RuntimeError(
                "[SavedModelReplace] quant node count mismatch: "
                f"reference={len(ref_nodes)} current={len(cur_nodes)}"
            )

        for idx, (ref_node, cur_node) in enumerate(zip(ref_nodes, cur_nodes)):
            if str(ref_node.target) != str(cur_node.target):
                raise RuntimeError(
                    "[SavedModelReplace] quant node target mismatch at index "
                    f"{idx}: reference={ref_node.target} current={cur_node.target}"
                )

            (
                ref_scale_idx,
                ref_zero_point_idx,
                ref_quant_min_idx,
                ref_quant_max_idx,
            ) = self._get_quant_arg_indices(ref_node)
            (
                cur_scale_idx,
                cur_zero_point_idx,
                cur_quant_min_idx,
                cur_quant_max_idx,
            ) = self._get_quant_arg_indices(cur_node)

            ref_quant_min = self._to_scalar(
                self._resolve_quant_arg_value(
                    ref_decoder, ref_node.args[ref_quant_min_idx]
                ),
                f"reference quant_min ({ref_node.name})",
            )
            cur_quant_min = self._to_scalar(
                self._resolve_quant_arg_value(
                    self.decoder, cur_node.args[cur_quant_min_idx]
                ),
                f"current quant_min ({cur_node.name})",
            )
            ref_quant_max = self._to_scalar(
                self._resolve_quant_arg_value(
                    ref_decoder, ref_node.args[ref_quant_max_idx]
                ),
                f"reference quant_max ({ref_node.name})",
            )
            cur_quant_max = self._to_scalar(
                self._resolve_quant_arg_value(
                    self.decoder, cur_node.args[cur_quant_max_idx]
                ),
                f"current quant_max ({cur_node.name})",
            )
            if ref_quant_min != cur_quant_min or ref_quant_max != cur_quant_max:
                raise RuntimeError(
                    "[SavedModelReplace] quant range mismatch at node index "
                    f"{idx} ({cur_node.name}): "
                    f"reference min/max=({ref_quant_min}, {ref_quant_max}) "
                    f"current min/max=({cur_quant_min}, {cur_quant_max})"
                )

            ref_scale = self._resolve_quant_arg_value(
                ref_decoder, ref_node.args[ref_scale_idx]
            )
            ref_zero_point = self._resolve_quant_arg_value(
                ref_decoder, ref_node.args[ref_zero_point_idx]
            )
            self._assign_quant_arg_value(self.decoder, cur_node, cur_scale_idx, ref_scale)
            self._assign_quant_arg_value(
                self.decoder, cur_node, cur_zero_point_idx, ref_zero_point
            )

        self.decoder.recompile()
        replaced_path = self._get_embedding_replaced_model_save_path()
        replaced_dir = os.path.dirname(replaced_path)
        if replaced_dir:
            os.makedirs(replaced_dir, exist_ok=True)
        replaced_ep = torch.export.export(self.decoder, self.export_input, strict=True)
        torch.export.save(replaced_ep, replaced_path)
        logging.info(
            "[SavedModelReplace][%s] replaced scale/zero_point from %s (%d quant nodes), saved replaced model to %s",
            self.mode.name,
            ref_path,
            len(cur_nodes),
            replaced_path,
        )

    def _save_input_kv_cache_quant_attrs(self):
        input_kv_cache_shape = {
            # single head, k input
            (
                self.meta["get_head_dim"],
                self.meta["get_max_context_len"] - self.meta["get_ar_len"],
            ),
            # single head, v input
            (
                self.meta["get_max_context_len"] - self.meta["get_ar_len"],
                self.meta["get_head_dim"],
            ),
        }

        idx = 0
        for node in self.decoder.graph.nodes:
            if (
                node.op == "placeholder"
                and len(users := list(node.users)) == 1
                and "val" in node.meta
                and node.meta["val"].size()[-2:] in input_kv_cache_shape
            ):
                scale_cache_name = f"get_k_scale_input_{idx}"
                zero_point_cache_name = f"get_k_zero_point_input_{idx}"
                if idx >= self.meta["get_n_layers"]:
                    scale_cache_name = (
                        f"get_v_scale_input_{idx % self.meta['get_n_layers']}"
                    )
                    zero_point_cache_name = (
                        f"get_v_zero_point_input_{idx % self.meta['get_n_layers']}"
                    )
                self.meta[scale_cache_name] = users[0].args[1]
                self.meta[zero_point_cache_name] = users[0].args[2]
                idx += 1

    def _save_output_kv_cache_quant_attrs(self):
        k_idx = 0
        v_idx = 0
        for node in self.decoder.graph.nodes:
            if not is_graph_output(node):
                continue
            cache_output_node = node.args[0].args[0]
            if cache_output_node.meta["val"].size()[-2:] in self.kv_cache_shape:
                # [QCOM_SCALE, QCOM_ZERO_POINT, QCOM_QUANT_MIN, QCOM_QUANT_MAX, QCOM_DTYPE]
                # This meta is for attention sink feature
                self.meta[f"get_kv_output_{k_idx+v_idx}_quant_attr"] = [
                    node.args[1],
                    node.args[2],
                    node.args[3],
                    node.args[4],
                    str(node.args[5]),
                ]
                if is_node_src_start_with_name(cache_output_node, "k_"):
                    self.meta[f"get_k_scale_output_{k_idx}"] = node.args[1]
                    self.meta[f"get_k_zero_point_output_{k_idx}"] = node.args[2]
                    k_idx += 1
                elif is_node_src_start_with_name(cache_output_node, "v_"):
                    self.meta[f"get_v_scale_output_{v_idx}"] = node.args[1]
                    self.meta[f"get_v_zero_point_output_{v_idx}"] = node.args[2]
                    v_idx += 1

    def _tag_ios(self, node, fixed_point_type):
        # For blender, no need to quant valid_mask and imp_indices.
        atten_mask_shape = {
            (
                self.meta["get_max_batch_size"],
                self.meta["get_ar_len"],
                self.meta["get_max_context_len"],
            ),
        }

        freq_shape = {
            (self.meta["get_ar_len"], self.meta["get_head_dim"] // 2),
        }

        freq_op = {
            exir_ops.edge.aten.select.int,
        }
        quant_io_type = None

        if node.op == "placeholder":
            if (
                (len(users := list(node.users)) == 1 or "args" in node.name)
                and users[0].meta["val"].size()[-2:] in self.kv_cache_shape
                and "constant" not in node.name
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in atten_mask_shape:
                quant_io_type = fixed_point_type["io_type"]

        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in self.kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if node.target in freq_op and node.meta["val"].size() in freq_shape:
            quant_io_type = fixed_point_type["io_type"]

        if ("args" in node.name):
            print(f"tag_ios, args, {node.name}, {quant_io_type}")
        return quant_io_type

    def _calibrate(
        self,
        model,
        tokenizer,
        event,
        user_calibration_data,
        tok_embedding=None,
        intermediate_outputs=None,
    ):
        """
        Calibrate the model using either task-based evaluation or prompt-based inference.

        This method performs Post-Training Quantization (PTQ) calibration by running inference
        on the model with either:
        1. Task-based datasets by lm_eval for text-only models in perplexity evaluation
        2. User-provided prompts for both text-only and multimodal models

        Args:
            model: The decoder model to calibrate (GraphModule after prepare_pt2e)
            tokenizer: Tokenizer for encoding text inputs
            event: Event name for logging (e.g., "prepare_pt2e", "convert_pt2e")
            tok_embedding: Optional text embedding module (required only for multimodal models)
            intermediate_outputs: Optional pre-computed embeddings from vision/audio encoder
                                 (required only for multimodal models)
        """
        # Determine if this is a multimodal model
        is_multimodal = tok_embedding is not None

        # Determine if task-based calibration is requested
        has_task_calibration = self.control_args.tasks is not None

        # Task-based calibration: Only for text-only LLMs
        # Multimodal models (VLMs) cannot use task-based evaluation currently.
        # BLENDER do not need task-based evaluation.
        if has_task_calibration and not is_multimodal:
            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=model,
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_context_len"],
                tasks=self.control_args.tasks,
                tasks_limit=self.control_args.limit,
                num_fewshot=self.control_args.num_fewshot,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_tasks",
                seq_mse_candidates=self.config.seq_mse_candidates,
                blend_config=self.blend_config
            )

        # prepare lookahead config if applicable
        lookahead_config = (
            (self.control_args.window, self.control_args.ngram, self.control_args.gcap)
            if (
                self.mode == Mode.DECODE and self.control_args.model_mode == "lookahead"
            )
            else None
        )
        # check user's prompt which helps calibrate special token
        print("self.mode: ", self.mode)
        print()
        for prompt in user_calibration_data:
            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                hidden_states=intermediate_outputs,  # hidden_states for multimodal
                module=model,
                tok_embedding=tok_embedding,
                modality_placeholder_token_id=self.meta.get(
                    "modality_placeholder_token_id", None
                ),
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_context_len"],
                prompt=prompt,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_prompt",
                lookahead_config=lookahead_config,
                blend_config=self.blend_config
            )

    @log_info
    def quantize(self, request: Request):  # noqa: C901
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

        data = request.method_data[TEXT_DECODER]

        image_embedding = None
        if self.apply_embedding:
            # For demo: get first data now
            image_embedding = request.method_data[
                VISION_ENCODER
            ].calibration_data.intermediate_outputs[0]

        quantizer = make_quantizer()
        for custom_annotation in data.custom_annotation:
            self.quant_recipe.recipe.custom_quant_annotations.append(custom_annotation)
        if self.mode == Mode.BLENDER:
            self.quant_recipe.recipe.custom_quant_annotations.append(annotate_blend)
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

            prepared_outputs_for_compare = None
            prepared_final_output_tensors = None
            if self.is_embedding:
                prepared_outputs_for_compare = self._collect_layer_hidden_states(
                    self.decoder
                )
                prepared_final_output_tensors = self._collect_output_predecessor_tensors(
                    self.decoder, num_predecessors=0
                )
            # start calibration
            self._calibrate(
                model=self.decoder,
                tokenizer=data.tokenizer,
                event="prepare_pt2e",
                user_calibration_data=data.calibration_data.datasets,
                tok_embedding=self.tok_embedding,
                intermediate_outputs=image_embedding,
            )
            self.decoder = convert_pt2e(self.decoder)

            for node in self.decoder.graph.nodes:
                print(node.name)
                print("  op:", node.op)
                print("  args:", node.args)
                print("  target:", node.target)
                print("  ")
                # print("  meta:", node.meta)
                print("\n\n\n")

            # Saving Decode QDQ Model EP for SQNR evaluation
            # if self.mode == Mode.DECODE:
            #     qdq_ep = torch.export.export(
            #         self.decoder, self.export_input, strict=True
            #     )
            #     qdq_ep_path = f"{self.control_args.artifact}/{DECODE_QDQ_FILENAME}"
            #     torch.export.save(qdq_ep, qdq_ep_path)
            #     logging.info(f"QDQ EP saved to {qdq_ep_path}")

            if self.apply_embedding:
                self.tok_embedding = convert_pt2e(self.tok_embedding)
            
<<<<<<< HEAD
            if prepared_outputs_for_compare is not None:
                converted_outputs_for_compare = self._collect_layer_hidden_states(
                    self.decoder
                )
                self._compare_prepared_vs_converted_by_layer(
                    prepared_outputs_for_compare,
                    converted_outputs_for_compare,
                )
            if prepared_final_output_tensors is not None:
                converted_out_pred_tensors = self._collect_output_predecessor_tensors(
                    self.decoder, num_predecessors=0
                )
                self._compare_output_and_predecessors(
                    prepared_final_output_tensors, converted_out_pred_tensors
                )
            # if self.is_embedding and self.mode == Mode.PREFILL:
            #     self._replace_quant_scale_zero_point_with_saved_embedding_reference()
            #     self._compare_with_saved_embedding_reference()

            # ppl test after quant
=======
            # store quant info for debug purpose only
            # _store_graph_quant_attrs(self.decoder, get_graph_name(self.mode), os.path.join(self.control_args.artifact, "quant_attrs"))
            if False:
                if self.mode == Mode.DECODE:
                    import pickle
                    decode_quant_attrs = []
                    for node in self.decoder.graph.nodes:
                        if (node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default):
                            decode_quant_attrs.append([node.args[0].name] + list(node.args[1:-1]))
                    pickle.dump(decode_quant_attrs, open(os.path.join(self.control_args.artifact, "decode_quant_attrs.pkl"), "wb"))
                if self.mode == Mode.BLENDER:
                    import pickle
                    blender_quant_attrs = []
                    decode_quant_attrs = pickle.load(open(os.path.join(self.control_args.artifact, "decode_quant_attrs.pkl"), "rb"))
                    scales = torch.tensor([qarg[1] for qarg in decode_quant_attrs])
                    zp = torch.tensor([qarg[2] for qarg in decode_quant_attrs])
                    for node in self.decoder.graph.nodes:
                        if (node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default):
                            blender_quant_attrs.append([node.args[0].name] + list(node.args[1:-1]))
                            match_score = torch.min(torch.abs(scales-node.args[1])/scales+torch.abs(zp-node.args[2])/zp)
                            match_item = torch.argmin(torch.abs(scales-node.args[1])/scales+torch.abs(zp-node.args[2])/zp)
                            print(f"blender_debug: {node.args[0:3]}, decode matched: {decode_quant_attrs[match_item][0:3]}, match score: {match_score}")

>>>>>>> origin/main
            if self.mode == Mode.PREFILL or self.mode == Mode.BLENDER:
                self._calibrate(
                    model=self.decoder,
                    tokenizer=data.tokenizer,
                    event="convert_pt2e",
                    user_calibration_data=data.calibration_data.datasets,
                    tok_embedding=self.tok_embedding,
                    intermediate_outputs=image_embedding,
                )

            if self.control_args.verbose:
                if self.apply_embedding:
                    image_embedding = request.method_data[
                        VISION_ENCODER
                    ].calibration_data.qdq_intermediate_outputs[0]
                self._calibrate(
                    model=self.decoder,
                    tokenizer=data.tokenizer,
                    event="convert_pt2e",
                    user_calibration_data=data.calibration_data.datasets,
                    tok_embedding=self.tok_embedding,
                    intermediate_outputs=image_embedding,
                )

        # save logit's quantization attributes to meta
        self._save_logits_quant_attrs()

        # save output KV cache's quantization attributes to meta for attention sink and multimodal
        self._save_output_kv_cache_quant_attrs()

        # LLM: propagate kv cache quantization attributes for prefill model
        if not self.apply_embedding:
            if not self.is_embedding and self.mode == Mode.DECODE:
                kv_quant_attrs, output_indices = {}, 0
                for node in self.decoder.graph.nodes:
                    if node.op == "output":
                        for output in node.args[0]:
                            print("determinine annotate_prefill_kv_output: ", output_indices, output.args[1:])
                            kv_quant_attrs[output_indices] = output.args[1:]
                            output_indices += 1
                        break

                data.custom_annotation += (
                    partial(
                        annotate_prefill_kv_output,
                        kv_quant_attrs=kv_quant_attrs,
                    ),
                )
        # MultiModal: save kv cache IO quantization attributes to requant kv cache from prefill output scale/zero_point to decode input scale/zero_point
        else:
            # save input kv cache's quantization attributes to meta
            if ((not self.is_embedding and self.mode == Mode.DECODE) 
                or (self.is_embedding and self.mode == Mode.PREFILL)):
                self._save_input_kv_cache_quant_attrs()

        # setup quantized IO
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)
        if self.tok_embedding_passes_job is not None:
            self.tok_embedding_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.tok_embedding_passes_job[TagQuantIO][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["get_quant_io_dtype_fn"] = partial(
                self._tag_ios, fixed_point_type=fixed_point_type
            )


class HybridTextDecoder(Component):
    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        apply_embedding: bool = False,
    ):
        self.decode = TextDecoder(
            control_args,
            config,
            Mode.DECODE,
            apply_embedding=apply_embedding,
        )
        self.prefill = TextDecoder(
            control_args,
            config,
            Mode.PREFILL,
            apply_embedding=apply_embedding,
        )
        self.control_args = control_args
        self.config = config

        if control_args.model_mode == "blender":
            self.blender = TextDecoder(
                control_args,
                config,
                Mode.BLENDER,
                apply_embedding=apply_embedding,
            )
            self.set_next(self.decode).set_next(self.prefill).set_next(self.blender)
        else:
            self.set_next(self.decode).set_next(self.prefill)

        self.apply_embedding = apply_embedding

    def _depends_on_node(self, node, source_node, memo):
        if node == source_node:
            return True
        if node in memo:
            return memo[node]
        depends = False
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if self._depends_on_node(arg, source_node, memo):
                    depends = True
                    break
            elif isinstance(arg, (tuple, list)):
                for item in arg:
                    if (
                        isinstance(item, torch.fx.Node)
                        and self._depends_on_node(item, source_node, memo)
                    ):
                        depends = True
                        break
                if depends:
                    break
        memo[node] = depends
        return depends

    def _extract_embedding_boundary(self, graph, tokens_node):
        memo = {}
        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if not self._depends_on_node(node, tokens_node, memo):
                continue
            val = node.meta.get("val", None)
            if val is None:
                continue
            node_dim = getattr(val, "dim", None)
            if callable(node_dim):
                if node_dim() == 3:
                    return node
            elif hasattr(val, "shape") and len(val.shape) == 3:
                return node
        raise RuntimeError(
            "Failed to identify token-embedding boundary node for separate embedding flow."
        )

    def _find_tokens_placeholder(self, graph, graph_name):
        for node in graph.nodes:
            if node.op == "placeholder" and "token" in node.name:
                return node
        raise RuntimeError(
            f"Unable to find token placeholder in graph {graph_name} for separate embedding flow."
        )

    def _find_embedding_node(self, graph, tokens_node, graph_name):
        memo = {}
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.embedding.default
                and self._depends_on_node(node, tokens_node, memo)
            ):
                return node
        raise RuntimeError(
            f"Unable to find embedding node in graph {graph_name} for separate embedding flow."
        )

    def _resolve_node_to_tensor(self, graph_module, value, preferred_dtype=None):
        if isinstance(value, torch.fx.Node):
            if value.op == "get_attr":
                value = getattr(graph_module, value.target)
            else:
                raise RuntimeError(
                    f"Expected get_attr node for tensor constant, got {value.op}:{value.target}"
                )

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, bool):
            return torch.tensor([int(value)], dtype=preferred_dtype or torch.int32)
        if isinstance(value, int):
            return torch.tensor([value], dtype=preferred_dtype or torch.int32)
        if isinstance(value, float):
            return torch.tensor([value], dtype=preferred_dtype or torch.float32)
        if isinstance(value, (list, tuple)):
            if preferred_dtype is not None:
                return torch.tensor(value, dtype=preferred_dtype)
            return torch.tensor(value)

        raise RuntimeError(f"Unsupported constant type for embedding payload: {type(value)}")

    def _torch_dtype_code(self, dtype):
        dtype_codes = {
            torch.float32: 1,
            torch.float16: 2,
            torch.int8: 3,
            torch.uint8: 4,
            torch.int16: 5,
            torch.uint16: 6,
            torch.int32: 7,
            torch.int64: 8,
            "int4": 9,
        }
        if dtype not in dtype_codes:
            raise RuntimeError(f"Unsupported dtype for separate embedding payload: {dtype}")
        return dtype_codes[dtype]

    def _resolve_node_to_scalar(self, graph_module, value):
        if isinstance(value, torch.fx.Node):
            if value.op != "get_attr":
                raise RuntimeError(
                    f"Expected get_attr node for scalar constant, got {value.op}:{value.target}"
                )
            value = getattr(graph_module, value.target)

        if isinstance(value, torch.Tensor):
            tensor_value = value.detach().cpu()
            if tensor_value.numel() != 1:
                raise RuntimeError(
                    f"Expected scalar tensor for quant range, got shape {list(tensor_value.shape)}"
                )
            value = tensor_value.item()

        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        raise RuntimeError(f"Unsupported scalar constant type: {type(value)}")

    def _get_quant_min_max(self, graph_module, dequant_node):
        if dequant_node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default:
            qmin_idx, qmax_idx = 3, 4
        elif dequant_node.target == torch.ops.quantized_decomposed.dequantize_per_channel.default:
            qmin_idx, qmax_idx = 5, 6
        elif dequant_node.target == torch.ops.torchao.dequantize_affine:
            qmin_idx, qmax_idx = 5, 6
        else:
            raise RuntimeError(f"Unsupported dequant op for quant range: {dequant_node.target}")

        if len(dequant_node.args) > max(qmin_idx, qmax_idx):
            quant_min = int(
                self._resolve_node_to_scalar(graph_module, dequant_node.args[qmin_idx])
            )
            quant_max = int(
                self._resolve_node_to_scalar(graph_module, dequant_node.args[qmax_idx])
            )
            return quant_min, quant_max

        if "quant_min" in dequant_node.kwargs and "quant_max" in dequant_node.kwargs:
            quant_min = int(
                self._resolve_node_to_scalar(graph_module, dequant_node.kwargs["quant_min"])
            )
            quant_max = int(
                self._resolve_node_to_scalar(graph_module, dequant_node.kwargs["quant_max"])
            )
            return quant_min, quant_max

        raise RuntimeError(
            f"Cannot read quant_min/quant_max from dequant node {dequant_node.target}."
        )

    def _storage_spec_from_quant_range(self, quant_min, quant_max):
        key = (int(quant_min), int(quant_max))
        if key == (0, 65535):
            return {"storage_type": "uint16", "bit_width": 16}
        if key == (-32767, 32767):
            return {"storage_type": "int16", "bit_width": 16}
        if key == (0, 255):
            return {"storage_type": "uint8", "bit_width": 8}
        if key == (-127, 127):
            return {"storage_type": "int8", "bit_width": 8}
        if key == (-7, 7):
            return {"storage_type": "int4", "bit_width": 4}
        raise RuntimeError(
            f"Unsupported embedding quant range ({quant_min}, {quant_max}) for separate embedding export."
        )

    def _pack_int4(self, tensor):
        flat = tensor.detach().cpu().contiguous().reshape(-1).to(torch.int32)
        if flat.numel() == 0:
            return b""
        if torch.any(flat < -8) or torch.any(flat > 7):
            raise RuntimeError(
                f"int4 packing received out-of-range values [{flat.min().item()}, {flat.max().item()}]."
            )
        lo = torch.bitwise_and(flat[0::2], 0xF).to(torch.uint8)
        hi_src = flat[1::2]
        if hi_src.numel() < lo.numel():
            hi_src = torch.cat((hi_src, torch.zeros(1, dtype=flat.dtype)))
        hi = torch.bitwise_left_shift(torch.bitwise_and(hi_src, 0xF), 4).to(torch.uint8)
        packed = torch.bitwise_or(lo, hi)
        return packed.numpy().tobytes(order="C")

    def _extract_embedding_payload(self, decoder_graph_module, graph_name):
        graph = decoder_graph_module.graph
        tokens_node = self._find_tokens_placeholder(graph, graph_name)
        embedding_node = self._find_embedding_node(graph, tokens_node, graph_name)
        weight_arg = embedding_node.args[0]

        dequant_ops = {
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.torchao.dequantize_affine,
        }

        if isinstance(weight_arg, torch.fx.Node) and weight_arg.target in dequant_ops:
            qweight = self._resolve_node_to_tensor(decoder_graph_module, weight_arg.args[0])
            scale_index, zp_index = 1, 2
            if weight_arg.target == torch.ops.torchao.dequantize_affine:
                scale_index, zp_index = 2, 3
            quant_min, quant_max = self._get_quant_min_max(
                decoder_graph_module, weight_arg
            )
            qweight_storage = self._storage_spec_from_quant_range(quant_min, quant_max)
            scale = self._resolve_node_to_tensor(
                decoder_graph_module,
                weight_arg.args[scale_index],
                preferred_dtype=torch.float32,
            )
            zp = self._resolve_node_to_tensor(
                decoder_graph_module,
                weight_arg.args[zp_index],
                preferred_dtype=torch.int32,
            )
            if qweight.dim() != 2:
                raise RuntimeError(
                    f"Expected 2D qweight matrix in graph {graph_name}, got shape {list(qweight.shape)}"
                )
            return {
                "quantized": True,
                "qweight": qweight,
                "scale": scale,
                "zp": zp,
                "quant_scheme": str(weight_arg.target),
                "quant_min": quant_min,
                "quant_max": quant_max,
                "qweight_storage": qweight_storage,
            }

        weight = self._resolve_node_to_tensor(decoder_graph_module, weight_arg)
        if weight.dim() != 2:
            raise RuntimeError(
                f"Expected 2D embedding weight matrix in graph {graph_name}, got shape {list(weight.shape)}"
            )
        return {
            "quantized": False,
            "weight": weight,
        }

    def _write_tensor_block(self, output_file, tensor, storage_spec=None):
        if tensor is None:
            output_file.write(struct.pack("<IIQ", 0, 0, 0))
            return {"dtype": None, "shape": []}

        payload_tensor = tensor.detach().cpu().contiguous()
        if payload_tensor.dtype == torch.bfloat16:
            payload_tensor = payload_tensor.to(torch.float32)
        shape = list(payload_tensor.shape)
        if storage_spec is not None:
            storage_type = storage_spec["storage_type"]
            if storage_type == "int4":
                dtype_code = self._torch_dtype_code("int4")
                raw = self._pack_int4(payload_tensor)
                dtype_name = "int4_packed"
            else:
                storage_type_to_torch_dtype = {
                    "uint16": torch.uint16,
                    "int16": torch.int16,
                    "uint8": torch.uint8,
                    "int8": torch.int8,
                }
                if storage_type not in storage_type_to_torch_dtype:
                    raise RuntimeError(
                        f"Unsupported storage_type for separate embedding payload: {storage_type}"
                    )
                storage_dtype = storage_type_to_torch_dtype[storage_type]
                payload_tensor = payload_tensor.to(storage_dtype).contiguous()
                dtype_code = self._torch_dtype_code(storage_dtype)
                raw = payload_tensor.numpy().tobytes(order="C")
                dtype_name = storage_type
        else:
            dtype_code = self._torch_dtype_code(payload_tensor.dtype)
            raw = payload_tensor.numpy().tobytes(order="C")
            dtype_name = str(payload_tensor.dtype)
        output_file.write(struct.pack("<IIQ", dtype_code, len(shape), len(raw)))
        if shape:
            output_file.write(struct.pack(f"<{len(shape)}I", *shape))
        output_file.write(raw)
        return {"dtype": dtype_name, "shape": shape}

    def _dump_separate_embedding_matrix(self, payload):
        output_path = os.path.join(
            self.control_args.artifact, SEPARATE_EMBED_MATRIX_FILENAME
        )
        with open(output_path, "wb") as output_file:
            # Magic 'SEMB', version=1, quantized flag.
            output_file.write(
                struct.pack("<4sII", b"SEMB", 1, 1 if payload["quantized"] else 0)
            )
            if payload["quantized"]:
                qweight_info = self._write_tensor_block(
                    output_file,
                    payload["qweight"],
                    storage_spec=payload["qweight_storage"],
                )
                scale_info = self._write_tensor_block(output_file, payload["scale"])
                zp_info = self._write_tensor_block(output_file, payload["zp"])
                weight_info = None
            else:
                weight_info = self._write_tensor_block(output_file, payload["weight"])
                qweight_info = self._write_tensor_block(output_file, None)
                scale_info = self._write_tensor_block(output_file, None)
                zp_info = self._write_tensor_block(output_file, None)

        return output_path, {
            "format": "SEMB_v1",
            "quantized": payload["quantized"],
            "weight": weight_info,
            "qweight": qweight_info,
            "scale": scale_info,
            "zp": zp_info,
            "quant_scheme": payload.get("quant_scheme", None),
            "quant_min": payload.get("quant_min", None),
            "quant_max": payload.get("quant_max", None),
        }

    def _prune_dead_nodes(self, graph):
        changed = True
        while changed:
            changed = False
            for node in list(graph.nodes)[::-1]:
                if node.op == "output":
                    continue
                if len(node.users) != 0:
                    continue
                graph.erase_node(node)
                changed = True

    def _rewrite_decoder_input_for_separate_embed(
        self, decoder_graph_module, graph_name
    ):
        graph = decoder_graph_module.graph
        tokens_node = self._find_tokens_placeholder(graph, graph_name)

        split_node = self._extract_embedding_boundary(graph, tokens_node)
        split_meta = dict(split_node.meta)
        split_val = split_node.meta.get("val", None)
        split_shape = list(split_val.shape) if hasattr(split_val, "shape") else None
        split_dtype = str(split_val.dtype) if hasattr(split_val, "dtype") else None
        hidden_states_example_input = None
        if split_shape is not None:
            hidden_states_example_input = torch.zeros(
                split_shape,
                dtype=split_val.dtype if hasattr(split_val, "dtype") else torch.float32,
            )

        with graph.inserting_before(tokens_node):
            hidden_states_node = graph.placeholder("hidden_states")
            hidden_states_node.meta = split_meta

        split_node.replace_all_uses_with(hidden_states_node)
        self._prune_dead_nodes(graph)
        graph.lint()
        decoder_graph_module.recompile()

        input_placeholders = [
            node.name for node in decoder_graph_module.graph.nodes if node.op == "placeholder"
        ]
        if not input_placeholders or input_placeholders[0] != "hidden_states":
            raise RuntimeError(
                f"Graph {graph_name} has invalid input order after separate embedding rewrite: {input_placeholders}"
            )

        info = {
            "graph_name": graph_name,
            "hidden_states_shape": split_shape,
            "hidden_states_dtype": split_dtype,
        }
        return info, hidden_states_example_input

    @log_info
    def compile(self, request: Request):  # noqa: C901
        # force overriding frozen parameters here for model quantizing under seq mse scenario
        # this will make weight sharing work properly
        def override_params(decode, prefill):
            override_nodes = {
                str(node.meta["nn_module_stack"].values()): node
                for node in prefill.graph.nodes
                if node.target == torch.ops.aten.conv2d.default
            }
            indices_map = {
                # (affine_tensor, group_size, scales, zero_points, dtype, min, max)
                torch.ops.torchao.dequantize_affine: [0, 2, 3],
                # (per_channel_tensor, scales, zero_points, dim, dtype, min, max)
                torch.ops.quantized_decomposed.dequantize_per_channel.default: [
                    0,
                    1,
                    2,
                ],
                # should not need to worry about per-tensor case
            }
            for node in decode.graph.nodes:
                if node.target == torch.ops.aten.conv2d.default:
                    if target_node := override_nodes.get(
                        str(node.meta["nn_module_stack"].values())
                    ):
                        # arguments of conv: (input, weight, bias)
                        for i, dq_node in enumerate(node.args[1:]):
                            for index in indices_map[dq_node.target]:
                                setattr(
                                    prefill,
                                    target_node.args[i + 1].args[index].target,
                                    getattr(decode, dq_node.args[index].target),
                                )
                    else:
                        raise RuntimeError("failed to override quantization attribute")

        if self.config.seq_mse_candidates != 0 and self.control_args.model_mode != "kv":
            decode = self.decode.decoder
            if self.control_args.model_mode == "blender":
                prefill = self.prefill.decoder
                blender = self.blender.decoder
                override_params(decode, prefill)
                override_params(decode, blender)
            else:
                prefill = self.prefill.decoder
                override_params(decode, prefill)

        # prepare lowering tok_embedding if applicable
        if self.apply_embedding:
            tok_embedding_data = request.method_data[TEXT_EMBEDDING]
            models = [
                d for d in [self.decode, self.prefill] if d.tok_embedding is not None
            ]
            tok_embedding_example_inputs = [
                m.tok_embedding_export_input for m in models if m is not None
            ]  # tokens
            tok_embedding_graph_names = TEXT_EMBEDDING_GRAPH_NAMES[: len(models)]

        # prepare lowering decoder
        data = request.method_data[TEXT_DECODER]
        if self.control_args.model_mode == "blender":
            models = [
                d
                for d in [self.decode, self.prefill, self.blender]
                if d.decoder is not None
            ]
            graph_names = BLENDER_DECODER_GRAPH_NAMES[: len(models)]
        else:
            models = [d for d in [self.decode, self.prefill] if d.decoder is not None]
            graph_names = DECODER_GRAPH_NAMES[: len(models)]

        if getattr(self.control_args, "separate_embed", False):
            embedding_payload = self._extract_embedding_payload(
                models[0].decoder, graph_names[0]
            )
            matrix_path, matrix_meta = self._dump_separate_embedding_matrix(
                embedding_payload
            )
            separate_embed_info = []
            for graph_name, model in zip(graph_names, models):
                graph_info, hidden_states_input = (
                    self._rewrite_decoder_input_for_separate_embed(
                        model.decoder, graph_name
                    )
                )
                if hidden_states_input is None:
                    raise RuntimeError(
                        f"Graph {graph_name} has no valid hidden_states shape after separate embedding rewrite."
                    )
                model.export_input = (
                    hidden_states_input,
                    *model.export_input[1:],
                )
                separate_embed_info.append(graph_info)
            sidecar_path = os.path.join(
                self.control_args.artifact, SEPARATE_EMBED_INFO_FILENAME
            )
            with open(sidecar_path, "w") as info_file:
                json.dump(
                    {
                        "model_mode": self.control_args.model_mode,
                        "embedding_matrix": {
                            "file": os.path.basename(matrix_path),
                            **matrix_meta,
                        },
                        "graphs": separate_embed_info,
                    },
                    info_file,
                    indent=2,
                )
            logging.info("Saved separate embedding matrix to %s", matrix_path)
            logging.info("Saved separate embedding sidecar to %s", sidecar_path)

        example_inputs = [m.export_input for m in models if m is not None]

        # start lowering
        if self.apply_embedding:
            tok_embedding_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding for model in models],
                    )
                ),
                inputs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_example_inputs)
                ),
                compiler_specs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_data.compile_spec)
                ),
                dep_table=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_dep_table for model in models],
                    )
                ),
                passes_job=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_passes_job for model in models],
                    )
                ),
            )
            if self.control_args.verbose:
                for ep in tok_embedding_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
            )
            tok_embedding_exec_prog_mgr = tok_embedding_edge_prog_mgr.to_executorch(
                executorch_config
            )
            data = request.method_data[TEXT_EMBEDDING]
            with open(
                f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
            ) as file:
                tok_embedding_exec_prog_mgr.write_to_file(file)

        # decoder lowering
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.decoder for model in models])),
            inputs=dict(zip(graph_names, example_inputs)),
            compiler_specs=dict(zip(graph_names, data.compile_spec)),
            constant_methods={**self.prefill.meta, **self.decode.meta},
            dep_table=dict(zip(graph_names, [model.dep_table for model in models])),
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1 and self.control_args.model_mode == "kv":
            # weight-sharing based context binaries cannot be opened in x86 host
            update_spill_fill_size(edge_prog_mgr.exported_program("kv_forward"))

        if self.control_args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        data = request.method_data[TEXT_DECODER]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)


class Modality(Component):
    def __init__(
        self, control_args: argparse.Namespace, config: LLMModelConfig, modality
    ):
        self.control_args = control_args
        self.model = None
        self.modality = modality
        repo_id = config.repo_id

        if config := getattr(config, modality, None):
            if modality == TEXT_ENCODER or modality == AUDIO_ENCODER:
                raise NotImplementedError(f"{modality} is under development")

            auto_model = AutoModel.from_pretrained(
                repo_id, _attn_implementation="eager"
            )
            # Create an instance of the config class since it has init=False
            self.model = config().create_encoder(auto_model.config)
            # set strict to false to simplify parameter loading for non-text models
            auto_model = auto_model.eval()
            self.model = self.model.eval()
            self.model.load_state_dict(auto_model.state_dict(), strict=False)
            self.example_input = self.model.get_example_inputs()
            self.preprocess = self.model.preprocess

            # set quant recipe
            self.quant_recipe: EncoderQuantRecipe = (
                config.quant_recipe(True) if config.quant_recipe else None
            )

    def compile(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=self.model,
            inputs=self.example_input,
            compiler_specs=request_data.compile_spec,
        )
        if self.control_args.verbose:
            print_delegation_info(edge_prog_mgr.exported_program().graph_module)

        exec_prog_mgr = edge_prog_mgr.to_executorch(ExecutorchBackendConfig())
        data = request.method_data[self.modality]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)

    def quantize(self, request: Request):
        if self.model is None or self.quant_recipe is None:
            return

        request_data = request.method_data[self.modality]
        with torch.no_grad():
            self.model = torch.export.export(self.model, self.example_input).module()

            quantizer = make_quantizer()
            quantizer.recipe = self.quant_recipe
            self.model = prepare_pt2e(self.model, quantizer)

            # calibration
            intermediate_outputs = []
            for data in request_data.calibration_data.datasets:
                output = self.model(*self.preprocess(data))
                intermediate_outputs.append(
                    (output,) if isinstance(output, torch.Tensor) else output
                )
            # update intermediate outputs for next modality
            request_data.calibration_data.intermediate_outputs = intermediate_outputs

            self.model = convert_pt2e(self.model)

            qdq_intermediate_outputs = []
            if self.control_args.verbose:
                for data in request_data.calibration_data.datasets:
                    output = self.model(*self.preprocess(data))
                    qdq_intermediate_outputs.append(
                        (output,) if isinstance(output, torch.Tensor) else output
                    )
                # update qdq intermediate outputs for next modality
                request_data.calibration_data.qdq_intermediate_outputs = (
                    qdq_intermediate_outputs
                )


class MultiModalManager(Component):
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.audio_encoder = Modality(
            control_args,
            config,
            AUDIO_ENCODER,
        )
        self.text_encoder = Modality(
            control_args,
            config,
            TEXT_ENCODER,
        )
        self.vision_encoder = Modality(
            control_args,
            config,
            VISION_ENCODER,
        )
        self.text_decoder = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=self.audio_encoder.model or self.vision_encoder.model,
        )
        self._modalities = [
            AUDIO_ENCODER,
            TEXT_ENCODER,
            VISION_ENCODER,
            TEXT_EMBEDDING,
            TEXT_DECODER,
        ]
        # build dependency chain
        self.set_next(self.vision_encoder).set_next(self.audio_encoder).set_next(
            self.text_decoder
        )

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def compile(
        self,
        compile_specs: Dict[str, List[CompileSpec]],
        pte_filenames: Dict[str, str],
    ):
        compile_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    compile_spec=compile_specs[m],
                    pte_filename=pte_filenames[m],
                )
                for m in self._modalities
            },
        )
        self.process(compile_request)

    @log_info
    def quantize(
        self,
        calibration_data: Dict[str, List[Any]],
        tokenizer,
    ):
        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    calibration_data=Request.CalibrationData(
                        datasets=calibration_data[m]
                    ),
                    tokenizer=tokenizer,
                )
                for m in self._modalities
            },
        )
        self.process(quantize_request)
