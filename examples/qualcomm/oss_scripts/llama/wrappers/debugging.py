# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import torch
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Mode,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    BLENDER_DECODER_GRAPH_NAMES,
    DECODER_GRAPH_NAMES,
)

def _to_serializable_quant_value(graph_module, value):
    if isinstance(value, torch.fx.Node):
        if value.op == "get_attr":
            value = getattr(graph_module, value.target)
        else:
            value = value.name

    if isinstance(value, torch.Tensor):
        cpu_value = value.detach().cpu()
        return {
            "dtype": str(cpu_value.dtype),
            "shape": list(cpu_value.shape),
            "values": cpu_value.tolist(),
        }

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, (list, tuple)):
        return [
            _to_serializable_quant_value(graph_module, item) for item in value
        ]

    return value


def _to_serializable_node_args(graph_module, value):
    if isinstance(value, tuple):
        return [_to_serializable_node_args(graph_module, item) for item in value]
    if isinstance(value, list):
        return [_to_serializable_node_args(graph_module, item) for item in value]
    if isinstance(value, torch.fx.Node):
        return value.name
    if isinstance(value, torch.Tensor):
        cpu_value = value.detach().cpu()
        return {
            "dtype": str(cpu_value.dtype),
            "shape": list(cpu_value.shape),
            "values": cpu_value.tolist(),
        }
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def _maybe_get_op(op_packet, overload):
    return getattr(op_packet, overload, None)


def _get_quant_arg_indices(node):
    per_tensor_ops = {
        op
        for op in (
            _maybe_get_op(torch.ops.quantized_decomposed.quantize_per_tensor, "default"),
            _maybe_get_op(
                torch.ops.quantized_decomposed.dequantize_per_tensor, "default"
            ),
            _maybe_get_op(torch.ops.quantized_decomposed.quantize_per_tensor, "tensor"),
            _maybe_get_op(
                torch.ops.quantized_decomposed.dequantize_per_tensor, "tensor"
            ),
        )
        if op is not None
    }
    per_channel_ops = {
        op
        for op in (
            _maybe_get_op(
                torch.ops.quantized_decomposed.quantize_per_channel, "default"
            ),
            _maybe_get_op(
                torch.ops.quantized_decomposed.dequantize_per_channel, "default"
            ),
            torch.ops.torchao.dequantize_affine,
        )
        if op is not None
    }

    if node.target in per_tensor_ops:
        return 1, 2
    if node.target in per_channel_ops:
        return 1, 2
    if is_graph_output(node) and len(node.args) >= 3:
        return 1, 2
    return None


def _extract_node_quant_attrs(graph_module, node):
    quant_attr_indices = _get_quant_arg_indices(node)
    if quant_attr_indices is not None:
        scale_idx, zero_point_idx = quant_attr_indices
        return {
            "scale": _to_serializable_quant_value(graph_module, node.args[scale_idx]),
            "zero_point": _to_serializable_quant_value(
                graph_module, node.args[zero_point_idx]
            ),
        }

    if node.op == "placeholder" and len(node.users) == 1:
        user = next(iter(node.users))
        quant_attr_indices = _get_quant_arg_indices(user)
        if quant_attr_indices is not None:
            scale_idx, zero_point_idx = quant_attr_indices
            return {
                "scale": _to_serializable_quant_value(
                    graph_module, user.args[scale_idx]
                ),
                "zero_point": _to_serializable_quant_value(
                    graph_module, user.args[zero_point_idx]
                ),
            }

    return {"scale": None, "zero_point": None}


def _store_graph_quant_attrs(graph_module, graph_name, artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)
    graph_quant_attrs = {}
    for node in graph_module.graph.nodes:
        quant_attrs = _extract_node_quant_attrs(graph_module, node)
        graph_quant_attrs[node.name] = {
            "op": node.op,
            "target": str(node.target),
            "args": _to_serializable_node_args(graph_module, list(node.args)),
            "scale": quant_attrs["scale"],
            "zero_point": quant_attrs["zero_point"],
        }

    output_path = os.path.join(artifact_dir, f"{graph_name}_quant_attrs.json")
    with open(output_path, "w") as file:
        json.dump(
            {
                "graph_name": graph_name,
                "nodes": graph_quant_attrs,
            },
            file,
            indent=2,
        )
    logging.info("Stored graph quantization attributes to %s", output_path)


def get_graph_name(mode):
    graph_names = {
        Mode.DECODE: DECODER_GRAPH_NAMES[0],
        Mode.PREFILL: DECODER_GRAPH_NAMES[1],
        Mode.BLENDER: BLENDER_DECODER_GRAPH_NAMES[-1],
    }
    return graph_names.get(mode, f"{mode.name.lower()}_forward")
