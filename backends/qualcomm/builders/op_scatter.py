from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import torch

from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpScatterElements, QNN_OP_PACKAGE_NAME_QTI_AISW
from executorch.backends.qualcomm.utils.constants import QCOM_DATA



@register_node_visitor
class ScatterVisitor(NodeVisitor):
    target = ["aten.scatter.src"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        index_node = self.get_node(node.args[2])
        index_tensor = self.get_tensor(index_node, node)
        index_tensor_wrapper = self.define_tensor(
            index_node,
            node,
            index_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # src tensor
        src_node = self.get_node(node.args[3])
        src_tensor = self.get_tensor(src_node, node)
        src_tensor_wrapper = self.define_tensor(
            src_node,
            node,
            src_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        
        # axis
        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        scatter_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpScatterElements.op_name,   # QNN op name
        )

        scatter_op.AddInputTensors(
            [
                input_tensor_wrapper,
                index_tensor_wrapper,
                src_tensor_wrapper,
            ]
        )

        scatter_op.AddOutputTensors([output_tensor_wrapper])

        # axis attribute
        scatter_op.AddScalarParam(
            OpScatterElements.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: dim},
        )

        return scatter_op
