---
name: llama-qnn-debug
description: Debug Qualcomm LLaMA/Qwen intermediate tensors with QNN dumps. Use when exporting `examples/qualcomm/oss_scripts/llama/llama.py`, running `qnn_llama_runner` with `--dump_intermediate_outputs`, inspecting `etdump*.etdp` + `debug_*.bin`, matching blender vs hybrid nodes, or validating gather/permute/softmax/scatter paths.
---

# LLaMA QNN Debug

Use this skill for the Qualcomm OSS LLaMA pipeline under `examples/qualcomm/oss_scripts/llama`.

## Scope
- Export with QNN intermediate dump enabled
- Run `qnn_llama_runner` and collect `etdump.etdp` + `debug_output.bin`
- Inspect dumped tensors with `executorch.devtools.Inspector`
- Compare blender vs hybrid tensors under gather semantics
- Validate node-by-node paths such as:
  - `aten_gather_default`
  - `aten_permute_copy_default`
  - `quantized_decomposed_dequantize_per_tensor_*`
  - `aten__softmax_default*`
  - `aten_scatter_src*`

## Export
- `examples/qualcomm/oss_scripts/llama/llama.py` already accepts `-D/--dump_intermediate_outputs`
- This only affects compile specs. Runtime dumping still requires runner flags.

Example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -c -m ${SOC_MODEL} \
  --model_mode blender \
  --prefill_ar_len 128 \
  --max_seq_len 1024 \
  --decoder_model qwen3-1_7b \
  -D \
  --prompt "Hello"
```

## Runner
- Use the patched `build-android/examples/qualcomm/oss_scripts/llama/qnn_llama_runner`
- Required flags:
  - `--dump_intermediate_outputs`
  - `--etdump_path`
  - `--debug_output_path`
  - optionally `--debug_buffer_size`

Example:
```bash
./qnn_llama_runner \
  --model_path blender_llama_qnn.pte \
  --tokenizer_path tokenizer.json \
  --decoder_model_version qwen3 \
  --seq_len 10 \
  --prompt Hello \
  --eval_mode 3 \
  --dump_intermediate_outputs \
  --etdump_path etdump_blender.etdp \
  --debug_output_path debug_blender.bin
```

## Environment
- Before inspecting `etdp` / `bin`, use the repo's `llm` conda env and Qualcomm toolchain exports
- Recommended setup:

```bash
conda activate llm
export QNN_SDK_ROOT=/root/autodl-tmp/tools/qairt/2.37.0.250724/
export ANDROID_NDK_ROOT=/root/autodl-tmp/tools/android-ndk-r27d/
export EXECUTORCH_ROOT=/root/autodl-tmp/executorch/
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
export PYTHONPATH=$EXECUTORCH_ROOT/..
export SOC_MODEL=SM8750
export CUDA_VISIBLE_DEVICES=-1
cd $EXECUTORCH_ROOT
```

## Inspector Pattern
```python
from executorch.devtools import Inspector

inspector = Inspector(
    etdump_path="llama_qnn/etdump_blender.etdp",
    debug_buffer_path="llama_qnn/debug_blender.bin",
)

for block in inspector.event_blocks:
    for event in block.events:
        if str(event.name) == "aten_gather_default@0":
            tensor = event.debug_data[0]
```

## Known Conventions
- Dumped QNN tensor names often end with `@0`
- Blender `output_aten_where_self_1@0` is the gather index tensor for the `32 <- 128` reduction
- For shape mismatch `32` vs `128`, compare in gathered semantics:
  - `idx = output_aten_where_self_1@0`
  - `torch.gather(hybrid_tensor, dim=2 or 3, index=expanded_idx)`
- `read_ptr.txt` style dumps may require layout-specific transforms before comparison
- For the `aten_scatter_src_h_*` path, the `9`-wide slice on the `1024` dim must be reversed before matching `output_aten_permute_copy_default_10@0`

## Established Correspondences
- `blender output_aten_permute_copy_default_10@0`
  matches gathered `hybrid output_aten_permute_copy_default_7_sha_concat@0`
- `blender output_aten_gather_default_1_sha_concat@0`
  matches gathered `hybrid output_aten_permute_copy_default_6_sha_concat@0`

## Softmax Guidance
- Compare softmax families with the same gather semantics
- Global match can look good while the hotspot window is bad
- Always inspect the local window if debugging boundary effects:
  - rows `0:9`
  - cols `896:905`

## Save Tensor Quickly
```python
from pathlib import Path
import torch

torch.save(tensor, Path("llama_qnn/node.pt"))
with open("llama_qnn/node.txt", "w") as f:
    for a in range(tensor.shape[0]):
        for b in range(tensor.shape[1]):
            for c in range(tensor.shape[2]):
                for d in range(tensor.shape[3]):
                    f.write(f"[{a}, {b}, {c}, {d}]: {int(tensor[a,b,c,d])}\n")
```

## Working Rule
- Do not compare blender `32` tensors directly to hybrid `128` tensors
- First align semantics:
  - gather
  - transpose/permute
  - reverse if the path is known to reverse
  - then compare
