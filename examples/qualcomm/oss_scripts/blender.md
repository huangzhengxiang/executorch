## Mobile Blend
Only support model with `qk_norm_before_rope` currently. (post norm not supported.)

### attention mask
If it has 3 past tokens, CL = 9 and ar_len = 3, and inputs another 2 tokens (padded to 3), ExecuTorch original attention mask shall be as though:

```python
# attention mask
# prefill
"""
  0    1    2  |   3    4    5  |   6    7    8 
------------------------------------------------
  0    0    0  | -255 -255 -255 |   0  -255 -255
  0    0    0  | -255 -255 -255 |   0    0  -255
-255 -255 -255 | -255 -255 -255 | -255 -255 -255
"""
# after which, decode
"""
  0  1  2  |  3  4   5  |   6    7   8 
---------------------------------------
  0  0  0  |  0  0 -255 | -255 -255  0
"""
```

For blender, kv caches of inputs are also concatenated to the last.
```python
# concatenation

# computation
diff_k = torch.sum(
    (k.transpose(2, 3) - k_caches[:, :, :, -self.ar_len:]) ** 2, dim=[1, 2],
    keepdim=False
)
```


### export
Is mixed precision possible? for diff_k related?
When exporting prefill module, store the KV cache at the end of prefill. 
When exporting blender module, assert chunk>=3, load chunk0, skip chunk1, and reuse the followings continuously.
For blender, skip _generate calibration.


### command

```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -c -m ${SOC_MODEL} --temperature 0 --model_mode blender --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen3-1_7b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```