/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/runtime/platform/assert.h>
namespace example {
template <typename T>
KVManager<T>::KVManager(Metadata metadata) : metadata_(metadata) {
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);

  // Calculate cache size
  size_t cache_in_bytes = metadata_.num_layers * metadata_.num_heads *
      metadata_.head_dim * metadata_.max_cache_len * sizeof(T);
  size_t cache_out_bytes = metadata_.num_layers * metadata_.num_heads *
      metadata_.head_dim * metadata_.max_ar_len * sizeof(T);
  total_cache_size_ = 2 * (cache_in_bytes + cache_out_bytes);
};

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  uint16_t* past_ptr = attention_mask;
  uint16_t* new_ptr = attention_mask + (metadata_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      uint16_t* parent_ptr = attention_mask + pidx * metadata_.context_len;
      std::memcpy(
          past_ptr, parent_ptr, metadata_.context_len * sizeof(uint16_t));
    }
    // Attend to itself
    new_ptr[i] = pos_val;
    past_ptr += metadata_.context_len;
    new_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  uint16_t* past_ptr = attention_mask;
  uint16_t* new_ptr = attention_mask + (metadata_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      uint16_t* parent_ptr = attention_mask + pidx * metadata_.context_len;
      std::memcpy(
          past_ptr, parent_ptr, metadata_.context_len * sizeof(uint16_t));
    }
    // Attend to itself
    new_ptr[i] = pos_val;

    // mask by limitation of sliding_window
    int32_t available_context_len = position_offset.empty()
        ? sliding_window - (i + 1) - n_past
        : sliding_window - (position_offset[i] + 1) - n_past;
    // if available_context_len is less than 0, it means we need to mask some
    // tokens in the past to avoid exceeding the sliding window
    if (available_context_len < 0) {
      std::fill_n(past_ptr, -available_context_len, neg_val);
    }

    past_ptr += metadata_.context_len;
    new_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  uint16_t pos_val = 65535;
  uint16_t* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    cur_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  uint16_t pos_val = 65535;
  uint16_t neg_val = 0;
  uint16_t* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    int32_t available_cache_len = position_offset.empty()
        ? sliding_window - (i + 1)
        : sliding_window - (position_offset[i] + 1);
    if (n_past + n_update > available_cache_len) {
      std::fill_n(
          cur_ptr - n_past, n_past + n_update - available_cache_len, neg_val);
    }
    cur_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  const size_t max_in_cache_block_in_bytes =
      metadata_.max_cache_len * sizeof(T);
  const size_t max_out_cache_block_in_bytes = metadata_.max_ar_len * sizeof(T);

  const size_t cache_in_bytes =
      metadata_.num_heads * metadata_.head_dim * max_in_cache_block_in_bytes;
  const size_t cache_out_bytes =
      metadata_.num_heads * metadata_.head_dim * max_out_cache_block_in_bytes;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // Allocate buffer for key cache and value cache
    T* single_layer_k_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_k_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));
    T* single_layer_v_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_v_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));

    k_cache_[layer].buffer = single_layer_k_cache_in;
    k_cache_[layer].output_buffer = single_layer_k_cache_out;
    v_cache_[layer].buffer = single_layer_v_cache_in;
    v_cache_[layer].output_buffer = single_layer_v_cache_out;
  }
}

template <typename T>
void KVManager<T>::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    rearrange_key(k_cache_[layer], ar_len_dst);
    rearrange_value(v_cache_[layer], ar_len_dst);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

template <typename T>
void KVManager<T>::rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* k_cache_in_read_ptr = k_cache.buffer;
  T* k_cache_in_write_ptr = k_cache.buffer;

  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(
          k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_num * sizeof(T));
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
  } else {
    k_cache_in_read_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * src_cache_num;
    k_cache_in_write_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(
          k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_num * sizeof(T));
      k_cache_in_read_ptr -= src_cache_num;
      k_cache_in_write_ptr -= dst_cache_num;
    }
  }
}

template <typename T>
void KVManager<T>::rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* v_cache_in_read_ptr = v_cache.buffer;
  T* v_cache_in_write_ptr = v_cache.buffer;
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          dst_cache_num * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr += src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr += dst_cache_num * metadata_.head_dim;
    }
  } else {
    v_cache_in_read_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * src_cache_num;
    v_cache_in_write_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          src_cache_num * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr -= src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr -= dst_cache_num * metadata_.head_dim;
    }
  }
}

template <typename T>
void KVManager<T>::update_cache(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // ET_LOG(Info, "kv output layer: %d", layer);
    update_key(k_cache_[layer], n_past, n_update, selected);
    update_value(v_cache_[layer], n_past, n_update, selected);
  }
}

template <typename T>
void KVManager<T>::transfer_cache(
    std::vector<T*>& k_buffer,
    std::vector<T*>& v_buffer,
    int buffer_seq_dim,
    int copy_len,
    int32_t n_past,
    bool directionM2D, // true: M2D, false: D2M
    int disk_start_id,
    int mem_start_id) {
  int32_t disk_cache_num = buffer_seq_dim;
  int32_t mem_cache_num = (cur_ar_len_ == metadata_.context_len)
          ? metadata_.context_len
          : metadata_.context_len - cur_ar_len_;
  int32_t src_cache_num;
  int32_t dst_cache_num;
  // mem_start_id += mem_cache_num - n_past; // reverse?
  T *k_cache_in_read_ptr, *k_cache_in_write_ptr, *v_cache_in_read_ptr, *v_cache_in_write_ptr;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    if (directionM2D) {
      // M2D, from cache to buffer
      src_cache_num = mem_cache_num;
      dst_cache_num = disk_cache_num;
      k_cache_in_read_ptr = k_cache_[layer].buffer + mem_start_id;
      k_cache_in_write_ptr = k_buffer[layer] + disk_start_id;
      v_cache_in_read_ptr = v_cache_[layer].buffer + mem_start_id * metadata_.head_dim;
      v_cache_in_write_ptr = v_buffer[layer] + disk_start_id * metadata_.head_dim;
    } else {
      // D2M, from buffer to cache
      src_cache_num = disk_cache_num;
      dst_cache_num = mem_cache_num;
      k_cache_in_read_ptr = k_buffer[layer] + disk_start_id;
      k_cache_in_write_ptr = k_cache_[layer].buffer + mem_start_id;
      v_cache_in_read_ptr = v_buffer[layer] + disk_start_id * metadata_.head_dim;
      v_cache_in_write_ptr = v_cache_[layer].buffer + mem_start_id * metadata_.head_dim;
    }
    // copy k
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memcpy(
          k_cache_in_write_ptr, k_cache_in_read_ptr, copy_len * sizeof(T));
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
    // copy v
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memcpy(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          copy_len * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr += src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr += dst_cache_num * metadata_.head_dim;
    }
  }
}

template <typename T>
void KVManager<T>::update_key(
    KVCache<T>& k_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * sizeof(T);
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t out_size = cur_ar_len_;
  const int32_t past_size = n_past;
  const int32_t n_iter = metadata_.head_dim * metadata_.num_heads;

  write_ptr += past_size;
  if (selected.empty()) {
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      // for (int j = 0; j < n_update; ++j) {
      //   ET_LOG(Info, "kv_caches[%d, %d]: %d", i, j, (int)write_ptr[j]);
      // }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    std::vector<int32_t> true_indices(n_update);
    for (int i = 0, j = 0; i < selected.size() && j < n_update; ++i) {
      if (selected[i]) {
        true_indices[j++] = i;
      }
    }
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto ind : true_indices) {
        *wp++ = rp[ind];
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template <typename T>
void KVManager<T>::update_value(
    KVCache<T>& v_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.output_buffer;
  const int32_t copy_size = n_update * metadata_.head_dim * sizeof(T);
  const int32_t past_size = n_past * metadata_.head_dim;
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * metadata_.head_dim
      : (metadata_.context_len - cur_ar_len_) * metadata_.head_dim;
  const int32_t out_size = cur_ar_len_ * metadata_.head_dim;

  write_ptr += past_size;

  if (selected.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    int32_t update_times = n_update;
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto sel : selected) {
        if (sel) {
          std::memcpy(wp, rp, metadata_.head_dim * sizeof(T));
          wp += metadata_.head_dim;
          update_times--;
          if (update_times == 0)
            break;
        }
        rp += metadata_.head_dim;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template <typename T>
void KVManager<T>::blender_update_cache(
      int32_t ar_len,
      int32_t n_past,
      int32_t blend_len,
      int32_t* imp_indices) {
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // ET_LOG(Info, "kv output layer: %d", layer);
    blender_update_key(k_cache_[layer], ar_len, n_past, blend_len, imp_indices);
    blender_update_value(v_cache_[layer], ar_len, n_past, blend_len, imp_indices);
  }
}

template <typename T>
void KVManager<T>::blender_update_key(
    KVCache<T>& k_cache,
    int32_t ar_len,
    int32_t n_past,
    int32_t blend_len,
    int32_t* imp_indices) {
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.buffer;
  const int32_t copy_size = ar_len * sizeof(T);
  const int32_t iter_size = metadata_.context_len;
  const int32_t past_size = n_past;
  const int32_t n_iter = metadata_.head_dim * metadata_.num_heads;
  
  // k_caches[layer][:,:,:,start_pos:start_pos+ar_len] = k_caches[layer][:,:,:,-ar_len:]
  const int32_t back_pos = metadata_.context_len - ar_len;
  write_ptr += past_size;
  read_ptr += back_pos;
  for (int i = 0; i < n_iter; ++i) {
    std::memmove(write_ptr, read_ptr, copy_size);
    write_ptr += iter_size;
    read_ptr += iter_size;
  }

  // scatter: k_caches[layer] = k_caches[layer].scatter(dim=3, src=new_k_caches[layer], index=cache_indices)
  write_ptr = k_cache.buffer + past_size;
  read_ptr = k_cache.output_buffer;
  const int32_t out_size = blend_len;
  for (int i = 0; i < n_iter; ++i) {
    for (int j = 0; j < blend_len; ++j) {
      write_ptr[imp_indices[j]] = read_ptr[j];
    }
    // for (int j = 0; j < blend_len; ++j) {
    //   ET_LOG(Info, "kv_caches[%d, %d]: %d", i, j, (int)write_ptr[j]);
    // }
    write_ptr += iter_size;
    read_ptr += out_size;
  }
}

template <typename T>
void KVManager<T>::blender_update_value(
    KVCache<T>& v_cache,
    int32_t ar_len,
    int32_t n_past,
    int32_t blend_len,
    int32_t* imp_indices) {
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.buffer;
  const int32_t copy_size = ar_len * metadata_.head_dim * sizeof(T);
  const int32_t past_size = n_past * metadata_.head_dim;
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = metadata_.context_len * metadata_.head_dim;

  // v_caches[layer][:,:,start_pos:start_pos+ar_len,:] = v_caches[layer][:,:,-ar_len:,:]
  const int32_t back_pos = (metadata_.context_len - ar_len)  * metadata_.head_dim;
  write_ptr += past_size;
  read_ptr += back_pos;
  for (int i = 0; i < n_iter; i++) {
    std::memmove(write_ptr, read_ptr, copy_size);
    write_ptr += iter_size;
    read_ptr += iter_size;
  }

  // scatter: v_caches[layer] = v_caches[layer].scatter(dim=2, src=new_v_caches[layer], index=cache_indices.transpose(2, 3))
  write_ptr = v_cache.buffer + past_size;
  read_ptr = v_cache.output_buffer;
  const int32_t out_size = blend_len * metadata_.head_dim;
  for (int i = 0; i < n_iter; i++) {
    for (int j = 0; j < blend_len; j++) {
      std::memcpy(write_ptr + imp_indices[j] * metadata_.head_dim, 
                  read_ptr + j * metadata_.head_dim, 
                  metadata_.head_dim * sizeof(T));
    }
    write_ptr += iter_size;
    read_ptr += out_size;
  }
}

// Explicit instantiations
template class KVManager<uint16_t>;
template class KVManager<uint8_t>;

} // namespace example
