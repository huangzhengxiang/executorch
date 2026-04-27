/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/separate_embed.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/assert.h>
#include <ops/ops.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace example {
namespace {
constexpr uint32_t kDtypeFloat32 = 1;
constexpr uint32_t kDtypeFloat16 = 2;
constexpr uint32_t kDtypeInt8 = 3;
constexpr uint32_t kDtypeUInt8 = 4;
constexpr uint32_t kDtypeInt16 = 5;
constexpr uint32_t kDtypeUInt16 = 6;
constexpr uint32_t kDtypeInt32 = 7;
constexpr uint32_t kDtypeInt64 = 8;
constexpr uint32_t kDtypeInt4 = 9;

template <typename U>
U load_scalar(const std::vector<uint8_t>& data) {
  ET_CHECK_MSG(
      data.size() >= sizeof(U),
      "Tensor block scalar data is too small: expected at least %zu bytes, got %zu bytes.",
      sizeof(U),
      data.size());
  U value;
  std::memcpy(&value, data.data(), sizeof(U));
  return value;
}
} // namespace

bool SeparateEmbedding::read_tensor_block(
    std::ifstream& matrix_file,
    TensorBlock* block,
    bool load_payload) {
  uint32_t dtype_code = 0;
  uint32_t ndim = 0;
  uint64_t nbytes = 0;
  matrix_file.read(reinterpret_cast<char*>(&dtype_code), sizeof(dtype_code));
  matrix_file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
  matrix_file.read(reinterpret_cast<char*>(&nbytes), sizeof(nbytes));
  if (!matrix_file.good()) {
    return false;
  }

  block->dtype_code = dtype_code;
  block->shape.clear();
  block->nbytes = nbytes;
  block->data_offset = 0;
  block->data.clear();
  if (dtype_code == 0 && nbytes == 0) {
    return true;
  }

  block->shape.resize(ndim);
  if (ndim > 0) {
    matrix_file.read(
        reinterpret_cast<char*>(block->shape.data()),
        sizeof(uint32_t) * ndim);
    if (!matrix_file.good()) {
      return false;
    }
  }

  block->data_offset = static_cast<uint64_t>(matrix_file.tellg());
  if (nbytes > 0) {
    if (load_payload) {
      block->data.resize(nbytes);
      matrix_file.read(reinterpret_cast<char*>(block->data.data()), nbytes);
      if (!matrix_file.good()) {
        return false;
      }
    } else {
      matrix_file.seekg(static_cast<std::streamoff>(nbytes), std::ios::cur);
      if (!matrix_file.good()) {
        return false;
      }
    }
  }
  return true;
}

void SeparateEmbedding::validate_shape_or_throw(
    const TensorBlock& block,
    const std::string& block_name) const {
  if (block.shape.size() != 2) {
    throw std::runtime_error(block_name + " must be a 2D matrix.");
  }
}

bool SeparateEmbedding::load(const std::string& matrix_path) {
  if (matrix_fd_ >= 0) {
    close(matrix_fd_);
    matrix_fd_ = -1;
  }

  loaded_ = false;
  quantized_ = false;
  vocab_size_ = 0;
  embedding_dim_ = 0;
  qweight_ = TensorBlock{};
  scale_ = TensorBlock{};
  zp_ = TensorBlock{};
  weight_ = TensorBlock{};

  std::ifstream matrix_file(matrix_path.c_str(), std::ios::binary);
  if (!matrix_file.is_open()) {
    ET_LOG(Error, "Failed to open separate embedding matrix %s", matrix_path.c_str());
    return false;
  }

  char magic[4];
  uint32_t version = 0;
  uint32_t quantized_flag = 0;
  matrix_file.read(magic, sizeof(magic));
  matrix_file.read(reinterpret_cast<char*>(&version), sizeof(version));
  matrix_file.read(reinterpret_cast<char*>(&quantized_flag), sizeof(quantized_flag));
  if (!matrix_file.good()) {
    ET_LOG(Error, "Failed to read matrix header from %s", matrix_path.c_str());
    return false;
  }
  if (std::memcmp(magic, "SEMB", 4) != 0 || version != 1) {
    ET_LOG(Error, "Unsupported separate embedding matrix format in %s", matrix_path.c_str());
    return false;
  }

  quantized_ = quantized_flag != 0;
  if (quantized_) {
    if (!read_tensor_block(matrix_file, &qweight_, false) ||
        !read_tensor_block(matrix_file, &scale_, true) ||
        !read_tensor_block(matrix_file, &zp_, true)) {
      ET_LOG(Error, "Failed to read quantized embedding blocks from %s", matrix_path.c_str());
      return false;
    }
    try {
      validate_shape_or_throw(qweight_, "qweight");
    } catch (const std::runtime_error& err) {
      ET_LOG(Error, "%s", err.what());
      return false;
    }
    vocab_size_ = static_cast<int32_t>(qweight_.shape[0]);
    embedding_dim_ = static_cast<int32_t>(qweight_.shape[1]);
  } else {
    if (!read_tensor_block(matrix_file, &weight_, false) ||
        !read_tensor_block(matrix_file, &qweight_, false) ||
        !read_tensor_block(matrix_file, &scale_, true) ||
        !read_tensor_block(matrix_file, &zp_, true)) {
      ET_LOG(Error, "Failed to read floating embedding blocks from %s", matrix_path.c_str());
      return false;
    }
    try {
      validate_shape_or_throw(weight_, "weight");
    } catch (const std::runtime_error& err) {
      ET_LOG(Error, "%s", err.what());
      return false;
    }
    vocab_size_ = static_cast<int32_t>(weight_.shape[0]);
    embedding_dim_ = static_cast<int32_t>(weight_.shape[1]);
  }

  matrix_fd_ = open(matrix_path.c_str(), O_RDONLY);
  if (matrix_fd_ < 0) {
    ET_LOG(
        Error,
        "Failed to open separate embedding matrix fd %s",
        matrix_path.c_str());
    return false;
  }

  loaded_ = true;
  ET_LOG(
      Info,
      "Loaded separate embedding matrix: quantized=%d, vocab=%d, dim=%d (row-on-demand io)",
      quantized_,
      vocab_size_,
      embedding_dim_);
  return true;
}

SeparateEmbedding::~SeparateEmbedding() {
  if (matrix_fd_ >= 0) {
    close(matrix_fd_);
    matrix_fd_ = -1;
  }
}

uint32_t SeparateEmbedding::embedding_dtype_code() const {
  return quantized_ ? qweight_.dtype_code : weight_.dtype_code;
}

size_t SeparateEmbedding::embedding_elem_size() const {
  const uint32_t dtype_code = embedding_dtype_code();
  switch (dtype_code) {
    case kDtypeFloat16:
    case kDtypeUInt16:
    case kDtypeInt16:
      return sizeof(uint16_t);
    case kDtypeFloat32:
    case kDtypeInt32:
      return sizeof(uint32_t);
    case kDtypeUInt8:
    case kDtypeInt8:
    case kDtypeInt4:
      return sizeof(uint8_t);
    default:
      ET_CHECK_MSG(false, "Unsupported embedding dtype code: %u", dtype_code);
      return 0;
  }
}

size_t SeparateEmbedding::row_bytes() const {
  if (embedding_dtype_code() == kDtypeInt4) {
    return (static_cast<size_t>(embedding_dim_) + 1) / 2;
  }
  return static_cast<size_t>(embedding_dim_) * embedding_elem_size();
}

void SeparateEmbedding::copy_row(
    uint64_t token_id,
    uint8_t* out_embedding_row,
    size_t bytes) const {
  ET_CHECK_MSG(loaded_, "Separate embedding table is not loaded.");
  ET_CHECK_MSG(out_embedding_row != nullptr, "Output embedding row is null.");
  ET_CHECK_MSG(
      token_id < static_cast<uint64_t>(vocab_size_),
      "Token id %llu out of vocab range %d",
      static_cast<unsigned long long>(token_id),
      vocab_size_);
  ET_CHECK_MSG(bytes == row_bytes(), "Invalid row size for embedding copy.");
  const TensorBlock& table = quantized_ ? qweight_ : weight_;
  const uint64_t src_offset =
      table.data_offset + static_cast<uint64_t>(token_id) * bytes;
  ET_CHECK_MSG(
      src_offset + bytes <= table.data_offset + table.nbytes,
      "Embedding row offset out of range.");
  ET_CHECK_MSG(
      read_from_file(src_offset, out_embedding_row, bytes),
      "Failed to read embedding row from matrix file.");
}

float SeparateEmbedding::quant_scale() const {
  ET_CHECK_MSG(loaded_ && quantized_, "Quantized embedding table is not loaded.");
  switch (scale_.dtype_code) {
    case kDtypeFloat32:
      return load_scalar<float>(scale_.data);
    case kDtypeInt32:
      return static_cast<float>(load_scalar<int32_t>(scale_.data));
    case kDtypeUInt16:
      return static_cast<float>(load_scalar<uint16_t>(scale_.data));
    case kDtypeInt16:
      return static_cast<float>(load_scalar<int16_t>(scale_.data));
    case kDtypeUInt8:
      return static_cast<float>(load_scalar<uint8_t>(scale_.data));
    case kDtypeInt8:
      return static_cast<float>(load_scalar<int8_t>(scale_.data));
    default:
      ET_CHECK_MSG(false, "Unsupported scale dtype code: %u", scale_.dtype_code);
      return 0.f;
  }
}

int32_t SeparateEmbedding::quant_zero_point() const {
  ET_CHECK_MSG(loaded_ && quantized_, "Quantized embedding table is not loaded.");
  switch (zp_.dtype_code) {
    case kDtypeInt32:
      return load_scalar<int32_t>(zp_.data);
    case kDtypeInt64:
      return static_cast<int32_t>(load_scalar<int64_t>(zp_.data));
    case kDtypeUInt16:
      return static_cast<int32_t>(load_scalar<uint16_t>(zp_.data));
    case kDtypeInt16:
      return static_cast<int32_t>(load_scalar<int16_t>(zp_.data));
    case kDtypeUInt8:
      return static_cast<int32_t>(load_scalar<uint8_t>(zp_.data));
    case kDtypeInt8:
      return static_cast<int32_t>(load_scalar<int8_t>(zp_.data));
    default:
      ET_CHECK_MSG(false, "Unsupported zero-point dtype code: %u", zp_.dtype_code);
      return 0;
  }
}

void SeparateEmbedding::dequantize_row_to_float(
    uint64_t token_id,
    float* out_embedding_row,
    size_t elems) const {
  ET_CHECK_MSG(loaded_ && quantized_, "Quantized embedding table is not loaded.");
  ET_CHECK_MSG(out_embedding_row != nullptr, "Output embedding row is null.");
  ET_CHECK_MSG(
      token_id < static_cast<uint64_t>(vocab_size_),
      "Token id %llu out of vocab range %d",
      static_cast<unsigned long long>(token_id),
      vocab_size_);
  ET_CHECK_MSG(
      elems == static_cast<size_t>(embedding_dim_),
      "Invalid row element count for embedding dequantize.");

  const size_t bytes = row_bytes();
  const uint64_t src_offset =
      qweight_.data_offset + static_cast<uint64_t>(token_id) * bytes;
  ET_CHECK_MSG(
      src_offset + bytes <= qweight_.data_offset + qweight_.nbytes,
      "Embedding row offset out of range.");
  std::vector<uint8_t> row_buffer(bytes);
  ET_CHECK_MSG(
      read_from_file(src_offset, row_buffer.data(), bytes),
      "Failed to read quantized embedding row from matrix file.");
  const uint8_t* src = row_buffer.data();
  const float scale = quant_scale();
  const int32_t zero_point = quant_zero_point();
  const auto dequant_start = std::chrono::steady_clock::now();

  switch (qweight_.dtype_code) {
    case kDtypeUInt8:
      LMStore::scalar_dequantize_per_tensor_u8(
          reinterpret_cast<const uint8_t*>(src),
          out_embedding_row,
          elems,
          scale,
          zero_point);
      break;
    case kDtypeInt8:
      LMStore::scalar_dequantize_per_tensor_i8(
          reinterpret_cast<const int8_t*>(src),
          out_embedding_row,
          elems,
          scale,
          zero_point);
      break;
    case kDtypeUInt16:
      LMStore::scalar_dequantize_per_tensor_u16(
          reinterpret_cast<const uint16_t*>(src),
          out_embedding_row,
          elems,
          scale,
          zero_point);
      break;
    case kDtypeInt16:
      LMStore::scalar_dequantize_per_tensor_i16(
          reinterpret_cast<const int16_t*>(src),
          out_embedding_row,
          elems,
          scale,
          zero_point);
      break;
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported embedding dtype code for row dequantize: %u",
          qweight_.dtype_code);
      break;
  }
  const auto dequant_end = std::chrono::steady_clock::now();
  const uint64_t dequant_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          dequant_end - dequant_start)
          .count());
  ET_LOG(
      Info,
      "separate_embed dequant: token=%llu elems=%llu cost=%lluus",
      static_cast<unsigned long long>(token_id),
      static_cast<unsigned long long>(elems),
      static_cast<unsigned long long>(dequant_us));
}

bool SeparateEmbedding::read_from_file(
    uint64_t offset,
    void* dst,
    size_t bytes) const {
  ET_CHECK_MSG(matrix_fd_ >= 0, "Embedding matrix fd is not open.");
  const auto read_start = std::chrono::steady_clock::now();
  uint8_t* out = reinterpret_cast<uint8_t*>(dst);
  size_t total = 0;
  while (total < bytes) {
    const ssize_t got = pread(
        matrix_fd_,
        out + total,
        bytes - total,
        static_cast<off_t>(offset + total));
    if (got <= 0) {
      return false;
    }
    total += static_cast<size_t>(got);
  }
  const auto read_end = std::chrono::steady_clock::now();
  const uint64_t read_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          read_end - read_start)
          .count());
  ET_LOG(
      Info,
      "separate_embed read: offset=%llu bytes=%llu cost=%lluus",
      static_cast<unsigned long long>(offset),
      static_cast<unsigned long long>(bytes),
      static_cast<unsigned long long>(read_us));
  return true;
}

} // namespace example
