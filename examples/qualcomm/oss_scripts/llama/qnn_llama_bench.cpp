/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Benchmark tool for Qualcomm Llama runner. It sweeps prefill length and
 * decode length combinations using dummy prompt tokens and reports throughput.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/tokenizer.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

DEFINE_string(decoder_model_version, "llama2", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model serialized in flatbuffer format.");
DEFINE_string(
    tokenizer_path,
    "tokenizer.bin",
    "Tokenizer path. Kept for compatibility with the runner interface.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Path used by runner internals to store speed for each run.");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; 0 = greedy argmax sampling (deterministic).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding / 3: BlenderMode");
DEFINE_bool(
    shared_buffer,
    false,
    "Use shared buffers for zero-copy use case between app and device.");
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] n-gram size used in the lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Number of future tokens attempted per step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Max speculation candidates considered per step.");
DEFINE_bool(
    kv_store,
    false,
    "Store and reload KV cache from disk.");
DEFINE_int32(
    test_level,
    false,
    "Debug test level for runner internals.");
DEFINE_int32(
    blend_len,
    32,
    "Blend length for BlenderMode.");
DEFINE_string(
    prefill_lengths,
    "32,128,512",
    "Comma-separated prefill token lengths to benchmark.");
DEFINE_string(
    generation_lengths,
    "32,128,256",
    "Comma-separated decode step counts to benchmark.");
DEFINE_int32(num_iters, 1, "Measured iterations per (prefill, generation) pair.");
DEFINE_int32(warmup_iters, 0, "Warmup iterations per (prefill, generation) pair.");
DEFINE_uint64(
    dummy_token_id,
    1,
    "Token id used to build dummy prompt tokens for prefill.");
DEFINE_string(
    output_path,
    "qnn_llama_bench.csv",
    "CSV output path for benchmark results.");

namespace {

double safe_rate(double token_count, double time_ms) {
  return time_ms > 0.0 ? token_count * 1000.0 / time_ms : 0.0;
}

std::string trim(std::string value) {
  value.erase(
      value.begin(),
      std::find_if(
          value.begin(), value.end(), [](unsigned char c) { return !isspace(c); }));
  value.erase(
      std::find_if(
          value.rbegin(), value.rend(), [](unsigned char c) { return !isspace(c); })
          .base(),
      value.end());
  return value;
}

std::vector<int32_t> parse_int_list(
    const std::string& csv,
    const char* flag_name,
    bool allow_zero) {
  std::vector<int32_t> values;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim(token);
    ET_CHECK_MSG(
        !token.empty(), "Empty value found in --%s: %s", flag_name, csv.c_str());

    char* end_ptr = nullptr;
    long parsed = std::strtol(token.c_str(), &end_ptr, 10);
    ET_CHECK_MSG(
        end_ptr != token.c_str() && *end_ptr == '\0',
        "Invalid integer value '%s' in --%s",
        token.c_str(),
        flag_name);
    ET_CHECK_MSG(
        parsed <= std::numeric_limits<int32_t>::max(),
        "Value '%s' in --%s is too large",
        token.c_str(),
        flag_name);
    ET_CHECK_MSG(
        allow_zero ? (parsed >= 0) : (parsed > 0),
        "Value '%s' in --%s must be %s",
        token.c_str(),
        flag_name,
        allow_zero ? ">= 0" : "> 0");
    values.push_back(static_cast<int32_t>(parsed));
  }
  ET_CHECK_MSG(!values.empty(), "--%s cannot be empty", flag_name);
  return values;
}

class BenchTokenizer final : public tokenizers::Tokenizer {
 public:
  static constexpr const char* kBenchPromptMagic = "__qnn_llama_bench_prompt__";
  static constexpr uint64_t kEosSentinel =
      std::numeric_limits<uint64_t>::max() - 1;

  BenchTokenizer(int32_t prefill_len, uint64_t dummy_token_id)
      : prefill_len_(prefill_len), dummy_token_id_(dummy_token_id) {
    initialized_ = true;
    vocab_size_ = std::numeric_limits<int32_t>::max();
    bos_tok_ = dummy_token_id_;
    eos_tok_ = kEosSentinel;
  }

  tokenizers::Error load(const std::string&) override {
    initialized_ = true;
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::string> id_to_piece(uint64_t token) const override {
    return std::to_string(token);
  }

  tokenizers::Result<std::vector<uint64_t>> encode(
      const std::string& input,
      int8_t /*bos*/,
      int8_t /*eos*/) const override {
    // Runner probes for special EOS strings during load(). Return a sentinel
    // id there so decode loop cannot terminate early on EOS.
    if (input != kBenchPromptMagic) {
      return std::vector<uint64_t>{kEosSentinel};
    }
    return std::vector<uint64_t>(
        static_cast<size_t>(prefill_len_), dummy_token_id_);
  }

  tokenizers::Result<std::string> decode(uint64_t, uint64_t) const override {
    // Bench mode does not need decoded text output.
    return std::string();
  }

 private:
  int32_t prefill_len_;
  uint64_t dummy_token_id_;
};

struct Sample {
  double prefill_ms;
  double decode_ms;
  double total_ms;
  double prefill_tok_per_s;
  double decode_tok_per_s;
  double total_tok_per_s;
  double ttfb_ms;
};

struct Aggregate {
  int32_t count{0};
  double prefill_ms{0.0};
  double decode_ms{0.0};
  double total_ms{0.0};
  double prefill_tok_per_s{0.0};
  double decode_tok_per_s{0.0};
  double total_tok_per_s{0.0};
  double ttfb_ms{0.0};

  void add(const Sample& sample) {
    count++;
    prefill_ms += sample.prefill_ms;
    decode_ms += sample.decode_ms;
    total_ms += sample.total_ms;
    prefill_tok_per_s += sample.prefill_tok_per_s;
    decode_tok_per_s += sample.decode_tok_per_s;
    total_tok_per_s += sample.total_tok_per_s;
    ttfb_ms += sample.ttfb_ms;
  }

  Sample average() const {
    ET_CHECK_MSG(count > 0, "Attempted to average empty benchmark samples");
    return Sample{
        prefill_ms / count,
        decode_ms / count,
        total_ms / count,
        prefill_tok_per_s / count,
        decode_tok_per_s / count,
        total_tok_per_s / count,
        ttfb_ms / count};
  }
};

template <typename T>
Sample run_single(int32_t prefill_len, int32_t generation_len) {
  auto module = std::make_unique<executorch::extension::Module>(
      FLAGS_model_path.c_str(),
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;
  if (!FLAGS_attention_sink_rope_path.empty()) {
    attention_sink_rope_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_attention_sink_rope_path.c_str(),
            executorch::extension::Module::LoadMode::
                MmapUseMlockIgnoreErrors);
  }

  auto tokenizer =
      std::make_unique<BenchTokenizer>(prefill_len, FLAGS_dummy_token_id);
  // Keep the parameter ordering aligned with qnn_llama_runner.cpp.
  example::Runner<T> runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      "",
      FLAGS_performance_output_path.c_str(),
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      FLAGS_kv_store,
      FLAGS_test_level,
      FLAGS_blend_len,
      nullptr,
      std::move(tokenizer),
      std::move(attention_sink_rope_module));

  ET_CHECK_MSG(
      static_cast<int64_t>(prefill_len) + generation_len + 1 <=
          std::numeric_limits<int32_t>::max(),
      "prefill_len + generation_len is too large");
  executorch::extension::llm::GenerationConfig config{
      false,
      true,
      generation_len,
      false,
      prefill_len + generation_len + 1,
      static_cast<float>(FLAGS_temperature),
      0,
      0};

  struct CapturedStats {
    long inference_start_ms;
    long prompt_eval_end_ms;
    long inference_end_ms;
    long first_token_ms;
    int64_t num_prompt_tokens;
    int64_t num_generated_tokens;
  };
  CapturedStats stats{};
  bool got_stats = false;
  auto token_callback = [](const std::string&) {};
  auto stats_callback = [&](const executorch::llm::Stats& run_stats) {
    stats.inference_start_ms = run_stats.inference_start_ms;
    stats.prompt_eval_end_ms = run_stats.prompt_eval_end_ms;
    stats.inference_end_ms = run_stats.inference_end_ms;
    stats.first_token_ms = run_stats.first_token_ms;
    stats.num_prompt_tokens = run_stats.num_prompt_tokens;
    stats.num_generated_tokens = run_stats.num_generated_tokens;
    got_stats = true;
  };

  auto err = runner.generate_from_prompt_or_file(
      BenchTokenizer::kBenchPromptMagic,
      false,
      config,
      token_callback,
      stats_callback);
  ET_CHECK_MSG(
      err == executorch::runtime::Error::Ok,
      "Runner invocation failed with error code %d",
      static_cast<int>(err));
  ET_CHECK_MSG(got_stats, "Failed to collect benchmark stats");
  ET_CHECK_MSG(
      stats.num_prompt_tokens == prefill_len,
      "Prefill length mismatch (requested %d, got %ld)",
      prefill_len,
      stats.num_prompt_tokens);
  ET_CHECK_MSG(
      stats.num_generated_tokens == generation_len,
      "Decode step mismatch (requested %d, got %ld). Ensure seq_len fits model context or use attention sink.",
      generation_len,
      stats.num_generated_tokens);

  const double prefill_ms = stats.prompt_eval_end_ms - stats.inference_start_ms;
  const double decode_ms = stats.inference_end_ms - stats.prompt_eval_end_ms;
  const double total_ms = stats.inference_end_ms - stats.inference_start_ms;
  const double ttfb_ms = stats.first_token_ms - stats.inference_start_ms;

  return Sample{
      prefill_ms,
      decode_ms,
      total_ms,
      safe_rate(prefill_len, prefill_ms),
      safe_rate(generation_len, decode_ms),
      safe_rate(generation_len, total_ms),
      ttfb_ms};
}

template <typename T>
void run_benchmarks(
    const std::vector<int32_t>& prefill_lengths,
    const std::vector<int32_t>& generation_lengths) {
  std::ofstream out(FLAGS_output_path.c_str());
  ET_CHECK_MSG(
      out.is_open(),
      "Failed to open output csv path: %s",
      FLAGS_output_path.c_str());
  out << "prefill_len,generation_len,avg_prefill_ms,avg_decode_ms,avg_total_ms,"
         "avg_ttfb_ms,avg_prefill_tok_per_s,avg_decode_tok_per_s,"
         "avg_end_to_end_decode_tok_per_s\n";

  printf(
      "%10s %14s %14s %14s %14s %18s %18s %18s\n",
      "prefill",
      "generation",
      "prefill_ms",
      "decode_ms",
      "total_ms",
      "ttfb_ms",
      "prefill_tok/s",
      "decode_tok/s");
  for (int32_t prefill_len : prefill_lengths) {
    for (int32_t generation_len : generation_lengths) {
      for (int i = 0; i < FLAGS_warmup_iters; ++i) {
        (void)run_single<T>(prefill_len, generation_len);
      }

      Aggregate aggregate;
      for (int i = 0; i < FLAGS_num_iters; ++i) {
        aggregate.add(run_single<T>(prefill_len, generation_len));
      }
      const Sample avg = aggregate.average();

      out << prefill_len << "," << generation_len << "," << avg.prefill_ms
          << "," << avg.decode_ms << "," << avg.total_ms << "," << avg.ttfb_ms
          << "," << avg.prefill_tok_per_s << "," << avg.decode_tok_per_s << ","
          << avg.total_tok_per_s << "\n";
      out.flush();

      printf(
          "%10d %14d %14.2f %14.2f %14.2f %14.2f %18.2f %18.2f\n",
          prefill_len,
          generation_len,
          avg.prefill_ms,
          avg.decode_ms,
          avg.total_ms,
          avg.ttfb_ms,
          avg.prefill_tok_per_s,
          avg.decode_tok_per_s);
    }
  }
  out.close();
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ET_CHECK_MSG(FLAGS_num_iters > 0, "--num_iters must be > 0");
  ET_CHECK_MSG(FLAGS_warmup_iters >= 0, "--warmup_iters must be >= 0");
  ET_CHECK_MSG(FLAGS_dummy_token_id <= std::numeric_limits<int32_t>::max(),
      "--dummy_token_id must fit int32 for models with int32 token input");

  const std::vector<int32_t> prefill_lengths =
      parse_int_list(FLAGS_prefill_lengths, "prefill_lengths", false);
  const std::vector<int32_t> generation_lengths =
      parse_int_list(FLAGS_generation_lengths, "generation_lengths", true);

  auto module = std::make_unique<executorch::extension::Module>(
      FLAGS_model_path.c_str(),
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
  if (module->method_names()->count("get_kv_io_bit_width") > 0) {
    kv_bitwidth = static_cast<example::KvBitWidth>(
        module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
  }

  if (kv_bitwidth == example::KvBitWidth::kWidth8) {
    run_benchmarks<uint8_t>(prefill_lengths, generation_lengths);
  } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
    run_benchmarks<uint16_t>(prefill_lengths, generation_lengths);
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(kv_bitwidth));
  }

  return 0;
}
