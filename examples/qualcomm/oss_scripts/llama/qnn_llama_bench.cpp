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
#include <random>
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
    rope_config_path,
    "",
    "Path to the RoPE config json used to precompute freqs_cos/freqs_sin.");
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
DEFINE_double(
    latency_ratio,
    0.2,
    "Latency ratio hyper-parameter passed to KVStore build_input().");
DEFINE_double(
    recompute_ratio,
    0.25,
    "Selective recompute reuse ratio passed to KVStore build_input().");
DEFINE_bool(
    enable_nonprefix_lcs,
    true,
    "Whether to enable non-prefix approximate LCS matching in KVStore build_input().");
DEFINE_bool(
    fp16,
    false,
    "Use fp16 rerotation path for non-prefix selective recompute.");
DEFINE_uint64(
    cpu_kv_pool_mb,
    512,
    "CPU KV pool size in MB used to cache row chunks before falling back to SQLite.");
DEFINE_bool(
    separate_embed,
    false,
    "Enable separate embedding runtime path (embedding is loaded from matrix file and fed as decoder input).");
DEFINE_string(
    embedding_matrix_path,
    "",
    "Path to separate embedding matrix file (e.g. separate_embed_matrix.bin). Required when --separate_embed=true.");
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
    "Fallback token id used only when --prompt_token_seed is set and a deterministic sequence is desired.");
DEFINE_uint64(
    prompt_token_seed,
    0,
    "Seed for benchmark prompt token generation. 0 uses nondeterministic seeding; non-zero makes prompt tokens reproducible.");
DEFINE_string(
    prefix_hit_ratios,
    "",
    "Comma-separated prefix hit ratios in percent for prefix-KV-reuse benchmarking (e.g. 25,50,75,100).");
DEFINE_string(
    nonprefix_hit_ratios,
    "",
    "Comma-separated non-prefix hit ratios in percent. Each warmup prompt is total_len+128 tokens: the first 128 tokens match, tokens [128,256) differ, and a continuous suffix-region hit is injected after the gap.");
DEFINE_string(
    output_path,
    "qnn_llama_bench.csv",
    "CSV output path for benchmark results.");
DEFINE_bool(
    enable_embed_feature,
    false,
    "Run a separate embedding-only path and write vector output to --embed_output_path.");
DEFINE_string(
    prompt,
    "Hello",
    "Prompt used by --enable_embed_feature.");
DEFINE_string(
    embed_output_path,
    "outputs.txt",
    "Output path for --enable_embed_feature.");
DEFINE_int32(
    embed_seq_len,
    1024,
    "seq_len used by --enable_embed_feature.");

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

std::vector<int32_t> parse_percent_list(
    const std::string& csv,
    const char* flag_name) {
  std::vector<int32_t> values = parse_int_list(csv, flag_name, true);
  for (int32_t value : values) {
    ET_CHECK_MSG(
        value >= 0 && value <= 100,
        "Value '%d' in --%s must be within [0, 100]",
        value,
        flag_name);
  }
  return values;
}

std::vector<int32_t> parse_optional_percent_list(
    const std::string& csv,
    const char* flag_name) {
  if (trim(csv).empty()) {
    return {};
  }
  return parse_percent_list(csv, flag_name);
}

int32_t infer_vocab_size(executorch::extension::Module* module) {
  auto method_meta = module->method_meta("kv_forward");
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for kv_forward: 0x%x",
      static_cast<unsigned int>(method_meta.error()));
  auto logits_meta = method_meta->output_tensor_meta(0);
  ET_CHECK_MSG(
      logits_meta.ok(),
      "Failed to get kv_forward output tensor meta: 0x%x",
      static_cast<unsigned int>(logits_meta.error()));
  const auto sizes = logits_meta->sizes();
  ET_CHECK_MSG(
      sizes.size() >= 3 && sizes[2] > 0 &&
          sizes[2] <= std::numeric_limits<int32_t>::max(),
      "Invalid vocab size inferred from kv_forward output tensor");
  return static_cast<int32_t>(sizes[2]);
}

class BenchTokenizer final : public tokenizers::Tokenizer {
 public:
  static constexpr const char* kBenchPromptMagic = "__qnn_llama_bench_prompt__";
  static constexpr uint64_t kEosSentinel =
      std::numeric_limits<uint64_t>::max() - 1;

  BenchTokenizer(
      uint64_t dummy_token_id,
      int32_t vocab_size,
      std::vector<uint64_t> prompt_tokens)
      : dummy_token_id_(dummy_token_id),
        vocab_size_limit_(vocab_size),
        prompt_tokens_(std::move(prompt_tokens)) {
    initialized_ = true;
    vocab_size_ = vocab_size_limit_;
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
    return prompt_tokens_;
  }

  tokenizers::Result<std::string> decode(uint64_t, uint64_t) const override {
    // Bench mode does not need decoded text output.
    return std::string();
  }

 private:
  uint64_t dummy_token_id_;
  int32_t vocab_size_limit_;
  std::vector<uint64_t> prompt_tokens_;
};

uint64_t next_prompt_seed(uint64_t base_seed) {
  static uint64_t call_index = 0;
  ++call_index;
  if (base_seed == 0) {
    return (static_cast<uint64_t>(std::random_device{}()) << 32) ^
        static_cast<uint64_t>(std::random_device{}()) ^ call_index;
  }
  return base_seed + call_index;
}

std::vector<uint64_t> build_random_prompt_tokens(
    int32_t prompt_len,
    int32_t vocab_size,
    uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> dist(
      0, static_cast<uint64_t>(vocab_size - 1));
  std::vector<uint64_t> tokens;
  tokens.reserve(static_cast<size_t>(prompt_len));
  for (int32_t i = 0; i < prompt_len; ++i) {
    tokens.push_back(dist(rng));
  }
  return tokens;
}

uint64_t sample_token_excluding(
    std::mt19937_64* rng,
    int32_t vocab_size,
    uint64_t forbidden) {
  std::uniform_int_distribution<uint64_t> dist(
      0, static_cast<uint64_t>(vocab_size - 1));
  uint64_t token = dist(*rng);
  while (token == forbidden) {
    token = dist(*rng);
  }
  return token;
}

struct Sample {
  double prefill_ms;
  double decode_ms;
  double total_ms;
  double prefill_tok_per_s;
  double decode_tok_per_s;
  double ttft_ms;
};

struct Aggregate {
  int32_t count{0};
  double prefill_ms{0.0};
  double decode_ms{0.0};
  double total_ms{0.0};
  double prefill_tok_per_s{0.0};
  double decode_tok_per_s{0.0};
  double ttft_ms{0.0};

  void add(const Sample& sample) {
    count++;
    prefill_ms += sample.prefill_ms;
    decode_ms += sample.decode_ms;
    total_ms += sample.total_ms;
    prefill_tok_per_s += sample.prefill_tok_per_s;
    decode_tok_per_s += sample.decode_tok_per_s;
    ttft_ms += sample.ttft_ms;
  }

  Sample average() const {
    ET_CHECK_MSG(count > 0, "Attempted to average empty benchmark samples");
    return Sample{
        prefill_ms / count,
        decode_ms / count,
        total_ms / count,
        prefill_tok_per_s / count,
        decode_tok_per_s / count,
        ttft_ms / count};
  }
};

template <typename T>
Sample run_single(
    const std::vector<uint64_t>& prompt_tokens,
    int32_t generation_len,
    int32_t vocab_size,
    std::shared_ptr<LMStore::CPUKVPool<T>>* kv_pool = nullptr) {
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
      std::make_unique<BenchTokenizer>(
          FLAGS_dummy_token_id,
          vocab_size,
          prompt_tokens);
  // Keep the parameter ordering aligned with qnn_llama_runner.cpp.
  example::Runner<T> runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      "",
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      FLAGS_kv_store,
      FLAGS_test_level,
      FLAGS_blend_len,
      static_cast<float>(FLAGS_latency_ratio),
      static_cast<float>(FLAGS_recompute_ratio),
      FLAGS_enable_nonprefix_lcs,
      FLAGS_fp16,
      FLAGS_cpu_kv_pool_mb,
      FLAGS_separate_embed,
      FLAGS_embedding_matrix_path.c_str(),
      FLAGS_rope_config_path.c_str(),
      nullptr,
      std::move(tokenizer),
      kv_pool ? *kv_pool : nullptr,
      std::move(attention_sink_rope_module));

  ET_CHECK_MSG(
      static_cast<int64_t>(prompt_tokens.size()) + generation_len + 1 <=
          std::numeric_limits<int32_t>::max(),
      "prefill_len + generation_len is too large");
  executorch::extension::llm::GenerationConfig config{
      false,
      true,
      generation_len,
      false,
      static_cast<int32_t>(prompt_tokens.size()) + generation_len + 1,
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
      stats.num_prompt_tokens == static_cast<int64_t>(prompt_tokens.size()),
      "Prefill length mismatch (requested %d, got %ld)",
      static_cast<int>(prompt_tokens.size()),
      stats.num_prompt_tokens);
  ET_CHECK_MSG(
      stats.num_generated_tokens == generation_len,
      "Decode step mismatch (requested %d, got %ld). Ensure seq_len fits model context or use attention sink.",
      generation_len,
      stats.num_generated_tokens);
  if (kv_pool) {
    *kv_pool = runner.get_kv_pool();
  }

  const double prefill_ms = stats.prompt_eval_end_ms - stats.inference_start_ms;
  const double decode_ms = stats.inference_end_ms - stats.first_token_ms;
  const double total_ms = stats.inference_end_ms - stats.inference_start_ms;
  const double ttft_ms = stats.first_token_ms - stats.inference_start_ms;
  const int64_t generation_only_tokens =
      std::max<int64_t>(0, stats.num_generated_tokens - 1);

  return Sample{
      prefill_ms,
      decode_ms,
      total_ms,
      safe_rate(prompt_tokens.size(), prefill_ms),
      safe_rate(generation_only_tokens, decode_ms),
      ttft_ms};
}

template <typename T>
Sample run_single_with_prefix_hit(
    int32_t prefill_len,
    int32_t generation_len,
    int32_t prefix_hit_ratio,
    int32_t vocab_size) {
  const uint64_t seed = next_prompt_seed(FLAGS_prompt_token_seed);
  const std::vector<uint64_t> full_prompt_tokens =
      build_random_prompt_tokens(prefill_len, vocab_size, seed);
  const int32_t prefix_len = static_cast<int32_t>(
      (static_cast<int64_t>(prefill_len) * prefix_hit_ratio) / 100);
  if (prefix_len <= 0) {
    return run_single<T>(full_prompt_tokens, generation_len, vocab_size);
  }
  std::vector<uint64_t> prefix_prompt_tokens(
      full_prompt_tokens.begin(), full_prompt_tokens.begin() + prefix_len);
  std::shared_ptr<LMStore::CPUKVPool<T>> kv_pool;
  (void)run_single<T>(prefix_prompt_tokens, 0, vocab_size, &kv_pool);
  return run_single<T>(full_prompt_tokens, generation_len, vocab_size, &kv_pool);
}

template <typename T>
Sample run_single_with_nonprefix_hit(
    int32_t prefill_len,
    int32_t generation_len,
    int32_t nonprefix_hit_ratio,
    int32_t vocab_size) {
  ET_CHECK_MSG(
      prefill_len > 128,
      "non-prefix reuse benchmark requires prefill_len > 128, got %d",
      prefill_len);
  const uint64_t seed = next_prompt_seed(FLAGS_prompt_token_seed);
  const std::vector<uint64_t> full_prompt_tokens =
      build_random_prompt_tokens(prefill_len, vocab_size, seed);
  const int32_t tail_len = prefill_len - 128;
  const int32_t hit_len = static_cast<int32_t>(
      (static_cast<int64_t>(tail_len) * nonprefix_hit_ratio) / 100);
  if (hit_len <= 0) {
    return run_single<T>(full_prompt_tokens, generation_len, vocab_size);
  }
  std::vector<uint64_t> warm_prompt_tokens;
  warm_prompt_tokens.reserve(static_cast<size_t>(prefill_len + 128));
  warm_prompt_tokens.insert(
      warm_prompt_tokens.end(), full_prompt_tokens.begin(), full_prompt_tokens.begin() + 128);

  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
  for (int32_t i = 0; i < 128; ++i) {
    const uint64_t forbidden =
        full_prompt_tokens[static_cast<size_t>(i) % full_prompt_tokens.size()];
    warm_prompt_tokens.push_back(
        sample_token_excluding(&rng, vocab_size, forbidden));
  }
  warm_prompt_tokens.insert(
      warm_prompt_tokens.end(),
      full_prompt_tokens.begin() + 128,
      full_prompt_tokens.begin() + 128 + hit_len);
  for (int32_t i = hit_len; i < tail_len; ++i) {
    warm_prompt_tokens.push_back(
        sample_token_excluding(&rng, vocab_size, full_prompt_tokens[128 + i]));
  }

  std::shared_ptr<LMStore::CPUKVPool<T>> kv_pool;
  (void)run_single<T>(warm_prompt_tokens, 0, vocab_size, &kv_pool);
  return run_single<T>(full_prompt_tokens, generation_len, vocab_size, &kv_pool);
}

template <typename T>
void run_benchmarks(
    const std::vector<int32_t>& prefill_lengths,
    const std::vector<int32_t>& generation_lengths,
    const std::vector<int32_t>& prefix_hit_ratios,
    const std::vector<int32_t>& nonprefix_hit_ratios,
    int32_t vocab_size) {
  std::ofstream out(FLAGS_output_path.c_str());
  ET_CHECK_MSG(
      out.is_open(),
      "Failed to open output csv path: %s",
      FLAGS_output_path.c_str());
  out << "reuse_type,prefill_len,prefix_len,hit_len,hit_ratio,generation_len,avg_prefill_ms,avg_decode_ms,avg_total_ms,"
         "avg_ttft_ms,avg_prefill_tok_per_s,avg_decode_tok_per_s\n";

  printf(
      "%12s %10s %12s %10s %10s %14s %14s %14s %14s %18s %18s %18s\n",
      "reuse_type",
      "prefill",
      "prefix_len",
      "hit_len",
      "hit_ratio",
      "generation",
      "prefill_ms",
      "decode_ms",
      "total_ms",
      "ttft_ms",
      "prefill_tok/s",
      "decode_tok/s");
  const bool run_naive =
      prefix_hit_ratios.empty() && nonprefix_hit_ratios.empty();
  for (int32_t prefill_len : prefill_lengths) {
    if (run_naive) {
      for (int32_t generation_len : generation_lengths) {
        for (int i = 0; i < FLAGS_warmup_iters; ++i) {
          const uint64_t seed = next_prompt_seed(FLAGS_prompt_token_seed);
          const std::vector<uint64_t> prompt_tokens =
              build_random_prompt_tokens(prefill_len, vocab_size, seed);
          (void)run_single<T>(prompt_tokens, generation_len, vocab_size);
        }

        Aggregate aggregate;
        for (int i = 0; i < FLAGS_num_iters; ++i) {
          const uint64_t seed = next_prompt_seed(FLAGS_prompt_token_seed);
          const std::vector<uint64_t> prompt_tokens =
              build_random_prompt_tokens(prefill_len, vocab_size, seed);
          aggregate.add(run_single<T>(prompt_tokens, generation_len, vocab_size));
        }
        const Sample avg = aggregate.average();

        out << "naive," << prefill_len << "," << 0 << "," << 0 << "," << 0
            << "," << generation_len << "," << avg.prefill_ms << ","
            << avg.decode_ms << "," << avg.total_ms << "," << avg.ttft_ms
            << "," << avg.prefill_tok_per_s << "," << avg.decode_tok_per_s
            << "\n";
        out.flush();

        printf(
            "%12s %10d %12d %10d %10d %14d %14.2f %14.2f %14.2f %18.2f %18.2f %18.2f\n",
            "naive",
            prefill_len,
            0,
            0,
            0,
            generation_len,
            avg.prefill_ms,
            avg.decode_ms,
            avg.total_ms,
            avg.ttft_ms,
            avg.prefill_tok_per_s,
            avg.decode_tok_per_s);
      }
      continue;
    }
    for (int32_t prefix_hit_ratio : prefix_hit_ratios) {
      const int32_t prefix_len = static_cast<int32_t>(
          (static_cast<int64_t>(prefill_len) * prefix_hit_ratio) / 100);
      for (int32_t generation_len : generation_lengths) {
        for (int i = 0; i < FLAGS_warmup_iters; ++i) {
          (void)run_single_with_prefix_hit<T>(
              prefill_len, generation_len, prefix_hit_ratio, vocab_size);
        }

        Aggregate aggregate;
        for (int i = 0; i < FLAGS_num_iters; ++i) {
          aggregate.add(run_single_with_prefix_hit<T>(
              prefill_len, generation_len, prefix_hit_ratio, vocab_size));
        }
        const Sample avg = aggregate.average();

        out << "prefix," << prefill_len << "," << prefix_len << ","
            << prefix_len << "," << prefix_hit_ratio
            << "," << generation_len << "," << avg.prefill_ms << ","
            << avg.decode_ms << "," << avg.total_ms << "," << avg.ttft_ms
            << "," << avg.prefill_tok_per_s << "," << avg.decode_tok_per_s
            << "\n";
        out.flush();

        printf(
            "%12s %10d %12d %10d %10d %14d %14.2f %14.2f %14.2f %18.2f %18.2f %18.2f\n",
            "prefix",
            prefill_len,
            prefix_len,
            prefix_len,
            prefix_hit_ratio,
            generation_len,
            avg.prefill_ms,
            avg.decode_ms,
            avg.total_ms,
            avg.ttft_ms,
            avg.prefill_tok_per_s,
            avg.decode_tok_per_s);
      }
    }
    for (int32_t nonprefix_hit_ratio : nonprefix_hit_ratios) {
      ET_CHECK_MSG(
          prefill_len > 128,
          "--nonprefix_hit_ratios requires prefill_len > 128, got %d",
          prefill_len);
      const int32_t prefix_len = 128;
      const int32_t hit_len = static_cast<int32_t>(
          (static_cast<int64_t>(prefill_len - 128) * nonprefix_hit_ratio) /
          100);
      for (int32_t generation_len : generation_lengths) {
        for (int i = 0; i < FLAGS_warmup_iters; ++i) {
          (void)run_single_with_nonprefix_hit<T>(
              prefill_len, generation_len, nonprefix_hit_ratio, vocab_size);
        }

        Aggregate aggregate;
        for (int i = 0; i < FLAGS_num_iters; ++i) {
          aggregate.add(run_single_with_nonprefix_hit<T>(
              prefill_len, generation_len, nonprefix_hit_ratio, vocab_size));
        }
        const Sample avg = aggregate.average();

        out << "nonprefix," << prefill_len << "," << prefix_len << ","
            << hit_len << "," << nonprefix_hit_ratio << "," << generation_len
            << "," << avg.prefill_ms << "," << avg.decode_ms << ","
            << avg.total_ms << "," << avg.ttft_ms << ","
            << avg.prefill_tok_per_s << "," << avg.decode_tok_per_s << "\n";
        out.flush();

        printf(
            "%12s %10d %12d %10d %10d %14d %14.2f %14.2f %14.2f %18.2f %18.2f %18.2f\n",
            "nonprefix",
            prefill_len,
            prefix_len,
            hit_len,
            nonprefix_hit_ratio,
            generation_len,
            avg.prefill_ms,
            avg.decode_ms,
            avg.total_ms,
            avg.ttft_ms,
            avg.prefill_tok_per_s,
            avg.decode_tok_per_s);
      }
    }
  }
  out.close();
}

template <typename T>
void run_embed_feature() {
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

  // Keep parameter ordering aligned with qnn_llama_runner.cpp.
  example::Runner<T> runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      "",
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      FLAGS_kv_store,
      FLAGS_test_level,
      FLAGS_blend_len,
      static_cast<float>(FLAGS_latency_ratio),
      static_cast<float>(FLAGS_recompute_ratio),
      FLAGS_enable_nonprefix_lcs,
      FLAGS_fp16,
      FLAGS_cpu_kv_pool_mb,
      FLAGS_separate_embed,
      FLAGS_embedding_matrix_path.c_str(),
      FLAGS_rope_config_path.c_str(),
      nullptr,
      nullptr,
      nullptr,
      std::move(attention_sink_rope_module));

  auto decoder_model_version = runner.get_decoder_model_version();
  ET_CHECK_MSG(
      decoder_model_version.ok(),
      "Failed to get decoder model version");
  ET_CHECK_MSG(
      decoder_model_version.get() == example::DecoderModelVersion::kQwen3Embed,
      "--enable_embed_feature currently only supports qwen3-embed model.");

  std::vector<char> buf;
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };

  executorch::extension::llm::GenerationConfig config{
      true,
      false,
      -1,
      false,
      FLAGS_embed_seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};

  auto err = runner.generate_from_prompt_or_file(
      FLAGS_prompt.c_str(), false, config, callback);
  ET_CHECK_MSG(
      err == executorch::runtime::Error::Ok,
      "Embedding run failed with error code %d",
      static_cast<int>(err));

  std::ofstream fout(FLAGS_embed_output_path.c_str());
  ET_CHECK_MSG(
      fout.is_open(),
      "Failed to open embedding output path: %s",
      FLAGS_embed_output_path.c_str());
  fout.write(buf.data(), buf.size());
  fout.close();
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ET_CHECK_MSG(FLAGS_num_iters > 0, "--num_iters must be > 0");
  ET_CHECK_MSG(FLAGS_warmup_iters >= 0, "--warmup_iters must be >= 0");
  if (FLAGS_separate_embed && FLAGS_embedding_matrix_path.empty()) {
    ET_CHECK_MSG(
        false,
        "--embedding_matrix_path must be provided when --separate_embed=true.");
  }
  ET_CHECK_MSG(FLAGS_dummy_token_id <= std::numeric_limits<int32_t>::max(),
      "--dummy_token_id must fit int32 for models with int32 token input");

  const std::vector<int32_t> prefill_lengths =
      parse_int_list(FLAGS_prefill_lengths, "prefill_lengths", false);
  const std::vector<int32_t> generation_lengths =
      parse_int_list(FLAGS_generation_lengths, "generation_lengths", true);
  const std::vector<int32_t> prefix_hit_ratios =
      parse_optional_percent_list(FLAGS_prefix_hit_ratios, "prefix_hit_ratios");
  const std::vector<int32_t> nonprefix_hit_ratios =
      parse_optional_percent_list(
          FLAGS_nonprefix_hit_ratios, "nonprefix_hit_ratios");

  auto module = std::make_unique<executorch::extension::Module>(
      FLAGS_model_path.c_str(),
      executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  const int32_t vocab_size = infer_vocab_size(module.get());
  example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
  if (module->method_names()->count("get_kv_io_bit_width") > 0) {
    kv_bitwidth = static_cast<example::KvBitWidth>(
        module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
  }

  if (FLAGS_enable_embed_feature) {
    ET_CHECK_MSG(FLAGS_embed_seq_len > 0, "--embed_seq_len must be > 0");
    if (kv_bitwidth == example::KvBitWidth::kWidth8) {
      run_embed_feature<uint8_t>();
    } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
      run_embed_feature<uint16_t>();
    } else {
      ET_CHECK_MSG(
          false,
          "Unsupported kv bitwidth: %ld",
          static_cast<int64_t>(kv_bitwidth));
    }
    return 0;
  }

  if (kv_bitwidth == example::KvBitWidth::kWidth8) {
    run_benchmarks<uint8_t>(
        prefill_lengths,
        generation_lengths,
        prefix_hit_ratios,
        nonprefix_hit_ratios,
        vocab_size);
  } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
    run_benchmarks<uint16_t>(
        prefill_lengths,
        generation_lengths,
        prefix_hit_ratios,
        nonprefix_hit_ratios,
        vocab_size);
  } else {
    ET_CHECK_MSG(
        false,
        "Unsupported kv bitwidth: %ld",
        static_cast<int64_t>(kv_bitwidth));
  }

  return 0;
}
