/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

DEFINE_string(decoder_model_version, "llama2", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model is serialized using the flatbuffer format.");
DEFINE_string(
    tokenizer_path,
    "tokenizer.bin",
    "Tokenizer stuff.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Records inference speed.");
DEFINE_string(
    dump_logits_path,
    "",
    "Optional logits dump path.");
DEFINE_string(
    system_prompt,
    "",
    "Default system prompt for new requests.");
DEFINE_bool(
    no_think,
    false,
    "For Qwen3, append <think></think> to the end of the formatted prompt.");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; 0 = greedy argmax sampling.");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding / 3: BlenderMode");
DEFINE_bool(
    shared_buffer,
    false,
    "Use shared buffers.");
DEFINE_int32(num_iters, 1, "Unused, kept aligned with qnn_llama_runner.");
DEFINE_int32(ngram, 0, "Lookahead decoding parameter.");
DEFINE_int32(window, 0, "Lookahead decoding parameter.");
DEFINE_int32(gcap, 0, "Lookahead decoding parameter.");
DEFINE_bool(kv_store, false, "Keep aligned with qnn_llama_runner.");
DEFINE_int32(test_level, 0, "Debug test level.");
DEFINE_int32(blend_len, 32, "Blend length for BlenderMode.");
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
    "Enable separate embedding runtime path.");
DEFINE_string(
    embedding_matrix_path,
    "",
    "Path to separate embedding matrix file.");
DEFINE_string(
    rope_config_path,
    "",
    "Path to the RoPE config json used to precompute freqs_cos/freqs_sin.");
DEFINE_string(server_host, "0.0.0.0", "HTTP server bind host.");
DEFINE_int32(server_port, 8080, "HTTP server bind port.");

namespace {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string body;
};

struct GenerateRequest {
  std::string prompt;
  std::string system_prompt;
  std::string request_id;
  int32_t seq_len{FLAGS_seq_len};
  float temperature{static_cast<float>(FLAGS_temperature)};
  bool has_request_id{false};
};

struct PendingConnection {
  int fd{-1};
  HttpRequest request;
};

class RequestQueue {
 public:
  void push(PendingConnection&& connection) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(connection));
    cv_.notify_one();
  }

  PendingConnection pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() { return !queue_.empty(); });
    PendingConnection connection = std::move(queue_.front());
    queue_.pop_front();
    return connection;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<PendingConnection> queue_;
};

class ConnectionGuard {
 public:
  explicit ConnectionGuard(PendingConnection* connection)
      : connection_(connection) {}

  ~ConnectionGuard() {
    if (connection_ == nullptr) {
      return;
    }
    if (connection_->fd >= 0) {
      close(connection_->fd);
      connection_->fd = -1;
    }
  }

 private:
  PendingConnection* connection_;
};

std::string generate_request_id() {
  static std::atomic<uint64_t> counter{0};
  std::ostringstream oss;
  oss << "req-" << static_cast<unsigned long long>(time(nullptr)) << "-"
      << static_cast<unsigned long long>(counter.fetch_add(1));
  return oss.str();
}

std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::DecoderModelVersion decoder_model_version) {
  std::string formatted_prompt;
  switch (decoder_model_version) {
    case example::DecoderModelVersion::kLlama2:
    case example::DecoderModelVersion::kQwen2_5:
    case example::DecoderModelVersion::kCodegen:
      formatted_prompt.append(prompt);
      break;
    case example::DecoderModelVersion::kLlama3:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      break;
    case example::DecoderModelVersion::kGemma:
    case example::DecoderModelVersion::kGemma3:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<end_of_turn>\n");
      }
      break;
    case example::DecoderModelVersion::kGemma2:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      break;
    case example::DecoderModelVersion::kGranite:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|start_of_role|>system<|end_of_role|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end_of_text|>\n");
      }
      formatted_prompt.append("<|start_of_role|>user<|end_of_role|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end_of_text|>\n");
      formatted_prompt.append("<|start_of_role|>assistant<|end_of_role|>");
      break;
    case example::DecoderModelVersion::kPhi4:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end|>");
      }
      formatted_prompt.append("<|user|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end|><|assistant|>");
      break;
    case example::DecoderModelVersion::kQwen3:
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>assistant");
      if (FLAGS_no_think) {
        formatted_prompt.append("<think></think>");
      }
      break;
    case example::DecoderModelVersion::kQwen3Embed:
      formatted_prompt.append(prompt);
      break;
    case example::DecoderModelVersion::kSmollm2_135m:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n\n");
      break;
    case example::DecoderModelVersion::kSmollm3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("\n\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      break;
    case example::DecoderModelVersion::kGlm:
      formatted_prompt.append("<|user|>\n");
      formatted_prompt.append(prompt);
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>\n");
        formatted_prompt.append(system_prompt);
      }
      formatted_prompt.append("<|assistant|>\n");
      break;
    default:
      formatted_prompt.append(prompt);
      break;
  }
  return formatted_prompt;
}

bool write_all(int fd, const std::string& data) {
  size_t total_written = 0;
  while (total_written < data.size()) {
    ssize_t written = send(
        fd,
        data.data() + total_written,
        data.size() - total_written,
        0);
    if (written <= 0) {
      return false;
    }
    total_written += static_cast<size_t>(written);
  }
  return true;
}

std::string build_json_response(
    int status_code,
    const std::function<void(
        rapidjson::Writer<rapidjson::StringBuffer>&,
        rapidjson::Document::AllocatorType&)>& writer_fn) {
  rapidjson::Document doc;
  doc.SetObject();
  auto& allocator = doc.GetAllocator();
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  writer.StartObject();
  writer.Key("status_code");
  writer.Int(status_code);
  writer_fn(writer, allocator);
  writer.EndObject();
  return std::string(buffer.GetString(), buffer.GetSize());
}

bool send_http_json(int fd, int status_code, const std::string& body) {
  const char* reason = "OK";
  switch (status_code) {
    case 200:
      reason = "OK";
      break;
    case 400:
      reason = "Bad Request";
      break;
    case 404:
      reason = "Not Found";
      break;
    case 405:
      reason = "Method Not Allowed";
      break;
    default:
      reason = "Internal Server Error";
      break;
  }

  std::ostringstream response;
  response << "HTTP/1.1 " << status_code << " " << reason << "\r\n"
           << "Content-Type: application/json\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n\r\n"
           << body;
  return write_all(fd, response.str());
}

bool read_http_request(int fd, HttpRequest* request, std::string* error_message) {
  ET_CHECK_MSG(request != nullptr, "request must not be null");
  std::string raw;
  char buffer[4096];
  size_t header_end = std::string::npos;
  while ((header_end = raw.find("\r\n\r\n")) == std::string::npos) {
    ssize_t nread = recv(fd, buffer, sizeof(buffer), 0);
    if (nread <= 0) {
      *error_message = "Failed to read HTTP request.";
      return false;
    }
    raw.append(buffer, nread);
    if (raw.size() > 1024 * 1024) {
      *error_message = "Request headers are too large.";
      return false;
    }
  }

  const std::string header_blob = raw.substr(0, header_end);
  std::istringstream header_stream(header_blob);
  std::string request_line;
  if (!std::getline(header_stream, request_line)) {
    *error_message = "Missing HTTP request line.";
    return false;
  }
  if (!request_line.empty() && request_line.back() == '\r') {
    request_line.pop_back();
  }

  std::istringstream request_line_stream(request_line);
  std::string http_version;
  if (!(request_line_stream >> request->method >> request->path >> http_version)) {
    *error_message = "Malformed HTTP request line.";
    return false;
  }

  size_t content_length = 0;
  std::string header_line;
  while (std::getline(header_stream, header_line)) {
    if (!header_line.empty() && header_line.back() == '\r') {
      header_line.pop_back();
    }
    size_t colon_pos = header_line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    std::string key = header_line.substr(0, colon_pos);
    std::string value = header_line.substr(colon_pos + 1);
    while (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }
    for (char& c : key) {
      c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    }
    if (key == "content-length") {
      content_length = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
    }
  }

  request->body = raw.substr(header_end + 4);
  while (request->body.size() < content_length) {
    ssize_t nread = recv(fd, buffer, sizeof(buffer), 0);
    if (nread <= 0) {
      *error_message = "Failed to read HTTP body.";
      return false;
    }
    request->body.append(buffer, nread);
  }
  if (request->body.size() > content_length) {
    request->body.resize(content_length);
  }
  return true;
}

bool parse_generate_request(
    const std::string& body,
    GenerateRequest* request,
    std::string* error_message) {
  ET_CHECK_MSG(request != nullptr, "request must not be null");
  rapidjson::Document doc;
  doc.Parse(body.c_str());
  if (doc.HasParseError()) {
    *error_message = std::string("Invalid JSON: ") +
        rapidjson::GetParseError_En(doc.GetParseError());
    return false;
  }
  if (!doc.IsObject()) {
    *error_message = "JSON body must be an object.";
    return false;
  }
  if (!doc.HasMember("prompt") || !doc["prompt"].IsString()) {
    *error_message = "`prompt` must be a string.";
    return false;
  }
  request->prompt = doc["prompt"].GetString();
  if (request->prompt.empty()) {
    *error_message = "`prompt` must not be empty.";
    return false;
  }
  request->system_prompt = FLAGS_system_prompt;
  if (doc.HasMember("system_prompt")) {
    if (!doc["system_prompt"].IsString()) {
      *error_message = "`system_prompt` must be a string.";
      return false;
    }
    request->system_prompt = doc["system_prompt"].GetString();
  }
  if (doc.HasMember("request_id")) {
    if (!doc["request_id"].IsString()) {
      *error_message = "`request_id` must be a string.";
      return false;
    }
    request->has_request_id = true;
    request->request_id = doc["request_id"].GetString();
  }
  if (doc.HasMember("seq_len")) {
    if (!doc["seq_len"].IsInt()) {
      *error_message = "`seq_len` must be an int.";
      return false;
    }
    request->seq_len = doc["seq_len"].GetInt();
  }
  if (doc.HasMember("temperature")) {
    if (!doc["temperature"].IsNumber()) {
      *error_message = "`temperature` must be a number.";
      return false;
    }
    request->temperature = static_cast<float>(doc["temperature"].GetDouble());
  }
  return true;
}

int create_listen_socket() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    return -1;
  }
  int enable = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(FLAGS_server_port));
  if (inet_pton(AF_INET, FLAGS_server_host.c_str(), &addr.sin_addr) != 1) {
    close(listen_fd);
    return -1;
  }
  if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(listen_fd);
    return -1;
  }
  if (listen(listen_fd, 128) < 0) {
    close(listen_fd);
    return -1;
  }
  return listen_fd;
}

void send_error_and_close(int fd, int status_code, const std::string& message) {
  std::string body = build_json_response(
      status_code,
      [&](rapidjson::Writer<rapidjson::StringBuffer>& writer,
          rapidjson::Document::AllocatorType&) {
        writer.Key("error");
        writer.String(message.c_str());
      });
  send_http_json(fd, status_code, body);
  if (fd >= 0) {
    close(fd);
  }
}

void accept_loop(int listen_fd, RequestQueue* queue) {
  while (true) {
    int client_fd = accept(listen_fd, nullptr, nullptr);
    if (client_fd < 0) {
      ET_LOG(Error, "accept failed: errno=%d", errno);
      continue;
    }

    HttpRequest request;
    std::string error_message;
    if (!read_http_request(client_fd, &request, &error_message)) {
      send_error_and_close(client_fd, 400, error_message);
      continue;
    }
    if (request.method != "POST") {
      send_error_and_close(client_fd, 405, "Only POST is supported.");
      continue;
    }
    if (request.path != "/generate") {
      send_error_and_close(client_fd, 404, "Only /generate is supported.");
      continue;
    }

    PendingConnection connection;
    connection.fd = client_fd;
    connection.request = std::move(request);
    queue->push(std::move(connection));
  }
}

template <typename T>
void serve_requests(
    example::Runner<T>* runner,
    example::DecoderModelVersion decoder_model_version,
    RequestQueue* queue) {
  std::unordered_map<std::string, typename example::Runner<T>::KVSnapshot>
      kv_memory_pool;
  while (true) {
    PendingConnection connection = queue->pop();
    ConnectionGuard guard(&connection);

    GenerateRequest request;
    std::string error_message;
    if (!parse_generate_request(connection.request.body, &request, &error_message)) {
      std::string body = build_json_response(
          400,
          [&](rapidjson::Writer<rapidjson::StringBuffer>& writer,
              rapidjson::Document::AllocatorType&) {
            writer.Key("error");
            writer.String(error_message.c_str());
          });
      send_http_json(connection.fd, 400, body);
      continue;
    }

    bool restored = false;
    std::string request_id = request.has_request_id ? request.request_id : generate_request_id();
    auto it = kv_memory_pool.find(request_id);
    if (request.has_request_id && it != kv_memory_pool.end()) {
      auto error = runner->import_kv_snapshot(it->second);
      if (error != executorch::runtime::Error::Ok) {
        std::string body = build_json_response(
            500,
            [&](rapidjson::Writer<rapidjson::StringBuffer>& writer,
                rapidjson::Document::AllocatorType&) {
              writer.Key("error");
              writer.String("Failed to import KV snapshot.");
            });
        send_http_json(connection.fd, 500, body);
        continue;
      }
      restored = true;
    } else {
      runner->reset();
    }

    const std::string effective_system_prompt = restored ? "" : request.system_prompt;
    const std::string formatted_prompt = get_formatted_prompt(
        request.prompt, effective_system_prompt, decoder_model_version);
    std::string generated_text;
    executorch::extension::llm::GenerationConfig config{
        false,
        false,
        -1,
        false,
        request.seq_len,
        request.temperature,
        0,
        0};
    auto callback = [&](const std::string& piece) { generated_text += piece; };
    auto error = runner->generate_from_prompt_or_file(
        formatted_prompt, false, config, callback);
    if (error != executorch::runtime::Error::Ok) {
      std::string body = build_json_response(
          500,
          [&](rapidjson::Writer<rapidjson::StringBuffer>& writer,
              rapidjson::Document::AllocatorType&) {
            writer.Key("request_id");
            writer.String(request_id.c_str());
            writer.Key("error");
            writer.String("Inference failed.");
          });
      send_http_json(connection.fd, 500, body);
      continue;
    }

    typename example::Runner<T>::KVSnapshot snapshot;
    error = runner->export_kv_snapshot(&snapshot);
    if (error != executorch::runtime::Error::Ok) {
      std::string body = build_json_response(
          500,
          [&](rapidjson::Writer<rapidjson::StringBuffer>& writer,
              rapidjson::Document::AllocatorType&) {
            writer.Key("request_id");
            writer.String(request_id.c_str());
            writer.Key("error");
            writer.String("Failed to export KV snapshot.");
          });
      send_http_json(connection.fd, 500, body);
      continue;
    }
    kv_memory_pool[request_id] = std::move(snapshot);

    std::string body = build_json_response(
        200,
        [&](rapidjson::Writer<rapidjson::StringBuffer>& writer,
            rapidjson::Document::AllocatorType&) {
          writer.Key("request_id");
          writer.String(request_id.c_str());
          writer.Key("restored");
          writer.Bool(restored);
          writer.Key("text");
          writer.String(generated_text.c_str());
        });
    send_http_json(connection.fd, 200, body);
  }
}

template <typename T>
int start_server(
    std::unique_ptr<executorch::extension::Module> module,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module) {
  example::Runner<T> runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
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
  auto decoder_model_version_res = runner.get_decoder_model_version();
  if (!decoder_model_version_res.ok()) {
    ET_LOG(Error, "Failed to load runner and resolve decoder model version.");
    return 1;
  }

  int listen_fd = create_listen_socket();
  if (listen_fd < 0) {
    ET_LOG(
        Error,
        "Failed to bind HTTP server on %s:%d",
        FLAGS_server_host.c_str(),
        FLAGS_server_port);
    return 1;
  }

  ET_LOG(
      Info,
      "qnn_llama_server listening on http://%s:%d/generate",
      FLAGS_server_host.c_str(),
      FLAGS_server_port);
  RequestQueue queue;
  std::thread accept_thread(accept_loop, listen_fd, &queue);
  accept_thread.detach();

  serve_requests<T>(&runner, decoder_model_version_res.get(), &queue);
  close(listen_fd);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_separate_embed && FLAGS_embedding_matrix_path.empty()) {
    ET_LOG(
        Error,
        "--embedding_matrix_path must be provided when --separate_embed=true.");
    return 1;
  }

  std::unique_ptr<executorch::extension::Module> module =
      std::make_unique<executorch::extension::Module>(
          FLAGS_model_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;
  if (!FLAGS_attention_sink_rope_path.empty()) {
    attention_sink_rope_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_attention_sink_rope_path.c_str(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
  if (module->method_names()->count("get_kv_io_bit_width") > 0) {
    kv_bitwidth = static_cast<example::KvBitWidth>(
        module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
  }

  if (kv_bitwidth == example::KvBitWidth::kWidth8) {
    return start_server<uint8_t>(
        std::move(module), std::move(attention_sink_rope_module));
  }
  if (kv_bitwidth == example::KvBitWidth::kWidth16) {
    return start_server<uint16_t>(
        std::move(module), std::move(attention_sink_rope_module));
  }

  ET_LOG(
      Error,
      "Unsupported kv bitwidth: %ld",
      static_cast<int64_t>(kv_bitwidth));
  return 1;
}
