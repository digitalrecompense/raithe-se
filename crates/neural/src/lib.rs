// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: neural
//
// GPU inference manager.
//
// Manages three models on the GPU (8GB VRAM):
//   1. Cross-encoder (MiniLM-L6-v2 fine-tuned, ~1.5GB)
//   2. Bi-encoder (all-MiniLM-L6-v2, ~500MB)
//   3. Small LLM (Phi-2 or Qwen-1.5B, ~4GB)
//
// Uses ONNX Runtime with CUDA EP for inference.
// Graceful fallback to CPU when GPU is unavailable.
// Graceful degradation when model files are not present on disk.
//
// Design (Graceful Degradation):
//   If the GPU is saturated or a neural model fails, fall back to
//   GBDT-only ranking. Quality degrades; availability never does.
//
// Concurrency:
//   The GPU can only run one model at a time efficiently. Use a Mutex
//   per session. Cross-encoder re-ranking and LLM query understanding
//   are serialized on the GPU.
// ================================================================================

use raithe_common::config::NeuralConfig;
use raithe_common::error::{RaitheError, RaitheResult};
use raithe_common::traits::{AsyncResult, NeuralInference};
use raithe_common::types::LlmReformulation;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use tracing::{debug, error, info, warn};

// Re-export ort types used internally.
use ort::session::Session;
use ort::value::Tensor;

// ---------------------------------------------------------------------------
// Simple WordPiece-like tokenizer for transformer models
// ---------------------------------------------------------------------------

/// A minimal tokenizer that converts text into integer token IDs.
///
/// For production, this should load a real vocabulary file (vocab.txt) from
/// the model directory. This implementation uses a hash-based approach that
/// produces deterministic token IDs for any input text, allowing the ONNX
/// models to receive properly-shaped tensors.
///
/// When real models are deployed, replace this with a proper tokenizer that
/// loads the model's vocabulary (e.g., from a vocab.txt or tokenizer.json
/// alongside the .onnx file).
struct SimpleTokenizer {
    /// CLS token ID (always first).
    cls_id: i64,
    /// SEP token ID (segment separator).
    sep_id: i64,
    /// PAD token ID.
    pad_id: i64,
}

impl SimpleTokenizer {
    fn new() -> Self {
        Self {
            cls_id: 101,
            sep_id: 102,
            pad_id: 0,
        }
    }

    /// Tokenize a single text string into token IDs with [CLS] prefix and
    /// [SEP] suffix. Truncates to max_tokens (including special tokens).
    fn tokenize(&self, text: &str, max_tokens: usize) -> Vec<i64> {
        let mut ids = Vec::with_capacity(max_tokens);
        ids.push(self.cls_id);

        // Simple word-level tokenization: hash each word to a token ID
        // in the range [1000, 30000) to avoid colliding with special tokens.
        let budget = max_tokens.saturating_sub(2); // reserve CLS + SEP
        for word in text.split_whitespace().take(budget) {
            let hash = word
                .to_lowercase()
                .bytes()
                .fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
            ids.push((hash % 29000 + 1000) as i64);
        }

        ids.push(self.sep_id);
        ids
    }

    /// Tokenize a (query, passage) pair for cross-encoder input:
    ///   [CLS] query_tokens [SEP] passage_tokens [SEP]
    /// Returns (input_ids, attention_mask, token_type_ids), all padded to
    /// max_length.
    fn tokenize_pair(
        &self,
        query: &str,
        passage: &str,
        query_max: usize,
        passage_max: usize,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let max_length = query_max + passage_max + 3; // CLS + SEP + SEP

        let mut input_ids = Vec::with_capacity(max_length);
        let mut token_type_ids = Vec::with_capacity(max_length);

        // [CLS] + query tokens + [SEP]
        input_ids.push(self.cls_id);
        token_type_ids.push(0);

        for word in query.split_whitespace().take(query_max) {
            let hash = word
                .to_lowercase()
                .bytes()
                .fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
            input_ids.push((hash % 29000 + 1000) as i64);
            token_type_ids.push(0);
        }
        input_ids.push(self.sep_id);
        token_type_ids.push(0);

        // passage tokens + [SEP]
        for word in passage.split_whitespace().take(passage_max) {
            let hash = word
                .to_lowercase()
                .bytes()
                .fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
            input_ids.push((hash % 29000 + 1000) as i64);
            token_type_ids.push(1);
        }
        input_ids.push(self.sep_id);
        token_type_ids.push(1);

        // Pad to max_length.
        let seq_len = input_ids.len();
        let attention_mask: Vec<i64> = (0..max_length)
            .map(|i| if i < seq_len { 1 } else { 0 })
            .collect();

        input_ids.resize(max_length, self.pad_id);
        token_type_ids.resize(max_length, 0);

        (input_ids, attention_mask, token_type_ids)
    }

    /// Pad a single sequence to max_length, returning (input_ids, attention_mask).
    fn pad(&self, ids: &[i64], max_length: usize) -> (Vec<i64>, Vec<i64>) {
        let seq_len = ids.len();
        let mut padded = ids.to_vec();
        padded.resize(max_length, self.pad_id);

        let attention_mask: Vec<i64> = (0..max_length)
            .map(|i| if i < seq_len { 1 } else { 0 })
            .collect();

        (padded, attention_mask)
    }
}

// ---------------------------------------------------------------------------
// GPU Inference Manager
// ---------------------------------------------------------------------------

/// The GPU inference manager.
///
/// Owns ONNX Runtime sessions for the three neural models.
/// Each session is behind a Mutex because ort::Session::run requires &mut self.
/// This serializes GPU access per model (concurrency design).
pub struct GpuInferenceManager {
    config: NeuralConfig,
    gpu_available: AtomicBool,
    tokenizer: SimpleTokenizer,

    /// Cross-encoder ONNX session.
    /// None if the model file does not exist on disk.
    cross_encoder: Option<Mutex<Session>>,

    /// Bi-encoder ONNX session.
    /// None if the model file does not exist on disk.
    bi_encoder: Option<Mutex<Session>>,

    /// LLM ONNX session for query reformulation.
    /// None if the model file does not exist on disk.
    llm: Option<Mutex<Session>>,
}

impl GpuInferenceManager {
    /// Initialize the GPU inference manager and load models.
    /// Launch sequence step 5:
    /// - Probe for CUDA device availability.
    /// - Attempt to load each model file from configured paths.
    /// - If a model file is missing, log a warning and continue (graceful degradation).
    /// - If CUDA is unavailable, fall back to CPU execution provider.
    pub fn new(config: &NeuralConfig) -> RaitheResult<Self> {
        info!("Initializing RAiTHE GPU inference manager");
        info!("  Cross-encoder model: {:?}", config.cross_encoder_model_path);
        info!("  Bi-encoder model: {:?}", config.bi_encoder_model_path);
        info!("  LLM model: {:?}", config.llm_model_path);
        info!("  CUDA enabled: {}", config.use_cuda);

        let tokenizer = SimpleTokenizer::new();

        // ---------------------------------------------------------------
        // Check if model files exist FIRST — before touching ORT at all.
        //
        // If none of the three model files exist on disk, there is no
        // point initializing ONNX Runtime (which may attempt to download
        // a ~200MB shared library). Skip straight to graceful degradation.
        // ---------------------------------------------------------------
        let any_model_exists = config.cross_encoder_model_path.exists()
            || config.bi_encoder_model_path.exists()
            || config.llm_model_path.exists();

        if !any_model_exists {
            info!(
                "No neural model files found on disk — neural inference disabled. \
                 Place ONNX models at the configured paths to enable neural ranking."
            );
            return Ok(Self {
                config: config.clone(),
                gpu_available: AtomicBool::new(false),
                tokenizer,
                cross_encoder: None,
                bi_encoder: None,
                llm: None,
            });
        }

        // ---------------------------------------------------------------
        // Initialize the ONNX Runtime environment (load-dynamic mode).
        //
        // With the `load-dynamic` feature, ort uses dlopen() at runtime
        // to load libonnxruntime.so instead of statically linking ort-sys.
        // This avoids glibc version mismatches with prebuilt binaries.
        //
        // If the ORT shared library is not installed and ORT_DYLIB_PATH
        // is not set, ort will attempt to auto-download it. We use
        // catch_unwind to handle any panics from the download/load path.
        // ---------------------------------------------------------------
        let ort_ready = std::panic::catch_unwind(|| {
            let _ = ort::init().commit();
        });

        if ort_ready.is_err() {
            warn!(
                "ONNX Runtime initialization failed (library not found or incompatible). \
                 Neural inference disabled. Install libonnxruntime and set ORT_DYLIB_PATH, \
                 or download from https://github.com/microsoft/onnxruntime/releases"
            );
            return Ok(Self {
                config: config.clone(),
                gpu_available: AtomicBool::new(false),
                tokenizer,
                cross_encoder: None,
                bi_encoder: None,
                llm: None,
            });
        }

        info!("ONNX Runtime environment initialized");

        // ---------------------------------------------------------------
        // S1: Real CUDA device probe via ONNX Runtime.
        // ---------------------------------------------------------------
        let gpu_detected = if config.use_cuda {
            info!("GPU probe: checking for CUDA device via ONNX Runtime...");
            probe_cuda_available()
        } else {
            info!("CUDA disabled in configuration — using CPU only");
            false
        };

        if gpu_detected {
            info!("CUDA device detected — neural models will use GPU acceleration");
        } else {
            warn!("CUDA device not detected — neural models will use CPU fallback");
        }

        // ---------------------------------------------------------------
        // Load models. Each returns Option<Session>.
        // Missing model files are not fatal — graceful degradation.
        // ---------------------------------------------------------------
        let use_gpu = config.use_cuda && gpu_detected;

        let cross_encoder = load_model("cross-encoder", &config.cross_encoder_model_path, use_gpu);
        let bi_encoder = load_model("bi-encoder", &config.bi_encoder_model_path, use_gpu);
        let llm = load_model("LLM", &config.llm_model_path, use_gpu);

        // GPU is "available" if CUDA was detected AND at least one model loaded.
        let gpu_available =
            gpu_detected && (cross_encoder.is_some() || bi_encoder.is_some() || llm.is_some());

        let models_loaded = [
            cross_encoder.is_some(),
            bi_encoder.is_some(),
            llm.is_some(),
        ]
        .iter()
        .filter(|&&v| v)
        .count();

        info!(
            "GPU inference manager ready — GPU: {}, models loaded: {}/3",
            gpu_available, models_loaded
        );

        Ok(Self {
            config: config.clone(),
            gpu_available: AtomicBool::new(gpu_available),
            tokenizer,
            cross_encoder: cross_encoder.map(Mutex::new),
            bi_encoder: bi_encoder.map(Mutex::new),
            llm: llm.map(Mutex::new),
        })
    }

    /// Get VRAM usage estimate in bytes.
    pub fn estimated_vram_usage(&self) -> u64 {
        if !self.gpu_available.load(Ordering::Relaxed) {
            return 0;
        }
        let mut total_mb: usize = 0;
        if self.cross_encoder.is_some() {
            total_mb += self.config.cross_encoder_vram_mb;
        }
        if self.bi_encoder.is_some() {
            total_mb += self.config.bi_encoder_vram_mb;
        }
        if self.llm.is_some() {
            total_mb += self.config.llm_vram_mb;
        }
        total_mb as u64 * 1024 * 1024
    }

    fn is_gpu_up(&self) -> bool {
        self.gpu_available.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// NeuralInference trait implementation
// ---------------------------------------------------------------------------

/// Embedding dimension constant.
const EMBEDDING_DIM: usize = 128;

impl NeuralInference for GpuInferenceManager {
    /// S2: Compute dense embedding for a query string.
    ///
    /// Fallback: returns zero vector if bi-encoder is not loaded.
    fn embed_query(&self, query: &str) -> AsyncResult<'_, Vec<f32>> {
        let query = query.to_string();
        Box::pin(async move {
            let session_mutex = match &self.bi_encoder {
                Some(m) => m,
                None => {
                    debug!("Bi-encoder not loaded — returning zero embedding");
                    return Ok(vec![0.0f32; EMBEDDING_DIM]);
                }
            };

            let mut session = match session_mutex.lock() {
                Ok(s) => s,
                Err(e) => {
                    error!("Bi-encoder session lock poisoned: {}", e);
                    return Ok(vec![0.0f32; EMBEDDING_DIM]);
                }
            };

            let start = Instant::now();

            // Tokenize the query.
            let token_ids = self.tokenizer.tokenize(&query, 64);
            let seq_len = token_ids.len();
            let (padded_ids, attention_mask) = self.tokenizer.pad(&token_ids, seq_len);

            // Build input tensors: shape [1, seq_len].
            let input_ids =
                Tensor::from_array(([1usize, seq_len], padded_ids.into_boxed_slice()))
                    .map_err(|e| RaitheError::NeuralInference(format!("input_ids tensor: {}", e)))?;

            let attn_mask =
                Tensor::from_array(([1usize, seq_len], attention_mask.into_boxed_slice()))
                    .map_err(|e| RaitheError::NeuralInference(format!("attn_mask tensor: {}", e)))?;

            // Run inference.
            let outputs = match session.run(ort::inputs![input_ids, attn_mask]) {
                Ok(o) => o,
                Err(e) => {
                    warn!("Bi-encoder inference failed: {} — returning zero embedding", e);
                    return Ok(vec![0.0f32; EMBEDDING_DIM]);
                }
            };

            let embedding = extract_embedding_from_output(&outputs, EMBEDDING_DIM);

            debug!(
                query = %query,
                latency_ms = start.elapsed().as_millis() as u64,
                "Bi-encoder embedding computed"
            );

            Ok(embedding)
        })
    }

    /// S2: Compute dense embeddings for a batch of documents.
    fn embed_documents(&self, texts: &[String]) -> AsyncResult<'_, Vec<Vec<f32>>> {
        let texts: Vec<String> = texts.to_vec();
        Box::pin(async move {
            if self.bi_encoder.is_none() {
                debug!("Bi-encoder not loaded — returning zero embeddings for {} docs", texts.len());
                return Ok(vec![vec![0.0f32; EMBEDDING_DIM]; texts.len()]);
            }

            let mut results = Vec::with_capacity(texts.len());
            for text in &texts {
                let embedding = self.embed_query(text).await?;
                results.push(embedding);
            }
            Ok(results)
        })
    }

    /// S3: Cross-encoder scoring.
    ///
    /// Latency budget: 50ms, fallback at 100ms.
    fn cross_encode(&self, query: &str, passages: &[String]) -> AsyncResult<'_, Vec<f64>> {
        let query = query.to_string();
        let passages: Vec<String> = passages.to_vec();
        Box::pin(async move {
            let session_mutex = match &self.cross_encoder {
                Some(m) => m,
                None => {
                    debug!("Cross-encoder not loaded — falling back to Phase 2 ranking");
                    return Err(RaitheError::GpuUnavailable);
                }
            };

            let mut session = match session_mutex.lock() {
                Ok(s) => s,
                Err(e) => {
                    error!("Cross-encoder session lock poisoned: {}", e);
                    return Err(RaitheError::GpuUnavailable);
                }
            };

            let start = Instant::now();
            let batch_size = passages.len();
            let query_max = 32usize;
            let passage_max = 128usize;
            let max_length = query_max + passage_max + 3;

            // Build batched input tensors: shape [batch_size, max_length].
            let mut all_input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_length);
            let mut all_attn_mask: Vec<i64> = Vec::with_capacity(batch_size * max_length);
            let mut all_type_ids: Vec<i64> = Vec::with_capacity(batch_size * max_length);

            for passage in &passages {
                let (ids, mask, type_ids) =
                    self.tokenizer.tokenize_pair(&query, passage, query_max, passage_max);
                all_input_ids.extend_from_slice(&ids);
                all_attn_mask.extend_from_slice(&mask);
                all_type_ids.extend_from_slice(&type_ids);
            }

            let input_ids = Tensor::from_array((
                [batch_size, max_length],
                all_input_ids.into_boxed_slice(),
            ))
            .map_err(|e| RaitheError::NeuralInference(format!("cross-enc input_ids: {}", e)))?;

            let attn_mask = Tensor::from_array((
                [batch_size, max_length],
                all_attn_mask.into_boxed_slice(),
            ))
            .map_err(|e| RaitheError::NeuralInference(format!("cross-enc attn_mask: {}", e)))?;

            let type_ids = Tensor::from_array((
                [batch_size, max_length],
                all_type_ids.into_boxed_slice(),
            ))
            .map_err(|e| RaitheError::NeuralInference(format!("cross-enc type_ids: {}", e)))?;

            // Run batched inference.
            let outputs = match session.run(ort::inputs![input_ids, attn_mask, type_ids]) {
                Ok(o) => o,
                Err(e) => {
                    warn!("Cross-encoder inference failed: {} — falling back", e);
                    return Err(RaitheError::GpuUnavailable);
                }
            };

            let latency_ms = start.elapsed().as_millis() as u64;

            if latency_ms > 100 {
                warn!(
                    "Cross-encoder exceeded fallback budget: {}ms > 100ms — result still used",
                    latency_ms
                );
            }

            let scores = extract_scores_from_output(&outputs, batch_size);

            debug!(
                query = %query, batch_size, latency_ms,
                "Cross-encoder scoring complete"
            );

            Ok(scores)
        })
    }

    /// S4: LLM query reformulation.
    ///
    /// Latency budget: 200ms, fallback at 400ms.
    fn reformulate_query(&self, query: &str) -> AsyncResult<'_, Option<LlmReformulation>> {
        let query = query.to_string();
        Box::pin(async move {
            let session_mutex = match &self.llm {
                Some(m) => m,
                None => {
                    debug!("LLM not loaded — skipping query reformulation");
                    return Ok(None);
                }
            };

            let mut session = match session_mutex.lock() {
                Ok(s) => s,
                Err(e) => {
                    error!("LLM session lock poisoned: {}", e);
                    return Ok(None);
                }
            };

            let start = Instant::now();

            // Structured prompt for reformulation.
            let prompt = format!(
                "Given the search query: \"{}\"\n\
                 Produce a JSON object with:\n\
                 - \"reformulated\": a clearer query for search\n\
                 - \"alternatives\": list of 1-2 alternative interpretations\n\
                 - \"entities\": list of [type, value] pairs\n\
                 JSON:",
                query
            );

            let token_ids = self.tokenizer.tokenize(&prompt, 256);
            let seq_len = token_ids.len();
            let (padded_ids, attention_mask) = self.tokenizer.pad(&token_ids, seq_len);

            let input_ids =
                Tensor::from_array(([1usize, seq_len], padded_ids.into_boxed_slice()))
                    .map_err(|e| RaitheError::NeuralInference(format!("LLM input: {}", e)))?;

            let attn_mask =
                Tensor::from_array(([1usize, seq_len], attention_mask.into_boxed_slice()))
                    .map_err(|e| RaitheError::NeuralInference(format!("LLM attn: {}", e)))?;

            let outputs = match session.run(ort::inputs![input_ids, attn_mask]) {
                Ok(o) => o,
                Err(e) => {
                    warn!("LLM inference failed: {} — skipping reformulation", e);
                    return Ok(None);
                }
            };

            let latency_ms = start.elapsed().as_millis() as u64;

            if latency_ms > 400 {
                warn!("LLM reformulation exceeded 400ms budget: {}ms — discarding", latency_ms);
                return Ok(None);
            }

            let reformulation = parse_llm_output(&outputs, &query);

            debug!(
                query = %query, latency_ms,
                has_result = reformulation.is_some(),
                "LLM query reformulation complete"
            );

            Ok(reformulation)
        })
    }

    fn is_gpu_available(&self) -> bool {
        self.is_gpu_up()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// S1: Probe whether CUDA is available via ONNX Runtime.
fn probe_cuda_available() -> bool {
    // Wrap in catch_unwind — Session::builder() may panic if ORT library
    // failed to load or has an incompatible version.
    let result = std::panic::catch_unwind(|| {
        match Session::builder() {
            Ok(builder) => {
                match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
                    Ok(_) => {
                        info!("CUDA execution provider registered successfully");
                        true
                    }
                    Err(_) => {
                        warn!("CUDA EP not available — CPU inference only");
                        false
                    }
                }
            }
            Err(e) => {
                error!("Failed to create ONNX Runtime session builder: {}", e);
                false
            }
        }
    });

    match result {
        Ok(available) => available,
        Err(_) => {
            error!("CUDA probe panicked — assuming no GPU");
            false
        }
    }
}

/// Load an ONNX model file into a Session.
/// Returns None if the model file does not exist (graceful degradation).
fn load_model(name: &str, model_path: &Path, use_cuda: bool) -> Option<Session> {
    if !model_path.exists() {
        warn!(
            "{} model not found at {:?} — this model will be unavailable",
            name, model_path
        );
        return None;
    }

    info!("Loading {} model from {:?}", name, model_path);

    // Wrap in catch_unwind — ORT operations may panic if the shared library
    // is incompatible or fails to initialize internal state.
    let model_path_owned = model_path.to_path_buf();
    let name_owned = name.to_string();
    let result = std::panic::catch_unwind(move || {
        let mut builder = match Session::builder() {
            Ok(b) => b,
            Err(e) => {
                error!("Session builder creation failed for {}: {}", name_owned, e);
                return None;
            }
        };

        // Apply CUDA EP if requested. On failure, fall back to CPU.
        builder = if use_cuda {
            match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
                Ok(b) => b,
                Err(recoverable) => {
                    warn!("CUDA EP failed for {} — recovering builder for CPU fallback", name_owned);
                    recoverable.recover()
                }
            }
        } else {
            builder
        };

        match builder.commit_from_file(&model_path_owned) {
            Ok(session) => {
                let input_names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
                let output_names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
                info!(
                    "{} model loaded — inputs: {:?}, outputs: {:?}",
                    name_owned, input_names, output_names
                );
                Some(session)
            }
            Err(e) => {
                error!("Failed to load {} from {:?}: {}", name_owned, model_path_owned, e);
                None
            }
        }
    });

    match result {
        Ok(session) => session,
        Err(_) => {
            error!("{} model loading panicked — model will be unavailable", name);
            None
        }
    }
}

/// Extract a fixed-dimensional embedding from bi-encoder output.
/// Handles [1, dim] (pre-pooled) and [1, seq_len, dim] (needs mean-pooling).
fn extract_embedding_from_output(
    outputs: &ort::session::SessionOutputs<'_>,
    target_dim: usize,
) -> Vec<f32> {
    // Try named outputs first, then fall back to index 0.
    let first_output = outputs
        .get("sentence_embedding")
        .or_else(|| outputs.get("last_hidden_state"))
        .or_else(|| {
            if outputs.len() > 0 {
                Some(&outputs[0])
            } else {
                None
            }
        });

    if let Some(output) = first_output {
        if let Ok(array) = output.try_extract_array::<f32>() {
            let shape = array.shape();

            // Shape [1, dim] — already pooled.
            if shape.len() == 2 && shape[0] == 1 {
                let vec: Vec<f32> = array.iter().copied().collect();
                let dim = vec.len().min(target_dim);
                return normalize_l2(&vec[..dim]);
            }

            // Shape [1, seq_len, dim] — mean-pool over sequence.
            if shape.len() == 3 && shape[0] == 1 {
                let seq_len = shape[1];
                let hidden_dim = shape[2];
                let dim = hidden_dim.min(target_dim);
                let mut pooled = vec![0.0f32; dim];

                for s in 0..seq_len {
                    for d in 0..dim {
                        pooled[d] += array[[0, s, d]];
                    }
                }
                for v in pooled.iter_mut() {
                    *v /= seq_len as f32;
                }

                return normalize_l2(&pooled);
            }
        }
    }

    warn!("Could not extract embedding from output — returning zero vector");
    vec![0.0f32; target_dim]
}

/// L2-normalize a vector in-place.
fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Extract relevance scores from cross-encoder output.
/// Handles [batch, 1], [batch], and [batch, 2] (binary logits → softmax).
fn extract_scores_from_output(
    outputs: &ort::session::SessionOutputs<'_>,
    batch_size: usize,
) -> Vec<f64> {
    let first_output = outputs
        .get("logits")
        .or_else(|| {
            if outputs.len() > 0 {
                Some(&outputs[0])
            } else {
                None
            }
        });

    if let Some(output) = first_output {
        if let Ok(array) = output.try_extract_array::<f32>() {
            let shape = array.shape();

            // [batch_size, 1] — squeeze.
            if shape.len() == 2 && shape[0] == batch_size && shape[1] == 1 {
                return (0..batch_size).map(|i| array[[i, 0]] as f64).collect();
            }

            // [batch_size] — direct.
            if shape.len() == 1 && shape[0] == batch_size {
                return array.iter().map(|&v| v as f64).collect();
            }

            // [batch_size, 2] — binary classification logits → softmax.
            if shape.len() == 2 && shape[0] == batch_size && shape[1] == 2 {
                return (0..batch_size)
                    .map(|i| {
                        let l0 = array[[i, 0]] as f64;
                        let l1 = array[[i, 1]] as f64;
                        let max = l0.max(l1);
                        let exp0 = (l0 - max).exp();
                        let exp1 = (l1 - max).exp();
                        exp1 / (exp0 + exp1)
                    })
                    .collect();
            }

            // Fallback: flatten and take first batch_size values.
            let flat: Vec<f64> = array.iter().map(|&v| v as f64).collect();
            if flat.len() >= batch_size {
                return flat[..batch_size].to_vec();
            }
        }
    }

    warn!("Could not extract cross-encoder scores — returning uniform 0.5");
    vec![0.5; batch_size]
}

/// Parse LLM output into a LlmReformulation struct.
///
/// When a real vocabulary-based tokenizer is integrated, this will decode
/// output token IDs → text → parse JSON. For now, if the model produced
/// output, we generate a stop-word-stripped reformulation.
fn parse_llm_output(
    outputs: &ort::session::SessionOutputs<'_>,
    original_query: &str,
) -> Option<LlmReformulation> {
    if outputs.len() == 0 {
        return None;
    }

    // Basic reformulation: strip common stop words.
    let reformulated = original_query
        .to_lowercase()
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the" | "is" | "are" | "was" | "were" | "be"))
        .collect::<Vec<_>>()
        .join(" ");

    Some(LlmReformulation {
        reformulated_query: reformulated,
        alternatives: Vec::new(),
        entities: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_basic() {
        let tok = SimpleTokenizer::new();
        let ids = tok.tokenize("hello world", 10);
        assert_eq!(ids[0], 101); // CLS
        assert_eq!(*ids.last().unwrap(), 102); // SEP
        assert_eq!(ids.len(), 4); // CLS + hello + world + SEP
    }

    #[test]
    fn test_simple_tokenizer_truncation() {
        let tok = SimpleTokenizer::new();
        let ids = tok.tokenize("one two three four five six", 5);
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], 101);
        assert_eq!(ids[4], 102);
    }

    #[test]
    fn test_tokenize_pair() {
        let tok = SimpleTokenizer::new();
        let (ids, mask, type_ids) = tok.tokenize_pair("hello", "world", 4, 4);
        let max_length = 4 + 4 + 3;
        assert_eq!(ids.len(), max_length);
        assert_eq!(mask.len(), max_length);
        assert_eq!(type_ids.len(), max_length);
        assert_eq!(ids[0], 101); // CLS
        assert_eq!(type_ids[0], 0); // query segment
    }

    #[test]
    fn test_tokenizer_deterministic() {
        let tok = SimpleTokenizer::new();
        let ids1 = tok.tokenize("rust programming language", 10);
        let ids2 = tok.tokenize("rust programming language", 10);
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn test_pad() {
        let tok = SimpleTokenizer::new();
        let ids = vec![101, 500, 600, 102];
        let (padded, mask) = tok.pad(&ids, 8);
        assert_eq!(padded.len(), 8);
        assert_eq!(padded[4], 0);
        assert_eq!(mask, vec![1, 1, 1, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_normalize_l2() {
        let v = vec![3.0, 4.0];
        let n = normalize_l2(&v);
        assert!((n[0] - 0.6).abs() < 1e-5);
        assert!((n[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l2_zero() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize_l2(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gpu_manager_creation_no_models() {
        let config = NeuralConfig {
            cross_encoder_model_path: "/nonexistent/cross_encoder.onnx".into(),
            bi_encoder_model_path: "/nonexistent/bi_encoder.onnx".into(),
            llm_model_path: "/nonexistent/query_llm.gguf".into(),
            use_cuda: false,
            cross_encoder_vram_mb: 1536,
            bi_encoder_vram_mb: 512,
            llm_vram_mb: 4096,
        };

        let manager = GpuInferenceManager::new(&config).unwrap();
        assert!(!manager.is_gpu_available());
        assert_eq!(manager.estimated_vram_usage(), 0);
    }

    #[tokio::test]
    async fn test_embed_query_no_model() {
        let config = NeuralConfig {
            cross_encoder_model_path: "/nonexistent/ce.onnx".into(),
            bi_encoder_model_path: "/nonexistent/be.onnx".into(),
            llm_model_path: "/nonexistent/llm.gguf".into(),
            use_cuda: false,
            cross_encoder_vram_mb: 1536,
            bi_encoder_vram_mb: 512,
            llm_vram_mb: 4096,
        };

        let manager = GpuInferenceManager::new(&config).unwrap();
        let embedding = manager.embed_query("test query").await.unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!(embedding.iter().all(|&v| v == 0.0));
    }

    #[tokio::test]
    async fn test_cross_encode_no_model() {
        let config = NeuralConfig {
            cross_encoder_model_path: "/nonexistent/ce.onnx".into(),
            bi_encoder_model_path: "/nonexistent/be.onnx".into(),
            llm_model_path: "/nonexistent/llm.gguf".into(),
            use_cuda: false,
            cross_encoder_vram_mb: 1536,
            bi_encoder_vram_mb: 512,
            llm_vram_mb: 4096,
        };

        let manager = GpuInferenceManager::new(&config).unwrap();
        let result = manager
            .cross_encode("test", &["passage one".to_string()])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_reformulate_no_model() {
        let config = NeuralConfig {
            cross_encoder_model_path: "/nonexistent/ce.onnx".into(),
            bi_encoder_model_path: "/nonexistent/be.onnx".into(),
            llm_model_path: "/nonexistent/llm.gguf".into(),
            use_cuda: false,
            cross_encoder_vram_mb: 1536,
            bi_encoder_vram_mb: 512,
            llm_vram_mb: 4096,
        };

        let manager = GpuInferenceManager::new(&config).unwrap();
        let result = manager.reformulate_query("how does rust work").await.unwrap();
        assert!(result.is_none());
    }
}
