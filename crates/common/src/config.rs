// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: common :: config
//
// Configuration loading via figment: defaults -> config file -> env -> CLI flags.
// Hot-reload of config values via notify file watcher.
// ================================================================================

use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::watch;
use tracing::{info, warn};

/// Default configuration file path.
pub const DEFAULT_CONFIG_PATH: &str = "/data/config/engine.toml";

/// Top-level engine configuration, mirroring all tunable parameters from the spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaitheConfig {
    pub general: GeneralConfig,
    pub crawler: CrawlerConfig,
    pub indexer: IndexerConfig,
    pub ranker: RankerConfig,
    pub query: QueryConfig,
    pub serving: ServingConfig,
    pub linkgraph: LinkGraphConfig,
    pub semantic: SemanticConfig,
    pub neural: NeuralConfig,
    pub instant: InstantConfig,
    pub storage: StorageConfig,
    pub session: SessionConfig,
    pub features: FeatureFlags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Base data directory (default: /data/)
    pub data_dir: PathBuf,
    /// Hostname for TLS and branding
    pub hostname: String,
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlerConfig {
    /// Max concurrent HTTP requests (default: 8)
    pub max_concurrent_requests: usize,
    /// Max connections per host (default: 4)
    pub max_connections_per_host: usize,
    /// Total connection pool size (default: 200)
    pub total_connections: usize,
    /// Default crawl delay per domain in ms (default: 1000)
    pub default_crawl_delay_ms: u64,
    /// Max response body size in bytes (default: 5MB)
    pub max_body_size: usize,
    /// HTTP timeout per page in seconds (default: 30)
    pub http_timeout_secs: u64,
    /// robots.txt cache size (default: 1_000_000)
    pub robots_cache_size: u64,
    /// robots.txt cache TTL in seconds (default: 86400 = 24h)
    pub robots_cache_ttl_secs: u64,
    /// SimHash hamming distance threshold (default: 3)
    pub simhash_distance_threshold: u32,
    /// Upload bandwidth pause threshold percent (default: 50)
    pub bandwidth_pause_threshold_pct: u8,
    /// Upload bandwidth resume threshold percent (default: 30)
    pub bandwidth_resume_threshold_pct: u8,
    /// Max concurrent parse operations (default: 4)
    pub max_concurrent_parses: usize,
    /// Minimum re-crawl interval in hours (default: 12)
    pub min_recrawl_interval_hours: u64,
    /// Maximum re-crawl interval in days (default: 30)
    pub max_recrawl_interval_days: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Max merge threads (default: 1)
    pub max_merge_threads: usize,
    /// Target merge segment size in bytes (default: 5GB)
    pub target_merge_size_bytes: u64,
    /// QPS threshold for allowing large merges (default: 1.0)
    pub merge_qps_threshold: f64,
    /// Indexer poll interval in seconds (default: 60)
    pub poll_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankerConfig {
    /// BM25F parameters
    pub bm25f_k1: f32,
    pub bm25f_b_body: f32,
    pub bm25f_b_title: f32,
    pub bm25f_b_anchor: f32,
    pub bm25f_w_title: f32,
    pub bm25f_w_body: f32,
    pub bm25f_w_anchor: f32,
    pub bm25f_w_url: f32,
    pub bm25f_w_headings: f32,
    /// Number of BM25F candidates to retrieve (default: 500)
    pub bm25f_candidates: usize,
    /// Number of ANN candidates (default: 100)
    pub ann_candidates: usize,
    /// RRF fusion parameter k (default: 60)
    pub rrf_k: usize,
    /// GBDT: number of trees (default: 300)
    pub gbdt_num_trees: usize,
    /// GBDT: max depth (default: 6)
    pub gbdt_max_depth: usize,
    /// Cross-encoder: number of candidates to re-rank (default: 20)
    pub cross_encoder_candidates: usize,
    /// Cross-encoder: query max tokens (default: 32)
    pub cross_encoder_query_max_tokens: usize,
    /// Cross-encoder: passage max tokens (default: 128)
    pub cross_encoder_passage_max_tokens: usize,
    /// Cross-encoder: latency budget ms (default: 50)
    pub cross_encoder_latency_budget_ms: u64,
    /// Cross-encoder: fallback threshold ms (default: 100)
    pub cross_encoder_fallback_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Fast path budget ms (default: 10)
    pub fast_path_budget_ms: u64,
    /// Enhanced path budget ms (default: 200)
    pub enhanced_path_budget_ms: u64,
    /// Enhanced path fallback ms (default: 400)
    pub enhanced_path_fallback_ms: u64,
    /// Intent classifier confidence threshold for enhanced path (default: 0.7)
    pub intent_confidence_threshold: f32,
    /// Query token count threshold for enhanced path (default: 8)
    pub enhanced_path_token_threshold: usize,
    /// LRU cache size for LLM reformulations (default: 100_000)
    pub llm_cache_size: u64,
    /// Max spell correction edit distance for short queries (default: 1)
    pub spell_max_edit_short: u32,
    /// Max spell correction edit distance for long queries (default: 2)
    pub spell_max_edit_long: u32,
    /// Synonym expansion weight (default: 0.3)
    pub synonym_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingConfig {
    /// HTTP listen address (default: 0.0.0.0)
    pub listen_addr: String,
    /// HTTP listen port (default: 8080)
    pub listen_port: u16,
    /// Metrics listen port (default: 9090)
    pub metrics_port: u16,
    /// Global QPS limit (default: 50)
    pub global_qps_limit: u32,
    /// Per-IP QPS limit (default: 50)
    pub per_ip_qps_limit: u32,
    /// Max snippet length in characters (default: 300)
    pub max_snippet_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkGraphConfig {
    /// PageRank damping factor (default: 0.85)
    pub pagerank_damping: f64,
    /// PageRank convergence threshold (default: 1e-6)
    pub pagerank_convergence_threshold: f64,
    /// PageRank max iterations (default: 50)
    pub pagerank_max_iterations: usize,
    /// Number of topic-specific PageRank vectors (default: 4)
    pub topic_pagerank_count: usize,
    /// Max anchor texts per URL (default: 100)
    pub max_anchor_texts_per_url: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Embedding dimension (default: 128)
    pub embedding_dim: usize,
    /// HNSW M parameter (default: 16)
    pub hnsw_m: usize,
    /// HNSW ef_construction (default: 100)
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search (default: 50)
    pub hnsw_ef_search: usize,
    /// PQ sub-quantizers (default: 16)
    pub pq_sub_quantizers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Path to cross-encoder ONNX model
    pub cross_encoder_model_path: PathBuf,
    /// Path to bi-encoder ONNX model
    pub bi_encoder_model_path: PathBuf,
    /// Path to small LLM model (ONNX or GGUF)
    pub llm_model_path: PathBuf,
    /// Use CUDA EP (default: true)
    pub use_cuda: bool,
    /// VRAM budget for cross-encoder in MB (default: 1536)
    pub cross_encoder_vram_mb: usize,
    /// VRAM budget for bi-encoder in MB (default: 512)
    pub bi_encoder_vram_mb: usize,
    /// VRAM budget for LLM in MB (default: 4096)
    pub llm_vram_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantConfig {
    /// Path to Wikidata knowledge graph SQLite DB
    pub knowledge_db_path: PathBuf,
    /// Path to Wiktionary definitions SQLite DB
    pub definitions_db_path: PathBuf,
    /// Instant answer confidence threshold (default: 0.8)
    pub confidence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Index directory (default: /data/index/)
    pub index_dir: PathBuf,
    /// Vectors directory (default: /data/vectors/)
    pub vectors_dir: PathBuf,
    /// Link graph directory (default: /data/linkgraph/)
    pub linkgraph_dir: PathBuf,
    /// Crawl log directory (default: /data/crawl-log/)
    pub crawl_log_dir: PathBuf,
    /// URL frontier SQLite path
    pub frontier_db_path: PathBuf,
    /// Knowledge graph directory (default: /data/knowledge/)
    pub knowledge_dir: PathBuf,
    /// Models directory (default: /data/models/)
    pub models_dir: PathBuf,
    /// Backups directory (default: /data/backups/)
    pub backups_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Max sessions in LRU cache (default: 1000)
    pub max_sessions: u64,
    /// Session TTL in seconds (default: 1800 = 30min)
    pub session_ttl_secs: u64,
    /// Queries to track per session (default: 5)
    pub queries_per_session: usize,
}

/// Feature flags — boolean toggles for all optional subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable cross-encoder neural re-ranking (3)
    pub enable_cross_encoder: bool,
    /// Enable LLM-assisted query understanding
    pub enable_llm_query: bool,
    /// Enable semantic ANN retrieval path
    pub enable_ann_retrieval: bool,
    /// Enable instant answers module
    pub enable_instant_answers: bool,
    /// Enable AI-generated content detection
    pub enable_ai_content_detection: bool,
    /// Enable session context features
    pub enable_session_context: bool,
}

impl Default for RaitheConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                data_dir: PathBuf::from("/data"),
                hostname: "raithe.ca".into(),
                log_level: "info".into(),
            },
            crawler: CrawlerConfig {
                max_concurrent_requests: 8,
                max_connections_per_host: 4,
                total_connections: 200,
                default_crawl_delay_ms: 1000,
                max_body_size: 5 * 1024 * 1024, // 5 MB
                http_timeout_secs: 30,
                robots_cache_size: 1_000_000,
                robots_cache_ttl_secs: 86400,
                simhash_distance_threshold: 3,
                bandwidth_pause_threshold_pct: 50,
                bandwidth_resume_threshold_pct: 30,
                max_concurrent_parses: 4,
                min_recrawl_interval_hours: 12,
                max_recrawl_interval_days: 30,
            },
            indexer: IndexerConfig {
                max_merge_threads: 1,
                target_merge_size_bytes: 5 * 1024 * 1024 * 1024, // 5 GB
                merge_qps_threshold: 1.0,
                poll_interval_secs: 60,
            },
            ranker: RankerConfig {
                bm25f_k1: 1.2,
                bm25f_b_body: 0.75,
                bm25f_b_title: 0.3,
                bm25f_b_anchor: 0.1,
                bm25f_w_title: 5.0,
                bm25f_w_body: 1.0,
                bm25f_w_anchor: 4.0,
                bm25f_w_url: 2.0,
                bm25f_w_headings: 2.5,
                bm25f_candidates: 500,
                ann_candidates: 100,
                rrf_k: 60,
                gbdt_num_trees: 300,
                gbdt_max_depth: 6,
                cross_encoder_candidates: 20,
                cross_encoder_query_max_tokens: 32,
                cross_encoder_passage_max_tokens: 128,
                cross_encoder_latency_budget_ms: 50,
                cross_encoder_fallback_ms: 100,
            },
            query: QueryConfig {
                fast_path_budget_ms: 10,
                enhanced_path_budget_ms: 200,
                enhanced_path_fallback_ms: 400,
                intent_confidence_threshold: 0.7,
                enhanced_path_token_threshold: 8,
                llm_cache_size: 100_000,
                spell_max_edit_short: 1,
                spell_max_edit_long: 2,
                synonym_weight: 0.3,
            },
            serving: ServingConfig {
                listen_addr: "0.0.0.0".into(),
                listen_port: 8080,
                metrics_port: 9090,
                global_qps_limit: 50,
                per_ip_qps_limit: 50,
                max_snippet_length: 300,
            },
            linkgraph: LinkGraphConfig {
                pagerank_damping: 0.85,
                pagerank_convergence_threshold: 1e-6,
                pagerank_max_iterations: 50,
                topic_pagerank_count: 4,
                max_anchor_texts_per_url: 100,
            },
            semantic: SemanticConfig {
                embedding_dim: 384,
                hnsw_m: 16,
                hnsw_ef_construction: 100,
                hnsw_ef_search: 50,
                pq_sub_quantizers: 16,
            },
            neural: NeuralConfig {
                cross_encoder_model_path: PathBuf::from("/data/models/deberta/onnx/model.onnx"),
                bi_encoder_model_path: PathBuf::from("/data/models/minilm/onnx/model_qint8_avx2.onnx"),
                llm_model_path: PathBuf::from("/data/models/flan-t5/onnx/model.onnx"), // OR gguf if replaced
                use_cuda: true,
                cross_encoder_vram_mb: 1536,
                bi_encoder_vram_mb: 512,
                llm_vram_mb: 4096,
            },
            instant: InstantConfig {
                knowledge_db_path: PathBuf::from("/data/knowledge/wikidata.db"),
                definitions_db_path: PathBuf::from("/data/knowledge/wiktionary.db"),
                confidence_threshold: 0.8,
            },
            storage: StorageConfig {
                index_dir: PathBuf::from("/data/index"),
                vectors_dir: PathBuf::from("/data/vectors"),
                linkgraph_dir: PathBuf::from("/data/linkgraph"),
                crawl_log_dir: PathBuf::from("/data/crawl-log"),
                frontier_db_path: PathBuf::from("/data/frontier/frontier.db"),
                knowledge_dir: PathBuf::from("/data/knowledge"),
                models_dir: PathBuf::from("/data/models"),
                backups_dir: PathBuf::from("/data/backups"),
            },
            session: SessionConfig {
                max_sessions: 1000,
                session_ttl_secs: 1800,
                queries_per_session: 5,
            },
            features: FeatureFlags {
                enable_cross_encoder: true,
                enable_llm_query: true,
                enable_ann_retrieval: true,
                enable_instant_answers: true,
                enable_ai_content_detection: true,
                enable_session_context: true,
            },
        }
    }
}

impl RaitheConfig {
    /// Load configuration following figment merge order:
    /// defaults (compiled-in) -> config file -> environment variables -> CLI overrides.
    pub fn load(config_path: Option<&Path>) -> Result<Self, figment::Error> {
        let path = config_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));

        let mut figment = Figment::from(Serialized::defaults(Self::default()));

        if path.exists() {
            figment = figment.merge(Toml::file(&path));
        }

        figment = figment.merge(Env::prefixed("RAITHE_").split("__"));

        figment.extract()
    }

    /// Spawn a background file watcher that sends updated configs through the channel.
    /// The engine picks up changes within 5 seconds.
    pub fn watch(
        config_path: PathBuf,
    ) -> (watch::Receiver<Arc<RaitheConfig>>, tokio::task::JoinHandle<()>) {
        let initial = Self::load(Some(&config_path)).unwrap_or_default();
        let (tx, rx) = watch::channel(Arc::new(initial));

        let handle = tokio::spawn(async move {
            use notify::{RecommendedWatcher, RecursiveMode, Watcher};
            use std::sync::mpsc;
            use std::time::Duration;

            let (file_tx, file_rx) = mpsc::channel();

            let mut watcher: RecommendedWatcher =
                notify::Watcher::new(file_tx, notify::Config::default().with_poll_interval(Duration::from_secs(5)))
                    .expect("Failed to create config file watcher");

            if let Err(e) = watcher.watch(&config_path, RecursiveMode::NonRecursive) {
                warn!("Failed to watch config file {:?}: {}", config_path, e);
                return;
            }

            info!("Watching config file {:?} for changes", config_path);

            loop {
                match file_rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(_event) => {
                        match Self::load(Some(&config_path)) {
                            Ok(new_config) => {
                                info!("Config file changed — reloading");
                                let _ = tx.send(Arc::new(new_config));
                            }
                            Err(e) => {
                                warn!("Failed to reload config: {}", e);
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        });

        (rx, handle)
    }
}
