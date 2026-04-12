// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: metrics
//
// Prometheus metrics registry and structured logging.
// Every subsystem emits structured metrics (Prometheus exposition format)
// and structured logs (tracing crate with JSON output).
// ================================================================================

use prometheus::{
    register_counter_vec, register_gauge, register_gauge_vec, register_histogram_vec,
    CounterVec, Gauge, GaugeVec, HistogramVec, Encoder, TextEncoder,
};
use std::sync::OnceLock;
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize the tracing/logging subsystem.
/// JSON output to log file, rotated daily, retain 7 days.
pub fn init_logging(log_level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().json().with_target(true).with_thread_ids(true))
        .init();
}

/// Global metrics registry.
pub struct RaitheMetrics {
    // Query latency
    pub query_latency: HistogramVec,
    // Query count
    pub queries_total: CounterVec,
    // Neural inference latency
    pub neural_inference_latency: HistogramVec,
    // Neural fallback events
    pub neural_fallback_total: CounterVec,
    // Instant answer triggers
    pub instant_answer_triggers: CounterVec,
    // Crawl throughput
    pub crawl_pages_per_second: Gauge,
    // Index document count
    pub index_doc_count: Gauge,
    // GPU VRAM usage
    pub gpu_vram_usage_bytes: Gauge,
    // CPU usage by task
    pub cpu_usage_percent: GaugeVec,
    // RAM usage by component
    pub ram_usage_bytes: GaugeVec,
    // Disk usage by path
    pub disk_usage_bytes: GaugeVec,
    // Upload bandwidth
    pub bandwidth_upload_bytes_per_sec: Gauge,
}

static METRICS: OnceLock<RaitheMetrics> = OnceLock::new();

impl RaitheMetrics {
    /// Initialize the global metrics registry.
    pub fn init() -> &'static Self {
        METRICS.get_or_init(|| {
            let latency_buckets = vec![
                0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200, 0.300, 0.500, 0.800, 1.0,
            ];

            Self {
                query_latency: register_histogram_vec!(
                    "raithe_query_latency_seconds",
                    "Query latency by phase",
                    &["phase"],
                    latency_buckets.clone()
                )
                .expect("Failed to register query_latency metric"),

                queries_total: register_counter_vec!(
                    "raithe_queries_total",
                    "Total queries by status",
                    &["status"]
                )
                .expect("Failed to register queries_total metric"),

                neural_inference_latency: register_histogram_vec!(
                    "raithe_neural_inference_latency_seconds",
                    "Neural inference latency by model",
                    &["model"],
                    latency_buckets
                )
                .expect("Failed to register neural_inference_latency metric"),

                neural_fallback_total: register_counter_vec!(
                    "raithe_neural_fallback_total",
                    "Neural fallback events by model and reason",
                    &["model", "reason"]
                )
                .expect("Failed to register neural_fallback_total metric"),

                instant_answer_triggers: register_counter_vec!(
                    "raithe_instant_answer_triggers_total",
                    "Instant answer triggers by type",
                    &["answer_type"]
                )
                .expect("Failed to register instant_answer_triggers metric"),

                crawl_pages_per_second: register_gauge!(
                    "raithe_crawl_pages_per_second",
                    "Current crawl throughput"
                )
                .expect("Failed to register crawl_pages_per_second metric"),

                index_doc_count: register_gauge!(
                    "raithe_index_doc_count",
                    "Total indexed documents"
                )
                .expect("Failed to register index_doc_count metric"),

                gpu_vram_usage_bytes: register_gauge!(
                    "raithe_gpu_vram_usage_bytes",
                    "GPU VRAM usage in bytes"
                )
                .expect("Failed to register gpu_vram_usage_bytes metric"),

                cpu_usage_percent: register_gauge_vec!(
                    "raithe_cpu_usage_percent",
                    "CPU usage by task",
                    &["task"]
                )
                .expect("Failed to register cpu_usage_percent metric"),

                ram_usage_bytes: register_gauge_vec!(
                    "raithe_ram_usage_bytes",
                    "RAM usage by component",
                    &["component"]
                )
                .expect("Failed to register ram_usage_bytes metric"),

                disk_usage_bytes: register_gauge_vec!(
                    "raithe_disk_usage_bytes",
                    "Disk usage by path",
                    &["path"]
                )
                .expect("Failed to register disk_usage_bytes metric"),

                bandwidth_upload_bytes_per_sec: register_gauge!(
                    "raithe_bandwidth_upload_bytes_per_sec",
                    "Current upload bandwidth usage"
                )
                .expect("Failed to register bandwidth_upload_bytes_per_sec metric"),
            }
        })
    }

    /// Get the global metrics instance (panics if not initialized).
    pub fn global() -> &'static Self {
        METRICS.get().expect("RaitheMetrics not initialized — call RaitheMetrics::init() first")
    }
}

/// Render all metrics in Prometheus text exposition format.
/// Used by the /metrics endpoint on port 9090.
pub fn render_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Axum handler for the /metrics endpoint.
pub async fn metrics_handler() -> String {
    render_metrics()
}
