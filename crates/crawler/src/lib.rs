// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler
//
// Async HTTP crawler with politeness enforcement, robots.txt caching,
// sitemap parsing, and bandwidth-aware throttling.
//
// Runs as a background task pool within the shared tokio runtime.
// Yields CPU to the serving path under load.
// ================================================================================

pub mod bandwidth;
pub mod fetcher;
pub mod frontier;
pub mod robots;
pub mod scheduler;
pub mod spam;

use raithe_common::config::RaitheConfig;
use raithe_common::traits::CrawlEmitter;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};
use url::Url;

/// The main crawler engine that coordinates all crawl subsystems.
pub struct Crawler {
    config: Arc<RaitheConfig>,
    frontier: frontier::UrlFrontier,
    robots_cache: robots::RobotsCache,
    bandwidth_monitor: bandwidth::BandwidthMonitor,
    /// Semaphore limiting concurrent HTTP requests.
    fetch_semaphore: Arc<Semaphore>,
    /// Semaphore limiting concurrent parse operations.
    parse_semaphore: Arc<Semaphore>,
    /// The crawl log emitter.
    emitter: Arc<dyn CrawlEmitter>,
    /// Shared HTTP client.
    http_client: reqwest::Client,
    /// Flag to pause crawling when bandwidth is saturated.
    paused: Arc<std::sync::atomic::AtomicBool>,
}

impl Crawler {
    /// Create a new crawler with the given configuration and emitter.
    pub async fn new(
        config: Arc<RaitheConfig>,
        emitter: Arc<dyn CrawlEmitter>,
    ) -> anyhow::Result<Self> {
        let frontier = frontier::UrlFrontier::new(&config.storage.frontier_db_path)?;

        let robots_cache = robots::RobotsCache::new(
            config.crawler.robots_cache_size,
            config.crawler.robots_cache_ttl_secs,
        );

        let bandwidth_monitor = bandwidth::BandwidthMonitor::new(
            config.crawler.bandwidth_pause_threshold_pct,
            config.crawler.bandwidth_resume_threshold_pct,
        );

        let fetch_semaphore = Arc::new(Semaphore::new(config.crawler.max_concurrent_requests));
        let parse_semaphore = Arc::new(Semaphore::new(config.crawler.max_concurrent_parses));

        let http_client = fetcher::build_http_client(
            config.crawler.max_connections_per_host,
            config.crawler.total_connections,
            config.crawler.http_timeout_secs,
        )?;

        Ok(Self {
            config,
            frontier,
            robots_cache,
            bandwidth_monitor,
            fetch_semaphore,
            parse_semaphore,
            emitter,
            http_client,
            paused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Run the crawler loop. Runs continuously in the background,
    /// yielding to serving under load.
    pub async fn run(&self) -> anyhow::Result<()> {
        info!(
            "RAiTHE Crawler starting — max {} concurrent requests",
            self.config.crawler.max_concurrent_requests
        );

        loop {
            // Check bandwidth — pause if upload is saturated.
            if self.bandwidth_monitor.should_pause() {
                self.paused
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                debug!("Crawler paused — upload bandwidth saturated");
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                continue;
            }
            self.paused
                .store(false, std::sync::atomic::Ordering::Relaxed);

            // Pop the next URL from the frontier.
            let entry = match self.frontier.pop() {
                Some(entry) => entry,
                None => {
                    debug!("Frontier empty — waiting 10s");
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    continue;
                }
            };

            // Acquire fetch permit (limits concurrency).
            let _permit = self.fetch_semaphore.acquire().await?;

            // Yield to other tasks.
            tokio::task::yield_now().await;

            // Execute the full crawl pipeline for this URL.
            if let Err(e) = self.crawl_url(&entry.url).await {
                warn!(url = %entry.url, error = %e, "Crawl failed");
            }

            // Per-domain politeness delay.
            let delay_ms = self
                .robots_cache
                .crawl_delay_ms(&entry.domain, self.config.crawler.default_crawl_delay_ms);
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
    }

    /// Execute the full crawl pipeline for a single URL.
    ///
    /// DNS → TLS → HTTP fetch → robots check → parse → dedup → link extraction → emit.
    async fn crawl_url(&self, url_str: &str) -> anyhow::Result<()> {
        let parsed_url =
            Url::parse(url_str).map_err(|e| anyhow::anyhow!("Invalid URL {}: {}", url_str, e))?;

        let domain = parsed_url
            .host_str()
            .unwrap_or("unknown")
            .to_string();

        // Step 1: Check robots.txt.
        let path = parsed_url.path();
        if let Some(allowed) = self.robots_cache.is_allowed(&domain, path) {
            if !allowed {
                debug!(url = %url_str, "Blocked by robots.txt");
                return Ok(());
            }
        } else {
            // Fetch and cache robots.txt for this domain.
            self.fetch_and_cache_robots(&parsed_url).await;
            if let Some(false) = self.robots_cache.is_allowed(&domain, path) {
                debug!(url = %url_str, "Blocked by robots.txt");
                return Ok(());
            }
        }

        // Step 2: HTTP fetch.
        let fetch_result = fetcher::fetch_url(
            &self.http_client,
            url_str,
            self.config.crawler.max_body_size,
        )
        .await?;

        if fetch_result.status == 429 || fetch_result.status == 503 {
            let backoff = robots::backoff_delay(1);
            warn!(
                url = %url_str,
                status = fetch_result.status,
                backoff_ms = backoff.as_millis() as u64,
                "Rate limited — backing off"
            );
            tokio::time::sleep(backoff).await;
            return Ok(());
        }

        if fetch_result.status >= 400 {
            debug!(url = %url_str, status = fetch_result.status, "HTTP error");
            return Ok(());
        }

        // Step 3: Parse HTML.
        let _parse_permit = self.parse_semaphore.acquire().await?;

        let parsed = raithe_parser::parse_html(
            &fetch_result.body,
            url_str,
            fetch_result.status,
            &fetch_result.content_type,
        )
        .map_err(|e| anyhow::anyhow!("Parse error for {}: {}", url_str, e))?;

        // Step 4: SimHash dedup.
        if let Some(canonical) = self.frontier.check_simhash_dedup(parsed.simhash) {
            debug!(
                url = %url_str,
                canonical = %canonical,
                "Near-duplicate detected — skipping"
            );
            return Ok(());
        }
        self.frontier.insert_simhash(parsed.simhash, url_str)?;

        // Step 5: Enqueue discovered links back into frontier.
        let mut links_enqueued = 0u32;
        for link in &parsed.outgoing_links {
            if !link.is_followed {
                continue;
            }
            // Normalize and enqueue with decayed priority (0.85 factor).
            if let Some(normalized) = fetcher::normalize_url(&link.url, Some(&parsed_url)) {
                // Only enqueue http/https.
                if normalized.starts_with("http://") || normalized.starts_with("https://") {
                    let _ = self.frontier.push_url(&normalized, 0.85);
                    links_enqueued += 1;
                }
            }
        }

        // Step 6: Emit to crawl log for indexer.
        self.emitter.emit(parsed.clone()).await.map_err(|e| {
            anyhow::anyhow!("Failed to emit document for {}: {}", url_str, e)
        })?;

        info!(
            url = %url_str,
            title = %parsed.title,
            words = parsed.word_count,
            links = links_enqueued,
            "Crawled and emitted"
        );

        // Step 7: Record crawl for adaptive scheduling.
        let content_hash = format!("{:016x}", parsed.simhash);
        let _ = self
            .frontier
            .record_crawl(url_str, &content_hash, 0.5);

        Ok(())
    }

    /// Fetch and cache robots.txt for a domain.
    async fn fetch_and_cache_robots(&self, url: &Url) {
        let robots_url = format!(
            "{}://{}/robots.txt",
            url.scheme(),
            url.host_str().unwrap_or("unknown")
        );
        let domain = url.host_str().unwrap_or("unknown").to_string();

        match self.http_client.get(&robots_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(body) = resp.text().await {
                    let rules = robots::RobotsRules::parse(&body);
                    self.robots_cache.insert(domain, rules);
                }
            }
            _ => {
                // No robots.txt or error — allow everything.
                let rules = robots::RobotsRules::parse("");
                self.robots_cache.insert(domain, rules);
            }
        }
    }

    /// Seed the frontier with initial URLs.
    pub fn seed(&self, urls: &[&str]) -> anyhow::Result<()> {
        info!("Seeding crawler frontier with {} URLs", urls.len());
        for url_str in urls {
            self.frontier.push_url(url_str, 1.0)?;
        }
        Ok(())
    }

    /// Check if the crawler is currently paused due to bandwidth.
    pub fn is_paused(&self) -> bool {
        self.paused.load(std::sync::atomic::Ordering::Relaxed)
    }
}
