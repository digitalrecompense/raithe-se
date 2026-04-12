// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler :: frontier
//
// URL frontier backed by SQLite.
//
// Schema:
//   - frontier: URLs with priority, crawl history, and scheduling.
//   - crawl_history: per-URL history of content hashes for change detection.
//   - simhash_dedup: near-duplicate detection via SimHash.
// ================================================================================

use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;
use tracing::info;
use url::Url;

/// A single entry in the URL frontier.
#[derive(Debug, Clone)]
pub struct FrontierEntry {
    pub url: String,
    pub domain: String,
    pub priority: f64,
    pub last_crawled: Option<chrono::DateTime<chrono::Utc>>,
    pub change_probability: f64,
    pub crawl_count: u32,
}

/// URL frontier backed by SQLite.
pub struct UrlFrontier {
    conn: Mutex<Connection>,
}

impl UrlFrontier {
    /// Create or open the frontier database at the given path.
    pub fn new(db_path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;

        // Initialize schema (idempotent).
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS frontier (
                url             TEXT PRIMARY KEY,
                domain          TEXT NOT NULL,
                priority        REAL NOT NULL DEFAULT 0.0,
                last_crawled    TEXT,
                change_prob     REAL NOT NULL DEFAULT 0.5,
                crawl_count     INTEGER NOT NULL DEFAULT 0,
                next_crawl_at   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_frontier_priority
                ON frontier (priority DESC);

            CREATE INDEX IF NOT EXISTS idx_frontier_domain
                ON frontier (domain);

            CREATE INDEX IF NOT EXISTS idx_frontier_next_crawl
                ON frontier (next_crawl_at ASC);

            -- Per-URL change history for adaptive scheduling
            CREATE TABLE IF NOT EXISTS crawl_history (
                url             TEXT NOT NULL,
                crawled_at      TEXT NOT NULL,
                content_hash    TEXT NOT NULL,
                FOREIGN KEY (url) REFERENCES frontier(url)
            );

            CREATE INDEX IF NOT EXISTS idx_crawl_history_url
                ON crawl_history (url, crawled_at DESC);

            -- SimHash dedup table
            CREATE TABLE IF NOT EXISTS simhash_dedup (
                simhash         INTEGER NOT NULL,
                canonical_url   TEXT NOT NULL,
                PRIMARY KEY (simhash)
            );"
        )?;

        info!("URL frontier database ready at {:?}", db_path);

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Push a URL into the frontier with a given priority.
    pub fn push_url(&self, url_str: &str, priority: f64) -> anyhow::Result<()> {
        let parsed = Url::parse(url_str)?;
        let domain = parsed.host_str().unwrap_or("unknown").to_string();

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO frontier (url, domain, priority) VALUES (?1, ?2, ?3)",
            params![url_str, domain, priority],
        )?;

        Ok(())
    }

    /// Push a FrontierEntry into the frontier.
    pub fn push(&self, entry: &FrontierEntry) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO frontier
             (url, domain, priority, last_crawled, change_prob, crawl_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                entry.url,
                entry.domain,
                entry.priority,
                entry.last_crawled.map(|dt| dt.to_rfc3339()),
                entry.change_probability,
                entry.crawl_count,
            ],
        )?;

        Ok(())
    }

    /// Pop the highest-priority URL that is ready to be crawled.
    pub fn pop(&self) -> Option<FrontierEntry> {
        let conn = self.conn.lock().unwrap();

        let result = conn.query_row(
            "SELECT url, domain, priority, last_crawled, change_prob, crawl_count
             FROM frontier
             WHERE next_crawl_at IS NULL OR next_crawl_at <= datetime('now')
             ORDER BY priority DESC
             LIMIT 1",
            [],
            |row| {
                Ok(FrontierEntry {
                    url: row.get(0)?,
                    domain: row.get(1)?,
                    priority: row.get(2)?,
                    last_crawled: row.get::<_, Option<String>>(3)?
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&chrono::Utc)),
                    change_probability: row.get(4)?,
                    crawl_count: row.get(5)?,
                })
            },
        );

        match result {
            Ok(entry) => {
                // Mark as in-progress by setting a future next_crawl_at.
                let _ = conn.execute(
                    "UPDATE frontier SET next_crawl_at = datetime('now', '+1 hour') WHERE url = ?1",
                    params![entry.url],
                );
                Some(entry)
            }
            Err(_) => None,
        }
    }

    /// Record a completed crawl and update scheduling.
    pub fn record_crawl(
        &self,
        url: &str,
        content_hash: &str,
        new_priority: f64,
    ) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();

        // Update the frontier entry.
        conn.execute(
            "UPDATE frontier SET
                last_crawled = datetime('now'),
                crawl_count = crawl_count + 1,
                priority = ?2
             WHERE url = ?1",
            params![url, new_priority],
        )?;

        // Record in crawl history (keep last 5).
        conn.execute(
            "INSERT INTO crawl_history (url, crawled_at, content_hash)
             VALUES (?1, datetime('now'), ?2)",
            params![url, content_hash],
        )?;

        // Prune old history entries (keep last 5).
        conn.execute(
            "DELETE FROM crawl_history WHERE url = ?1 AND rowid NOT IN (
                SELECT rowid FROM crawl_history WHERE url = ?1
                ORDER BY crawled_at DESC LIMIT 5
            )",
            params![url],
        )?;

        Ok(())
    }

    /// Check SimHash dedup table. Returns the canonical URL if near-duplicate exists.
    pub fn check_simhash_dedup(&self, simhash: u64) -> Option<String> {
        let conn = self.conn.lock().unwrap();

        // For exact match first (distance 0).
        if let Ok(canonical) = conn.query_row(
            "SELECT canonical_url FROM simhash_dedup WHERE simhash = ?1",
            params![simhash as i64],
            |row| row.get::<_, String>(0),
        ) {
            return Some(canonical);
        }

        // For near-duplicates within Hamming distance 3, we'd need to check
        // all entries — for now, only exact match is efficient in SQLite.
        // A more sophisticated approach would use bit-manipulation queries
        // or an in-memory lookup table.
        None
    }

    /// Insert into SimHash dedup table.
    pub fn insert_simhash(&self, simhash: u64, canonical_url: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO simhash_dedup (simhash, canonical_url) VALUES (?1, ?2)",
            params![simhash as i64, canonical_url],
        )?;
        Ok(())
    }

    /// Get the total number of URLs in the frontier.
    pub fn count(&self) -> u64 {
        let conn = self.conn.lock().unwrap();
        conn.query_row("SELECT COUNT(*) FROM frontier", [], |row| row.get(0))
            .unwrap_or(0)
    }
}
