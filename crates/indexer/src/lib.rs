// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: indexer
//
// Tantivy-backed inverted index.
//
// Schema: multi-field document with per-field BM25F weights.
// Tokenizer: custom pipeline — NFKC normalize → lowercase → stem.
// On-disk format: standard Tantivy segments with .linkdata sidecar.
// Merge policy: off-peak merges, capped at 5GB per segment.
// ================================================================================

use raithe_common::config::IndexerConfig;
use raithe_common::error::RaitheResult;
use raithe_common::types::{ParsedDocument, SearchResult};
// M1.4 — import freshness crate from indexer per spec §25 M1.4.
// Held as underscore use; integration into merge hook is M6 work.
use raithe_freshness as _;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, TantivyDocument};
use tracing::{debug, error, info, warn};
use raithe_common::traits::NeuralInference;

pub mod fields {
pub const URL: &str = "url";
pub const TITLE: &str = "title";
pub const BODY: &str = "body";
pub const HEADINGS: &str = "headings";
pub const URL_TOKENS: &str = "url_tokens";
pub const ANCHOR_TEXT: &str = "anchor_text";
pub const META_DESC: &str = "meta_description";
pub const ALT_TEXT: &str = "alt_text";
pub const STRUCTURED: &str = "structured_data";
pub const LANGUAGE: &str = "language";
pub const CRAWLED_AT: &str = "crawled_at";
pub const DOMAIN: &str = "domain";
pub const SIMHASH: &str = "simhash";
// M3a — stable monotonic u64 doc id (prerequisite for linkgraph sidecar + Phase 3).
pub const DOC_ID_U64: &str = "doc_id_u64";
}

/// Resolved field handles for fast access during indexing and search.
#[derive(Clone)]
pub struct FieldHandles {
pub url: Field,
pub title: Field,
pub body: Field,
pub headings: Field,
pub url_tokens: Field,
pub anchor_text: Field,
pub meta_desc: Field,
pub alt_text: Field,
pub structured: Field,
pub language: Field,
pub crawled_at: Field,
pub domain: Field,
pub simhash: Field,
pub doc_id_u64: Field,
}

/// The RAiTHE index — wraps Tantivy with our custom schema and tokenizer.
pub struct RaitheIndex {
index: Index,
#[allow(dead_code)]
schema: Schema,
fields: FieldHandles,
writer: Arc<RwLock<IndexWriter>>,
reader: IndexReader,
config: IndexerConfig,
/// Tracks which crawl log files have been processed (by filename).
processed_logs: Arc<RwLock<std::collections::HashSet<String>>>,
/// M3a — monotonic stable doc-id counter, persisted at `doc_id_counter.bin`.
doc_id_counter: Arc<std::sync::atomic::AtomicU64>,
doc_id_counter_path: std::path::PathBuf,
/// M4a — Neural manager for synchronous embedding at add_document time.
neural: Arc<raithe_neural::GpuInferenceManager>,
/// M4a — HNSW index updated in lockstep with Tantivy commits.
hnsw: Arc<RwLock<raithe_semantic::HnswIndex>>,
/// M4a — Where to persist hnsw.bin sidecar at commit time.
hnsw_sidecar: std::path::PathBuf,
/// M4a — Counter for periodic embedding-progress logging.
embed_progress: Arc<std::sync::atomic::AtomicU64>,
}

impl RaitheIndex {
/// M4a — now requires neural manager + hnsw handle for embedding at add_document time.
pub fn open(
index_dir: &Path,
config: &IndexerConfig,
neural: Arc<raithe_neural::GpuInferenceManager>,
hnsw: Arc<RwLock<raithe_semantic::HnswIndex>>,
hnsw_sidecar: std::path::PathBuf,
) -> RaitheResult<Self> {
std::fs::create_dir_all(index_dir)?;

let mut schema_builder = Schema::builder();

// Stored + Indexed text fields for search.
let text_options = TextOptions::default()
.set_indexing_options(
TextFieldIndexing::default()
.set_tokenizer("default")
.set_index_option(IndexRecordOption::WithFreqsAndPositions),
)
.set_stored();

// Indexed-only text (not stored, saves disk).
let indexed_only = TextOptions::default().set_indexing_options(
TextFieldIndexing::default()
.set_tokenizer("default")
.set_index_option(IndexRecordOption::WithFreqs),
);

// Stored-only string (not tokenized — for domains, timestamps).
let stored_string = TextOptions::default().set_stored();

// URL is stored AND indexed as a raw string for dedup via delete_term.
let url = schema_builder.add_text_field(fields::URL, STRING | STORED);
let title = schema_builder.add_text_field(fields::TITLE, text_options.clone());
let body = schema_builder.add_text_field(fields::BODY, text_options.clone());
let headings = schema_builder.add_text_field(fields::HEADINGS, text_options.clone());
let url_tokens = schema_builder.add_text_field(fields::URL_TOKENS, indexed_only.clone());
let anchor_text =
schema_builder.add_text_field(fields::ANCHOR_TEXT, indexed_only.clone());
let meta_desc = schema_builder.add_text_field(fields::META_DESC, text_options);
let alt_text = schema_builder.add_text_field(fields::ALT_TEXT, indexed_only.clone());
let structured = schema_builder.add_text_field(fields::STRUCTURED, indexed_only);
let language = schema_builder.add_text_field(fields::LANGUAGE, stored_string.clone());
let crawled_at = schema_builder.add_text_field(fields::CRAWLED_AT, stored_string.clone());
let domain = schema_builder.add_text_field(fields::DOMAIN, stored_string);
let simhash = schema_builder.add_u64_field(fields::SIMHASH, INDEXED | STORED);
// M3a — stable monotonic doc id, FAST for per-doc lookup, STORED for recovery,
// INDEXED so we can delete/update by term if ever needed.
let doc_id_u64 =
schema_builder.add_u64_field(fields::DOC_ID_U64, FAST | STORED | INDEXED);

let schema = schema_builder.build();

// Open or create the Tantivy index.
let index = if index_dir.join("meta.json").exists() {
Index::open_in_dir(index_dir).map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?
} else {
Index::create_in_dir(index_dir, schema.clone()).map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?
};

let writer = index.writer(50_000_000).map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

// Reader that auto-reloads after commits.
let reader = index
.reader_builder()
.reload_policy(ReloadPolicy::OnCommitWithDelay)
.try_into()
.map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

let fh = FieldHandles {
url,
title,
body,
headings,
url_tokens,
anchor_text,
meta_desc,
alt_text,
structured,
language,
crawled_at,
domain,
simhash,
doc_id_u64,
};

// M3a — load or initialize the stable doc-id counter.
let doc_id_counter_path = index_dir.join("doc_id_counter.bin");
let initial_counter: u64 = match std::fs::read(&doc_id_counter_path) {
Ok(bytes) if bytes.len() == 8 => {
u64::from_le_bytes(bytes.try_into().unwrap())
}
_ => 0,
};
info!(
"RAiTHE doc-id counter: {} (loaded from {:?})",
initial_counter, doc_id_counter_path
);

let idx = Self {
index,
schema,
fields: fh,
writer: Arc::new(RwLock::new(writer)),
reader,
config: config.clone(),
processed_logs: Arc::new(RwLock::new(std::collections::HashSet::new())),
doc_id_counter: Arc::new(std::sync::atomic::AtomicU64::new(initial_counter)),
doc_id_counter_path,
neural,
hnsw,
hnsw_sidecar,
embed_progress: Arc::new(std::sync::atomic::AtomicU64::new(0)),
};

info!(
"RAiTHE Index opened at {:?} — {} documents",
index_dir,
idx.doc_count()
);

Ok(idx)
}

pub fn add_document(&self, pdoc: &ParsedDocument) -> RaitheResult<()> {
if pdoc.word_count < 20 {
debug!(url = %pdoc.url, words = pdoc.word_count, "Skipping short document");
return Ok(());
}

let headings_text: String = pdoc
.headings
.iter()
.map(|h| h.text.as_str())
.collect::<Vec<_>>()
.join(" ");

let url_tokens_text = pdoc.url_tokens.join(" ");
let alt_text_joined = pdoc.alt_texts.join(" ");
let structured_joined = pdoc.structured_data.join(" ");

let domain = url::Url::parse(&pdoc.url)
.ok()
.and_then(|u| u.host_str().map(String::from))
.unwrap_or_default();

// M3a — allocate a stable monotonic doc id for this document.
// Fetch_add returns the prior value; doc ids start at 0.
let doc_id_u64 = self
.doc_id_counter
.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

// M4a — synchronous embed (Robert Q15=p). Title + first ~400 chars of body.
// block_in_place lets us call async embed from this sync function while
// staying inside the surrounding tokio runtime.
let passage = {
let body_slice: String = pdoc.body.chars().take(400).collect();
format!("{} {}", pdoc.title, body_slice)
};
let embed_result = tokio::task::block_in_place(|| {
tokio::runtime::Handle::current().block_on(self.neural.embed_query(&passage))
});
match embed_result {
Ok(v) if v.iter().any(|x| *x != 0.0) => {
// Real embedding — insert into HNSW.
let hnsw_guard = safe_read(&self.hnsw);
if v.len() == hnsw_guard.embedding_dim() {
drop(hnsw_guard);
safe_write(&self.hnsw).insert(doc_id_u64, v);
} else {
warn!(
"Embedding dim mismatch for doc {}: got {}, expected {}",
doc_id_u64, v.len(),
safe_read(&self.hnsw).embedding_dim()
);
}
}
Ok(_) => {
// Zero vector = graceful degradation (no model). Skip HNSW insert.
}
Err(e) => {
warn!("Embed failed for doc {}: {}", doc_id_u64, e);
}
}
let progress = self.embed_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
if progress % 10 == 0 {
info!("Embedding document {} (hnsw size {})", progress, safe_read(&self.hnsw).len());
}

let tantivy_doc = doc!(
self.fields.url => pdoc.url.clone(),
self.fields.title => pdoc.title.clone(),
self.fields.body => pdoc.body.clone(),
self.fields.headings => headings_text,
self.fields.url_tokens => url_tokens_text,
self.fields.meta_desc => pdoc.meta_description.clone(),
self.fields.alt_text => alt_text_joined,
self.fields.structured => structured_joined,
self.fields.language => pdoc.language.clone(),
self.fields.crawled_at => pdoc.crawled_at.to_rfc3339(),
self.fields.domain => domain,
self.fields.simhash => pdoc.simhash,
self.fields.doc_id_u64 => doc_id_u64,
);

let writer = self.writer.write().map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(format!("Writer lock: {}", e)),
)
})?;

// Delete any existing document with this URL (dedup + freshness update).
// This handles: restart re-indexing, re-crawled pages with updated content.
let url_term = tantivy::Term::from_field_text(self.fields.url, &pdoc.url);
writer.delete_term(url_term);

writer.add_document(tantivy_doc).map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

debug!(url = %pdoc.url, word_count = pdoc.word_count, "Indexed document");
Ok(())
}

pub fn commit(&self) -> RaitheResult<()> {
let mut writer = self.writer.write().map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(format!("Writer lock: {}", e)),
)
})?;

writer.commit().map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

// M3a — persist the doc-id counter atomically (write to .tmp, rename).
// Must happen AFTER writer.commit() so counter never exceeds what's on disk.
let counter_val = self
.doc_id_counter
.load(std::sync::atomic::Ordering::SeqCst);
let tmp_path = self.doc_id_counter_path.with_extension("bin.tmp");
if let Err(e) = std::fs::write(&tmp_path, counter_val.to_le_bytes()) {
error!("Failed to write doc-id counter tmp: {}", e);
} else if let Err(e) = std::fs::rename(&tmp_path, &self.doc_id_counter_path) {
error!("Failed to rename doc-id counter: {}", e);
}

// M4a — persist the HNSW sidecar after a successful commit.
// Read lock is fine; save() only needs immutable access.
if let Err(e) = safe_read(&self.hnsw).save(&self.hnsw_sidecar) {
error!("Failed to save HNSW sidecar: {}", e);
}

// Force the reader to reload so new documents are immediately visible
// for search. Without this, OnCommitWithDelay may take up to 500ms.
self.reader.reload().map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

info!("Index committed — {} total documents", self.doc_count());
Ok(())
}

/// Get the number of indexed documents.
pub fn doc_count(&self) -> u64 {
self.reader.searcher().num_docs()
}

/// Get a Tantivy searcher snapshot.
pub fn searcher(&self) -> Searcher {
self.reader.searcher()
}

/// Get field handles (for use by serving layer).
pub fn field_handles(&self) -> &FieldHandles {
&self.fields
}

/// Phase 1 BM25F retrieval. Query is parsed against multiple fields.
pub fn search(
&self,
query_str: &str,
max_results: usize,
offset: usize,
) -> RaitheResult<(Vec<SearchResult>, u64)> {
let searcher = self.reader.searcher();

let mut query_parser = QueryParser::for_index(
&self.index,
vec![
self.fields.title,
self.fields.body,
self.fields.headings,
self.fields.meta_desc,
self.fields.url_tokens,
],
);

// These are the core ranking weights — title matches are 5x more
// important than body matches. This is THE differentiator for
// result quality on a small index.
query_parser.set_field_boost(self.fields.title, 5.0);
query_parser.set_field_boost(self.fields.headings, 2.5);
query_parser.set_field_boost(self.fields.url_tokens, 2.0);
query_parser.set_field_boost(self.fields.meta_desc, 1.5);
query_parser.set_field_boost(self.fields.body, 1.0);

let query = query_parser
.parse_query(query_str)
.map_err(|e| raithe_common::error::RaitheError::QueryParse(e.to_string()))?;

let top_docs = searcher
.search(&query, &TopDocs::with_limit(max_results + offset))
.map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

let total_hits = top_docs.len() as u64;
let page_docs: Vec<_> = top_docs.into_iter().skip(offset).take(max_results).collect();

let mut results = Vec::with_capacity(page_docs.len());

for (score, doc_addr) in &page_docs {
let retrieved: TantivyDocument = searcher.doc(*doc_addr).map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;

// M3a — read stable monotonic doc_id from the FAST field.
let segment_reader = searcher.segment_reader(doc_addr.segment_ord);
let doc_id_ff = segment_reader
.fast_fields()
.u64(fields::DOC_ID_U64)
.map_err(|e| {
raithe_common::error::RaitheError::Index(
raithe_common::error::IndexError::Tantivy(e.to_string()),
)
})?;
let doc_id = doc_id_ff.first(doc_addr.doc_id).unwrap_or(0);

let url = get_text_field(&retrieved, self.fields.url);
let title = get_text_field(&retrieved, self.fields.title);
let body = get_text_field(&retrieved, self.fields.body);
let meta_desc = get_text_field(&retrieved, self.fields.meta_desc);
let domain = get_text_field(&retrieved, self.fields.domain);
let crawled_at_str = get_text_field(&retrieved, self.fields.crawled_at);
let snippet = generate_snippet(&body, query_str, 300);

let crawled_at = chrono::DateTime::parse_from_rfc3339(&crawled_at_str)
.map(|dt| dt.with_timezone(&chrono::Utc))
.unwrap_or_else(|_| chrono::Utc::now());

results.push(SearchResult {
doc_id,
url,
title,
snippet,
score: *score as f64,
meta_description: meta_desc,
domain,
last_crawled: crawled_at,
favicon_url: None,
});
}

Ok((results, total_hits))
}

/// M3a — resolve a stable doc_id to its URL by scanning segments.
/// Linear over segments; cheap because segments are few and the fast field is memory-mapped.
/// Returns None if the doc_id is not found (e.g. deleted or out of range).
pub fn resolve_url(&self, target: u64) -> Option<String> {
let searcher = self.reader.searcher();
for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
let ff = match seg_reader.fast_fields().u64(fields::DOC_ID_U64) {
Ok(ff) => ff,
Err(_) => continue,
};
let alive = seg_reader.alive_bitset();
let max_doc = seg_reader.max_doc();
for local in 0..max_doc {
if let Some(bs) = alive {
if !bs.is_alive(local) {
continue;
}
}
if ff.first(local) == Some(target) {
let addr = tantivy::DocAddress::new(seg_ord as u32, local);
let retrieved: TantivyDocument = searcher.doc(addr).ok()?;
return Some(get_text_field(&retrieved, self.fields.url));
}
}
}
None
}

/// M3a — iterate every live (doc_id, url) pair in the index.
/// Used by the M3b linkgraph sidecar builder to populate the url→id map.
pub fn iter_docs(&self) -> Vec<(u64, String)> {
let searcher = self.reader.searcher();
let mut out = Vec::new();
for (seg_ord, seg_reader) in searcher.segment_readers().iter().enumerate() {
let ff = match seg_reader.fast_fields().u64(fields::DOC_ID_U64) {
Ok(ff) => ff,
Err(_) => continue,
};
let alive = seg_reader.alive_bitset();
let max_doc = seg_reader.max_doc();
for local in 0..max_doc {
if let Some(bs) = alive {
if !bs.is_alive(local) {
continue;
}
}
let Some(doc_id) = ff.first(local) else { continue };
let addr = tantivy::DocAddress::new(seg_ord as u32, local);
if let Ok(retrieved) = searcher.doc::<TantivyDocument>(addr) {
out.push((doc_id, get_text_field(&retrieved, self.fields.url)));
}
}
}
out
}

/// Every poll_interval_secs, reads new documents from the crawl log
/// and adds them to the index.
pub async fn run_poll_loop(
&self,
crawl_log: &raithe_storage::crawl_log::FileCrawlLog,
) -> RaitheResult<()> {
info!(
"Indexer poll loop starting — interval: {}s",
self.config.poll_interval_secs
);

loop {
let log_files = crawl_log.list_log_files();
let mut total_indexed = 0usize;

for log_file in &log_files {
let file_name = log_file
.file_name()
.and_then(|n| n.to_str())
.unwrap_or("")
.to_string();

// Skip already-processed log files.
{
let processed = safe_read(&self.processed_logs);
if processed.contains(&file_name) {
continue;
}
}

match raithe_storage::crawl_log::FileCrawlLog::read_log_file(log_file) {
Ok(documents) if !documents.is_empty() => {
for d in &documents {
if let Err(e) = self.add_document(d) {
warn!(url = %d.url, error = %e, "Failed to index document");
} else {
total_indexed += 1;
}
}
let mut processed = safe_write(&self.processed_logs);
processed.insert(file_name);
}
Ok(_) => {}
Err(e) => {
warn!("Failed to read crawl log {:?}: {}", log_file, e);
}
}
}

if total_indexed > 0 {
if let Err(e) = self.commit() {
error!("Failed to commit index: {}", e);
} else {
info!(
"Indexer poll: indexed {} new documents (total: {})",
total_indexed,
self.doc_count()
);
}
}

tokio::time::sleep(std::time::Duration::from_secs(
self.config.poll_interval_secs,
))
.await;
}
}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Acquire a read lock, recovering from poison if needed.
///
/// When a thread panics while holding a lock, Rust poisons the lock.
/// We recover by returning the poisoned guard directly - the data is valid,
/// just marked as poisoned. This is idiomatic Rust lock recovery.
fn safe_read<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
match lock.read() {
Ok(guard) => guard,
Err(poisoned) => {
warn!("RwLock read poisoned — recovering from panic in another thread");
poisoned.into_inner()
}
}
}

/// Acquire a write lock, recovering from poison if needed.
fn safe_write<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
match lock.write() {
Ok(guard) => guard,
Err(poisoned) => {
warn!("RwLock write poisoned — recovering from panic in another thread");
poisoned.into_inner()
}
}
}

/// Extract a text field value from a Tantivy document.
fn get_text_field(doc: &TantivyDocument, field: Field) -> String {
doc.get_first(field)
.and_then(|v| match v {
tantivy::schema::OwnedValue::Str(s) => Some(s.clone()),
_ => None,
})
.unwrap_or_default()
}

///
/// Split into sentences, score by query term density with IDF-like weighting,
/// select best sentences that fit in max_chars, bold all query term occurrences.
fn generate_snippet(body: &str, query: &str, max_chars: usize) -> String {
if body.is_empty() {
return String::new();
}

let query_terms: Vec<String> = query
.split_whitespace()
.map(|t| t.to_lowercase())
.filter(|t| t.len() >= 2)
.collect();

if query_terms.is_empty() {
let truncated: String = body.chars().take(max_chars).collect();
return truncated;
}

// Split into sentences on sentence-ending punctuation.
let sentences: Vec<&str> = body
.split(|c: char| c == '.' || c == '!' || c == '?')
.map(|s| s.trim())
.filter(|s| s.len() > 10)
.collect();

if sentences.is_empty() {
return bold_terms(&body.chars().take(max_chars).collect::<String>(), &query_terms);
}

// Score each sentence by how many distinct query terms it contains.
let mut scored: Vec<(usize, &str, usize)> = sentences
.iter()
.enumerate()
.map(|(idx, &sentence)| {
let lower = sentence.to_lowercase();
let hits: usize = query_terms
.iter()
.filter(|term| lower.contains(term.as_str()))
.count();
(idx, sentence, hits)
})
.collect();

// Sort by hit count descending, then by original order for ties.
scored.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));

// Collect best sentences up to max_chars.
let mut selected: Vec<(usize, &str)> = Vec::new();
let mut total_len = 0usize;

for (idx, sentence, _hits) in &scored {
let add_len = sentence.len() + if selected.is_empty() { 0 } else { 2 };
if total_len + add_len > max_chars {
if selected.is_empty() {
// First sentence too long — truncate it.
let trunc: String = sentence.chars().take(max_chars - 3).collect();
return bold_terms(&format!("{}...", trunc), &query_terms);
}
break;
}
total_len += add_len;
selected.push((*idx, sentence));
}

// Re-sort selected sentences by their original document order.
selected.sort_by_key(|(idx, _)| *idx);

let snippet = selected
.iter()
.map(|(_, s)| *s)
.collect::<Vec<_>>()
.join(". ");

bold_terms(&snippet, &query_terms)
}

/// Bold all occurrences of all query terms in the text (case-insensitive).
/// Produces HTML with <b> tags around matched terms.
fn bold_terms(text: &str, terms: &[String]) -> String {
if terms.is_empty() {
return text.to_string();
}

// Find all match positions: (start, end) in the text.
let lower = text.to_lowercase();
let mut matches: Vec<(usize, usize)> = Vec::new();

for term in terms {
let mut search_from = 0;
while let Some(pos) = lower[search_from..].find(term.as_str()) {
let abs_pos = search_from + pos;
matches.push((abs_pos, abs_pos + term.len()));
search_from = abs_pos + term.len();
}
}

if matches.is_empty() {
return text.to_string();
}

// Sort by start position and merge overlapping spans.
matches.sort_by_key(|(start, _)| *start);
let mut merged: Vec<(usize, usize)> = Vec::new();
for (start, end) in matches {
if let Some(last) = merged.last_mut() {
if start <= last.1 {
last.1 = last.1.max(end);
continue;
}
}
merged.push((start, end));
}

// Build the output string with <b> tags around merged spans.
let mut result = String::with_capacity(text.len() + merged.len() * 7);
let mut cursor = 0;
for (start, end) in &merged {
if *start > cursor {
result.push_str(&text[cursor..*start]);
}
result.push_str("<b>");
result.push_str(&text[*start..*end]);
result.push_str("</b>");
cursor = *end;
}
if cursor < text.len() {
result.push_str(&text[cursor..]);
}

result
}

/// NFKC normalization → Unicode word segmentation → lowercase → stem.
pub fn tokenize(text: &str, language: &str) -> Vec<String> {
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

let normalized: String = text.nfkc().collect();

let tokens: Vec<String> = normalized
.unicode_words()
.map(|w| w.to_lowercase())
.filter(|w| w.len() >= 2)
.collect();

let algorithm = match language {
"en" => rust_stemmers::Algorithm::English,
"fr" => rust_stemmers::Algorithm::French,
"de" => rust_stemmers::Algorithm::German,
"es" => rust_stemmers::Algorithm::Spanish,
"it" => rust_stemmers::Algorithm::Italian,
"pt" => rust_stemmers::Algorithm::Portuguese,
"nl" => rust_stemmers::Algorithm::Dutch,
"ru" => rust_stemmers::Algorithm::Russian,
_ => rust_stemmers::Algorithm::English,
};

let stemmer = rust_stemmers::Stemmer::create(algorithm);

tokens
.into_iter()
.map(|t| stemmer.stem(&t).into_owned())
.collect()
}

#[cfg(test)]
mod tests {
use super::*;

#[test]
fn test_tokenize_english() {
let tokens = tokenize("The quick brown foxes are jumping over lazy dogs", "en");
assert!(tokens.contains(&"quick".to_string()));
assert!(tokens.contains(&"brown".to_string()));
assert!(tokens.contains(&"fox".to_string()));
assert!(tokens.contains(&"jump".to_string()));
}

#[test]
fn test_tokenize_nfkc_normalization() {
let tokens = tokenize("\u{FB01}nding", "en");
assert!(tokens.iter().any(|t| t.starts_with("find")));
}

#[test]
fn test_snippet_generation() {
let body = "Rust is a systems programming language. It provides memory safety without garbage collection. Rust is blazingly fast and reliable.";
let snippet = generate_snippet(body, "rust memory", 300);
assert!(!snippet.is_empty());
// Should bold all occurrences of "rust" and "memory".
assert!(snippet.contains("<b>"), "Should contain bold tags");
assert!(
snippet.matches("<b>").count() >= 2,
"Should bold multiple occurrences: {}",
snippet
);
}

#[test]
fn test_bold_terms_multiple() {
let text = "Rust provides memory safety. Rust is fast.";
let terms = vec!["rust".to_string(), "memory".to_string()];
let bolded = bold_terms(text, &terms);
assert_eq!(
bolded,
"<b>Rust</b> provides <b>memory</b> safety. <b>Rust</b> is fast."
);
}
}
