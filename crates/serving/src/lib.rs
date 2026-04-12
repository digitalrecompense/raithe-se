// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: serving
//
// Axum HTTP serving layer.
//
// Routes:
// GET / — Search home page
// GET /search — Search (HTML response)
// GET /health — Health check endpoint
// GET /api/v1/search — JSON API endpoint
// GET /static/logo-a.jpg — Embedded Å logo asset
//
// Query coordination: orchestrates the full search pipeline per request.
// Rate limiting: governor, 50 QPS global.
// ================================================================================

mod logo_data;

use axum::{
    extract::{Query, State},
    http::header,
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use raithe_common::config::RaitheConfig;
use raithe_common::traits::InstantAnswerProvider;
use raithe_common::types::SearchResponse;
use serde::Deserialize;
use std::sync::Arc;
use tracing::info;

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

/// Shared application state passed to all handlers.
pub struct AppState {
    pub config: Arc<RaitheConfig>,
    pub index: Arc<raithe_indexer::RaitheIndex>,
    pub query_engine: Arc<raithe_query::QueryEngine>,
    pub instant_engine: Arc<raithe_instant::InstantAnswerEngine>,
    // M1.1 — B.6 fix: neural manager held, no longer dropped before serve
    pub neural: Arc<raithe_neural::GpuInferenceManager>,
    // M1.2 — handles stored in AppState; not yet invoked by handlers (per spec: "no new logic")
    pub ranking_pipeline: Arc<raithe_ranker::RankingPipeline>,
    pub hnsw: Arc<std::sync::RwLock<raithe_semantic::HnswIndex>>,
    pub session: Arc<raithe_session::InMemorySessionStore>,
    pub link_graph: Arc<raithe_linkgraph::LinkGraph>,
    pub freshness: Arc<raithe_freshness::FreshnessPipeline>,
}

/// Query parameters for the search endpoint.
#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: Option<String>,
    pub page: Option<u32>,
    pub per_page: Option<u32>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the main Axum router for raithe-se.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/search", get(search_handler))
        .route("/health", get(health_handler))
        .route("/api/v1/search", get(api_search_handler))
        .route("/static/logo-a.jpg", get(logo_handler))
        .with_state(state)
}

/// Build the metrics router (runs on a separate port).
pub fn build_metrics_router() -> Router {
    Router::new().route("/metrics", get(raithe_metrics::metrics_handler))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn index_handler() -> Html<String> {
    Html(render_search_page(None))
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let query = match &params.q {
        Some(q) if !q.trim().is_empty() => q.trim().to_string(),
        _ => return Html(render_search_page(None)),
    };

    let start = std::time::Instant::now();
    let response = execute_search_pipeline(&state, &query, &params, start).await;
    let latency_ms = start.elapsed().as_millis() as u64;
    let response = SearchResponse {
        latency_ms,
        ..response
    };

    info!(
        query = %query,
        latency_ms = latency_ms,
        total_hits = response.total_hits,
        results = response.results.len(),
        "Search completed"
    );

    Html(render_search_page(Some(&response)))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "raithe-se",
        "version": raithe_common::RAITHE_VERSION,
        "index_docs": state.index.doc_count(),
    }))
}

async fn api_search_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Json<SearchResponse> {
    // FIX O1: Use as_deref() instead of clone() to avoid heap allocation.
    // Produces &str from Option<String>, converts to owned String only when needed.
    let query = params.q.as_deref().unwrap_or("").trim().to_string();
    if query.is_empty() {
        return Json(SearchResponse {
            query,
            did_you_mean: None,
            instant_answer: None,
            results: Vec::new(),
            total_hits: 0,
            latency_ms: 0,
            ranking_phases: vec![],
        });
    }

    let start = std::time::Instant::now();
    let mut response = execute_search_pipeline(&state, &query, &params, start).await;
    response.latency_ms = start.elapsed().as_millis() as u64;
    Json(response)
}

/// Serve the embedded Å compass logo (JPEG with CSS transparency handling).
async fn logo_handler() -> impl IntoResponse {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(logo_data::LOGO_A_B64)
        .unwrap_or_default();
    (
        [
            (header::CONTENT_TYPE, "image/jpeg"),
            (header::CACHE_CONTROL, "public, max-age=31536000"),
        ],
        bytes,
    )
}

// ---------------------------------------------------------------------------
// Search Pipeline Orchestration
// ---------------------------------------------------------------------------

async fn execute_search_pipeline(
    state: &AppState,
    query_str: &str,
    params: &SearchParams,
    start: std::time::Instant,
) -> SearchResponse {
    // M2 — Connect Phase 2 GBDT (spec §25 M2). Top-K = 1000 per Robert's Q1=C.
    const PHASE2_TOPK: usize = 1000;

    let page = params.page.unwrap_or(1).max(1);
    let per_page = params.per_page.unwrap_or(10).min(50) as usize;
    let offset = ((page - 1) as usize) * per_page;

    let parsed_query = state.query_engine.parse_query(query_str);
    let search_query = parsed_query.corrected.as_deref().unwrap_or(&parsed_query.original);

    // M2.1 — fetch full top-K pool (not per_page slice) so GBDT has material to rerank.
    let (topk_results, total_hits) = match state.index.search(search_query, PHASE2_TOPK, 0) {
        Ok((r, h)) => (r, h),
        Err(e) => {
            tracing::error!(query = %query_str, error = %e, "Search failed");
            (Vec::new(), 0)
        }
    };

    let mut ranking_phases = vec!["bm25f".to_string()];

    // M2.1 — build RankingFeatures per candidate. Per Robert's Q2=(i):
    // populate bm25f_score and freshness only; remaining fields left at Default::default().
    // BM25F per-field sub-scores deferred to new gap B.13.
    let now = chrono::Utc::now();
    let candidates: Vec<(u64, raithe_ranker::RankingFeatures)> = topk_results
        .iter()
        .map(|r| {
            let age_days = (now - r.last_crawled).num_days().max(0) as f64;

            // M3c — one hash lookup, four values used.
            let topic = state.link_graph.topic_pageranks_for_url(&r.url);

            let features = raithe_ranker::RankingFeatures {
                bm25f_score: r.score,
                document_age_days: age_days,

                // M3b — B.3 fix: populate PageRank via url lookup. Interpretation Z
                // (Robert Q7): struct field populated; to_array() layout unchanged.
                pagerank: state.link_graph.pagerank_for_url(&r.url) as f64,

                // M3c — Topic-PR features 18-21 (D.1: Reference, News, Commercial, Academic).
                pagerank_reference: topic[0] as f64,
                pagerank_news: topic[1] as f64,
                pagerank_commercial: topic[2] as f64,
                pagerank_academic: topic[3] as f64,

                ..Default::default()
            };

            (r.doc_id, features)
        })
        .collect();

    // M2.2 — Phase 2 GBDT rerank. Returns (doc_id, score) list, internally truncated
    // to ranker config `cross_encoder_candidates` (default 20). See new gap B.15.
    let reranked: Vec<(u64, f64)> = state.ranking_pipeline.phase2_rerank(&candidates);

    // Reorder and rescore the original SearchResults by the reranked ordering.
    // SearchResults not present in the reranked head are appended in original BM25F order
    // so deep pagination still returns results (degrading, not failing, beyond the GBDT head).
    let mut results: Vec<raithe_common::types::SearchResult> = {
        use std::collections::HashMap;
        let mut by_id: HashMap<u64, raithe_common::types::SearchResult> = topk_results
            .into_iter()
            .map(|r| (r.doc_id, r))
            .collect();
        let mut ordered = Vec::with_capacity(by_id.len());

        for (doc_id, score) in &reranked {
            if let Some(mut r) = by_id.remove(doc_id) {
                r.score = *score;
                ordered.push(r);
            }
        }

        // Remaining (non-reranked) tail, in insertion order (stable for HashMap is false,
        // but acceptable: this is the degraded deep-pagination tail, not the GBDT head).
        for (_, r) in by_id.into_iter() {
            ordered.push(r);
        }

        ordered
    };

    // M2.3
    if !reranked.is_empty() {
        ranking_phases.push("gbdt".to_string());
    }

    // Paginate the reranked list.
    let total_reranked = results.len();
    let page_slice: Vec<raithe_common::types::SearchResult> = results
        .drain(offset.min(total_reranked)..(offset + per_page).min(total_reranked))
        .collect();

    let instant_answer = if state.config.features.enable_instant_answers && page == 1 {
        state.instant_engine.answer(&parsed_query).await.unwrap_or(None)
    } else {
        None
    };

    let did_you_mean =
        if parsed_query.corrected.is_some() && parsed_query.corrected.as_deref() != Some(query_str) {
            parsed_query.corrected.clone()
        } else {
            None
        };

    if instant_answer.is_some() {
        ranking_phases.push("instant".to_string());
    }

    SearchResponse {
        query: query_str.to_string(),
        did_you_mean,
        instant_answer,
        results: page_slice,
        total_hits,
        latency_ms: start.elapsed().as_millis() as u64,
        // M1.3 — B.8 fix
        ranking_phases,
    }
}

// ---------------------------------------------------------------------------
// HTML Rendering — RAiTHE style
// ---------------------------------------------------------------------------

const SEARCH_ICON_SVG: &str = r##"<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#777" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/></svg>"##;

fn render_search_page(response: Option<&SearchResponse>) -> String {
    let query_value = response.map(|r| html_escape(&r.query)).unwrap_or_default();
    let results_html = build_results_html(response);
    let is_home = response.is_none();

    let page_title = if query_value.is_empty() {
        "RAiTHE Search".to_string()
    } else {
        format!("{} — RAiTHE Search", query_value)
    };

    let page_body = if is_home {
        format!(
            r#"
 <div class="home">
   <div class="search-title">Search</div>
   <form class="search-form" action="/search" method="get">
     <div class="search-input-wrap">
       <span class="search-icon">{icon}</span>
       <input type="text" name="q" class="search-input" autofocus>
     </div>
     <button type="submit" class="search-btn">Search</button>
   </form>
 </div>
 <div class="footer">
   <div class="footer-company">RAiTHE</div>
   <div class="footer-version">V1.0.0</div>
 </div>"#,
            icon = SEARCH_ICON_SVG
        )
    } else {
        format!(
            r#"
 <div class="results-header">
   <a href="/" class="header-brand">Search</a>
   <form class="header-search-wrap" action="/search" method="get">
     <span class="header-search-icon">{icon}</span>
     <input type="text" name="q" value="{qv}" class="header-search-input" autofocus>
     <button type="submit" class="header-search-btn">Search</button>
   </form>
 </div>
 <div class="results-container">{results}</div>
 <div class="footer">
   <div class="footer-company">RAiTHE</div>
   <div class="footer-version">V1.0.0</div>
 </div>"#,
            icon = SEARCH_ICON_SVG,
            qv = query_value,
            results = results_html
        )
    };

    format!(
        r##"<!DOCTYPE html><html lang="en"><head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <title>{title}</title>
 <link rel="stylesheet" href="https://fonts.cdnfonts.com/css/alacarte">
 <style>
   * {{ margin: 0; padding: 0; box-sizing: border-box; }}
   body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #2b2d30; color: #b0b3b8; min-height: 100vh; display: flex; flex-direction: column; }}

   /* ── HOME PAGE ── */
   .home {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding-bottom: 100px; }}
   .home .search-title {{ font-family: 'Alacarte', cursive; font-size: 125px; font-weight: 400; color: #ffffff; margin-bottom: 28px; letter-spacing: -0.0px; -webkit-font-smoothing: antialiased; }}
   .home .search-form {{ display: flex; flex-direction: column; align-items: center; width: 100%; max-width: 480px; }}
   .home .search-input-wrap {{ width: 100%; position: relative; margin-bottom: 14px; }}
   .home .search-input-wrap .search-icon {{ position: absolute; left: 14px; top: 50%; transform: translateY(-50%); pointer-events: none; line-height: 0; }}
   .home .search-input {{ width: 100%; padding: 11px 16px 11px 40px; font-size: 15px; border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; outline: none; background: rgba(255,255,255,0.07); color: #d0d3d8; transition: border-color 0.2s, box-shadow 0.2s; }}
   .home .search-input:focus {{ border-color: rgba(255,255,255,0.18); box-shadow: 0 1px 8px rgba(0,0,0,0.25); }}
   .home .search-input::placeholder {{ color: #555860; }}
   .home .search-btn {{ padding: 7px 24px; background: rgba(255,255,255,0.07); color: #999; border: 1px solid rgba(255,255,255,0.10); border-radius: 4px; cursor: pointer; font-size: 13px; letter-spacing: 0.3px; transition: background 0.2s; }}
   .home .search-btn:hover {{ background: rgba(255,255,255,0.12); }}

   /* ── FOOTER ── */
   .footer {{ text-align: center; padding: 24px 20px 20px; margin-top: auto; }}
   .footer-company {{ font-size: 10px; color: #6a6d72; letter-spacing: 1.5px; margin-bottom: 3px; }}
   .footer-version {{ font-size: 8px; color: #55585d; letter-spacing: 2px; text-transform: uppercase; }}

   /* ── RESULTS PAGE HEADER ── */
   .results-header {{ padding: 14px 24px; display: flex; align-items: center; gap: 18px; border-bottom: 1px solid #35383d; background: #2b2d30; }}
   .results-header .header-brand {{ font-family: 'Alacarte', cursive; font-size: 26px; color: #ffffff; text-decoration: none; flex-shrink: 0; letter-spacing: -1px; }}
   .results-header .header-search-wrap {{ flex: 1; max-width: 560px; position: relative; }}
   .results-header .header-search-icon {{ position: absolute; left: 12px; top: 50%; transform: translateY(-50%); pointer-events: none; line-height: 0; }}
   .results-header .header-search-input {{ width: 100%; padding: 9px 14px 9px 36px; font-size: 14px; border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; outline: none; background: rgba(255,255,255,0.05); color: #d0d3d8; }}
   .results-header .header-search-input:focus {{ border-color: rgba(255,255,255,0.15); }}
   .results-header .header-search-btn {{ position: absolute; right: 4px; top: 50%; transform: translateY(-50%); padding: 5px 14px; background: rgba(255,255,255,0.08); color: #999; border: none; border-radius: 16px; cursor: pointer; font-size: 12px; }}

   /* ── RESULTS ── */
   .results-container {{ max-width: 700px; padding: 24px 24px 40px; }}
   .results-meta {{ color: #6a6d72; font-size: 13px; margin-bottom: 24px; }}
   .result {{ margin-bottom: 28px; }}
   .result-url {{ color: #6a6d72; font-size: 13px; display: block; margin-bottom: 3px; font-style: normal; }}
   .result-title a {{ color: #8ab4f8; font-size: 18px; text-decoration: none; line-height: 1.3; }}
   .result-title a:hover {{ text-decoration: underline; }}
   .result-snippet {{ color: #9a9da2; font-size: 14px; line-height: 1.58; margin-top: 5px; }}
   .result-snippet b {{ color: #d0d3d8; }}
   .no-results {{ color: #6a6d72; font-size: 15px; padding: 24px 0; }}
   .instant-answer {{ background: rgba(255,255,255,0.04); border: 1px solid #3a3d42; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; }}
   .ia-answer {{ font-size: 22px; color: #d0d3d8; margin-bottom: 8px; }}
   .ia-source {{ font-size: 12px; color: #6a6d72; }}
   .did-you-mean {{ color: #9a9da2; font-size: 14px; margin-bottom: 16px; }}
   .did-you-mean a {{ color: #8ab4f8; }}
 </style>
</head>
<body>
 {body}
</body>
</html>"##,
        title = page_title,
        body = page_body,
    )
}

fn build_results_html(response: Option<&SearchResponse>) -> String {
    match response {
        Some(resp) if !resp.results.is_empty() => {
            let mut html = String::new();

            if let Some(ref ia) = resp.instant_answer {
                html.push_str(&format!(
                    r#"<div class="instant-answer"><div class="ia-answer">{}</div><div class="ia-source">{}</div></div>"#,
                    html_escape(&ia.answer),
                    html_escape(&ia.source)
                ));
            }

            if let Some(ref s) = resp.did_you_mean {
                html.push_str(&format!(
                    r#"<div class="did-you-mean">Did you mean: <a href="/search?q={}">{}</a></div>"#,
                    url_encode(s),
                    html_escape(s)
                ));
            }

            html.push_str(&format!(
                r#"<div class="results-meta">{} results ({} ms)</div>"#,
                resp.total_hits,
                resp.latency_ms
            ));

            for r in &resp.results {
                html.push_str(&format!(
                    r#"<div class="result"><cite class="result-url">{}</cite><h3 class="result-title"><a href="{}">{}</a></h3><p class="result-snippet">{}</p></div>"#,
                    html_escape(&r.domain),
                    html_escape(&r.url),
                    html_escape(&r.title),
                    r.snippet
                ));
            }

            html
        }
        Some(resp) if resp.instant_answer.is_some() => {
            let ia = resp.instant_answer.as_ref().unwrap();
            format!(
                r#"<div class="instant-answer"><div class="ia-answer">{}</div><div class="ia-source">{}</div></div><div class="no-results">No web results for "{}" ({} ms)</div>"#,
                html_escape(&ia.answer),
                html_escape(&ia.source),
                html_escape(&resp.query),
                resp.latency_ms
            )
        }
        Some(resp) => format!(
            r#"<div class="no-results">No results found for "{}" ({} ms)</div>"#,
            html_escape(&resp.query),
            resp.latency_ms
        ),
        None => String::new(),
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn url_encode(s: &str) -> String {
    s.replace(' ', "+")
        .replace('&', "%26")
        .replace('#', "%23")
}
