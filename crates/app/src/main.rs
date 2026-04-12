// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: app
//
// Single-binary entry point.
//
// Launch sequence:
// 1. Print RAiTHE banner.
// 2. Load configuration.
// 3. Initialize logging and metrics.
// 4. Initialize storage directories.
// 5. Initialize GPU inference manager.
// 6. Open or create the search index.
// 7. Initialize query engine and instant answers.
// 8. Start crawl log emitter.
// 9. Start the crawler in the background.
// 10. Start the indexer poll loop.
// 11. Start the HTTP server.
// 12. Start the metrics server on a separate port.
// ================================================================================
use raithe_common::config::RaitheConfig;
use raithe_common::traits::NeuralInference;
use raithe_common::RAITHE_BANNER;
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Step 1: Print the RAiTHE banner.
    eprintln!("{}", RAITHE_BANNER);

    // Step 2: Load configuration.
    let config_path = std::env::args().nth(1).map(std::path::PathBuf::from);
    let config = RaitheConfig::load(config_path.as_deref())?;
    let config = Arc::new(config);

    // Step 3: Initialize logging.
    raithe_metrics::init_logging(&config.general.log_level);
    info!(
        "raithe-se v{} starting on {}",
        raithe_common::RAITHE_VERSION,
        config.general.hostname
    );

    // Step 4: Initialize metrics.
    raithe_metrics::RaitheMetrics::init();
    info!("Prometheus metrics registered");

    // Step 5: Initialize storage directories.
    raithe_storage::init_storage_dirs(&config.storage)?;
    info!("Storage directories initialized");

    // Step 6: Initialize GPU inference manager.
    // M1.1 — B.6 fix: wrap in Arc and retain; formerly bound to `_neural` and dropped.
    let neural = Arc::new(raithe_neural::GpuInferenceManager::new(&config.neural)?);
    info!(
        "GPU inference manager ready (GPU available: {})",
        neural.is_gpu_available()
    );

    // Step 7: Open the search index.
    // M4a — HNSW must be constructed BEFORE the index because RaitheIndex now
    // owns a handle to it for synchronous embedding at add_document time.
    let hnsw_sidecar = config.storage.vectors_dir.join("hnsw.bin");
    let hnsw_inner = if hnsw_sidecar.exists() {
        match raithe_semantic::HnswIndex::load(&hnsw_sidecar, &config.semantic) {
            Ok(h) => { info!("HNSW loaded from sidecar"); h }
            Err(e) => {
                tracing::warn!("HNSW sidecar load failed ({}) — starting empty", e);
                raithe_semantic::HnswIndex::new(&config.semantic)
            }
        }
    } else {
        std::fs::create_dir_all(&config.storage.vectors_dir).ok();
        raithe_semantic::HnswIndex::new(&config.semantic)
    };
    let hnsw = Arc::new(std::sync::RwLock::new(hnsw_inner));
    let index = Arc::new(raithe_indexer::RaitheIndex::open(
        &config.storage.index_dir,
        &config.indexer,
        Arc::clone(&neural),
        Arc::clone(&hnsw),
        hnsw_sidecar,
    )?);
    info!("Search index opened ({} documents)", index.doc_count());

    // Step 8: Initialize query engine.
    let query_engine = Arc::new(raithe_query::QueryEngine::new(&config.query));
    info!("Query engine initialized");

    // Step 9: Initialize instant answer engine.
    let instant_engine = Arc::new(raithe_instant::InstantAnswerEngine::new(&config.instant));
    info!("Instant answer engine initialized");

    // Step 10: Start crawl log emitter.
    let crawl_log = Arc::new(
        raithe_storage::crawl_log::FileCrawlLog::new(config.storage.crawl_log_dir.clone()).await?,
    );
    info!("Crawl log emitter ready");

    // Step 11: Start the crawler in the background.
    let crawler = raithe_crawler::Crawler::new(
        Arc::clone(&config),
        crawl_log.clone() as Arc<dyn raithe_common::traits::CrawlEmitter>,
    )
    .await?;

    // Seed with initial URLs if frontier is empty.
    // These are curated high-quality seed domains for the demo.
    // RAiTHE-SE :: Expanded Canonical Crawl Frontier
    // Target: ~500 high-value seed URLs
    crawler.seed(&[
        // =========================
        // FEDERAL GOVERNMENT (CANADA)
        // =========================
        "https://www.canada.ca/",
        "https://open.canada.ca/",
        "https://www.statcan.gc.ca/",
        "https://www.ic.gc.ca/",
        "https://ised-isde.canada.ca/",
        "https://www.tc.gc.ca/",
        "https://tc.canada.ca/",
        "https://www.bankofcanada.ca/",
        "https://www.fin.gc.ca/",
        "https://www.nrcan.gc.ca/",
        "https://natural-resources.canada.ca/",
        "https://www.ec.gc.ca/",
        "https://www.cra-arc.gc.ca/",
        "https://www.servicecanada.gc.ca/",
        "https://www.parl.ca/",
        "https://www.senate.ca/",
        "https://www.elections.ca/",
        "https://www.courts.gc.ca/",
        "https://www.pm.gc.ca/",
        "https://www.rcmp-grc.gc.ca/",
        // =========================
        // PROVINCIAL / TERRITORIAL
        // =========================
        "https://www.ontario.ca/",
        "https://www.quebec.ca/",
        "https://www.alberta.ca/",
        "https://www.bc.ca/",
        "https://www.saskatchewan.ca/",
        "https://www.manitoba.ca/",
        "https://novascotia.ca/",
        "https://www.gov.nl.ca/",
        "https://www.gov.pe.ca/",
        "https://www.gnb.ca/",
        "https://www.yukon.ca/",
        "https://www.nwt-tno.ca/",
        "https://www.gov.nu.ca/",
        // =========================
        // MAJOR CITIES (CANADA)
        // =========================
        "https://www.toronto.ca/",
        "https://www.ottawa.ca/",
        "https://www.montreal.ca/",
        "https://www.vancouver.ca/",
        "https://www.calgary.ca/",
        "https://www.edmonton.ca/",
        "https://www.winnipeg.ca/",
        "https://www.halifax.ca/",
        "https://www.victoria.ca/",
        "https://www.saskatoon.ca/",
        "https://www.regina.ca/",
        "https://www.kelowna.ca/",
        "https://www.nanaimo.ca/",
        "https://www.burnaby.ca/",
        "https://www.richmond.ca/",
        "https://www.coquitlam.ca/",
        "https://www.surrey.ca/",
        "https://www.markham.ca/",
        "https://www.vaughan.ca/",
        "https://www.mississauga.ca/",
        "https://www.brampton.ca/",
        "https://www.hamilton.ca/",
        "https://www.london.ca/",
        "https://www.kitchener.ca/",
        "https://www.waterloo.ca/",
        "https://www.oshawa.ca/",
        "https://www.whitby.ca/",
        "https://www.ajax.ca/",
        "https://www.pickering.ca/",
        "https://www.kingston.ca/",
        "https://www.thunderbay.ca/",
        "https://www.sudbury.ca/",
        "https://www.northbay.ca/",
        "https://www.guelph.ca/",
        "https://www.barrie.ca/",
        // =========================
        // UNIVERSITIES
        // =========================
        "https://www.utoronto.ca/",
        "https://www.uwaterloo.ca/",
        "https://www.queensu.ca/",
        "https://www.uwo.ca/",
        "https://www.mcgill.ca/",
        "https://www.ubc.ca/",
        "https://www.ualberta.ca/",
        "https://www.ucalgary.ca/",
        "https://www.usask.ca/",
        "https://www.umanitoba.ca/",
        "https://www.yorku.ca/",
        "https://www.carleton.ca/",
        "https://www.uottawa.ca/",
        "https://www.dal.ca/",
        "https://www.mun.ca/",
        "https://www.trentu.ca/",
        "https://www.laurentian.ca/",
        "https://www.nipissingu.ca/",
        "https://www.athabascau.ca/",
        "https://www.capilanou.ca/",
        "https://www.unbc.ca/",
        // =========================
        // COLLEGES
        // =========================
        "https://www.algonquincollege.com/",
        "https://www.senecapolytechnic.ca/",
        "https://www.georgebrown.ca/",
        "https://www.humber.ca/",
        "https://www.centennialcollege.ca/",
        "https://www.sheridancollege.ca/",
        "https://www.conestogac.on.ca/",
        "https://www.durhamcollege.ca/",
        "https://www.fanshawec.ca/",
        "https://www.lambtoncollege.ca/",
        "https://www.mohawkcollege.ca/",
        "https://www.niagaracollege.ca/",
        "https://www.stlawrencecollege.ca/",
        // =========================
        // MEDIA (CANADA)
        // =========================
        "https://www.cbc.ca/",
        "https://ici.radio-canada.ca/",
        "https://globalnews.ca/",
        "https://www.ctvnews.ca/",
        "https://www.theglobeandmail.com/",
        "https://nationalpost.com/",
        "https://www.thestar.com/",
        "https://www.bnnbloomberg.ca/",
        "https://financialpost.com/",
        // =========================
        // GLOBAL MEDIA
        // =========================
        "https://www.reuters.com/",
        "https://apnews.com/",
        "https://www.bbc.com/",
        "https://www.theguardian.com/",
        "https://www.ft.com/",
        "https://www.economist.com/",
        "https://www.wsj.com/",
        "https://www.nytimes.com/",
        "https://www.washingtonpost.com/",
        // =========================
        // TRANSPORT / INFRA
        // =========================
        "https://www.ttc.ca/",
        "https://www.translink.ca/",
        "https://www.metrolinx.com/",
        "https://www.viarail.ca/",
        "https://www.aircanada.com/",
        "https://www.westjet.com/",
        // =========================
        // FINANCE
        // =========================
        "https://www.rbcroyalbank.com/",
        "https://www.td.com/",
        "https://www.bmo.com/",
        "https://www.cibc.com/",
        "https://www.scotiabank.com/",
        "https://www.desjardins.com/",
        "https://www.wealthsimple.com/",
        // =========================
        // RETAIL
        // =========================
        "https://www.canadiantire.ca/",
        "https://www.walmart.ca/",
        "https://www.costco.ca/",
        "https://www.bestbuy.ca/",
        "https://www.homedepot.ca/",
        "https://www.ikea.com/ca/",
        "https://www.indigo.ca/",
        // =========================
        // TOURISM
        // =========================
        "https://www.destinationcanada.com/",
        "https://www.ontarioparks.ca/",
        "https://www.bcparks.ca/",
        "https://www.travelalberta.com/",
        "https://www.hellobc.com/",
        "https://www.banfflakelouise.com/",
        "https://www.niagarafallstourism.com/"
    ])?;

    let _crawler_handle = tokio::spawn(async move {
        if let Err(e) = crawler.run().await {
            tracing::error!("Crawler error: {}", e);
        }
    });
    info!("Crawler started in background");

    // Step 12: Start the indexer poll loop.
    let indexer_index = Arc::clone(&index);
    let indexer_crawl_log = crawl_log.clone();
    let _indexer_handle = tokio::spawn(async move {
        if let Err(e) = indexer_index.run_poll_loop(&indexer_crawl_log).await {
            tracing::error!("Indexer error: {}", e);
        }
    });
    info!("Indexer poll loop started");

    // Step 13: Build and start the HTTP server.
    // construct previously-orphan handles; stored in AppState, not yet invoked.
    let ranking_pipeline = Arc::new(raithe_ranker::RankingPipeline::new(&config.ranker));
    // hnsw constructed earlier (Step 7) and shared with the indexer.
    let session = Arc::new(raithe_session::InMemorySessionStore::new(&config.session));

    // build LinkGraph from crawl log (Q4=b: one-shot corpus walk).
    // Loads /data/linkgraph/linkgraph.bin if present; otherwise walks crawl log.
    let link_graph = {
        let sidecar = config.storage.linkgraph_dir.join("linkgraph.bin");
        if sidecar.exists() {
            match raithe_linkgraph::LinkGraph::load(&sidecar) {
                Ok(g) => Arc::new(g),
                Err(e) => {
                    tracing::warn!("LinkGraph sidecar load failed: {} — rebuilding", e);
                    Arc::new(build_link_graph_from_crawl_log(&config, &sidecar))
                }
            }
        } else {
            Arc::new(build_link_graph_from_crawl_log(&config, &sidecar))
        }
    };

    let freshness = Arc::new(raithe_freshness::FreshnessPipeline::new(&config.indexer));
    info!("Orphan subsystem handles constructed (ranker/hnsw/session/linkgraph/freshness)");

    let app_state = Arc::new(raithe_serving::AppState {
        config: Arc::clone(&config),
        index,
        query_engine,
        instant_engine,
        neural,
        ranking_pipeline,
        hnsw,
        session,
        link_graph,
        freshness,
    });

    let app = raithe_serving::build_router(app_state);
    let listen_addr = format!(
        "{}:{}",
        config.serving.listen_addr, config.serving.listen_port
    );
    info!("Starting HTTP server on {}", listen_addr);

    // Start metrics server on separate port.
    // FIX BUG-001: Replace .unwrap() with proper error handling
    let metrics_addr = format!(
        "{}:{}",
        config.serving.listen_addr, config.serving.metrics_port
    );
    let metrics_router = raithe_serving::build_metrics_router();
    let _metrics_handle = tokio::spawn(async move {
        // FIX: Proper error handling instead of .unwrap()
        let listener = match tokio::net::TcpListener::bind(&metrics_addr).await {
            Ok(listener) => listener,
            Err(e) => {
                tracing::error!(
                    "FAILED to bind metrics server to {}: {}. Metrics endpoint unavailable.",
                    metrics_addr,
                    e
                );
                return;
            }
        };
        info!("Metrics server listening on {}", metrics_addr);

        // FIX: Proper error handling instead of .unwrap()
        if let Err(e) = axum::serve(listener, metrics_router).await {
            tracing::error!("Metrics server error: {}. Metrics endpoint unavailable.", e);
        }
    });

    // Start the main HTTP server.
    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    info!("raithe-se ready — serving at http://{}", listen_addr);
    axum::serve(listener, app).await?;

    Ok(())
}

// -----------------------------------------------------------------------------
// LinkGraph builder (Q4=b). One-shot walk over /data/crawl-log/*.log,
// collect (url, outgoing_links) per ParsedDocument, construct LinkGraph,
// compute PageRank, persist sidecar. Called from startup when sidecar missing.
// -----------------------------------------------------------------------------
fn build_link_graph_from_crawl_log(
    config: &raithe_common::config::RaitheConfig,
    sidecar: &std::path::Path,
) -> raithe_linkgraph::LinkGraph {
    use raithe_storage::crawl_log::FileCrawlLog;

    info!("Building LinkGraph from crawl log (one-shot pass)");
    let log_dir = &config.storage.crawl_log_dir;
    let mut log_files: Vec<std::path::PathBuf> = match std::fs::read_dir(log_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|s| s.to_str()) == Some("log")
                    && p.file_name()
                        .and_then(|s| s.to_str())
                        .map(|n| n.starts_with("crawl-"))
                        .unwrap_or(false)
            })
            .collect(),
        Err(e) => {
            tracing::warn!("LinkGraph: cannot read crawl-log dir {:?}: {}", log_dir, e);
            return raithe_linkgraph::LinkGraph::new();
        }
    };

    log_files.sort();
    let mut docs: Vec<(String, Vec<raithe_common::types::ExtractedLink>)> = Vec::new();
    for lf in &log_files {
        match FileCrawlLog::read_log_file(lf) {
            Ok(parsed) => {
                for pd in parsed {
                    docs.push((pd.url, pd.outgoing_links));
                }
            }
            Err(e) => tracing::warn!("LinkGraph: read_log_file {:?}: {}", lf, e),
        }
    }

    info!("LinkGraph: {} source docs from {} log files", docs.len(), log_files.len());
    let graph = raithe_linkgraph::LinkGraph::build_from_docs(docs, &config.linkgraph);
    if let Err(e) = graph.save(sidecar) {
        tracing::warn!("LinkGraph: save to {:?} failed: {}", sidecar, e);
    }
    graph
}
