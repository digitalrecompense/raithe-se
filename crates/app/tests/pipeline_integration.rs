// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: integration test
//
// End-to-end pipeline test:
//   1. Parse raw HTML into a ParsedDocument.
//   2. Emit it to the crawl log (FileCrawlLog).
//   3. Read it back from the crawl log.
//   4. Index it into Tantivy via RaitheIndex.
//   5. Search for it and verify the result.
//   6. Verify instant answers work alongside.
//   7. Add a second document, verify ranking order.
// ================================================================================

use raithe_common::config::RaitheConfig;
use raithe_common::traits::{CrawlEmitter, InstantAnswerProvider};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_full_pipeline_crawl_index_search() {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path();

    let index_dir = base.join("index");
    let crawl_log_dir = base.join("crawl-log");
    std::fs::create_dir_all(&index_dir).unwrap();
    std::fs::create_dir_all(&crawl_log_dir).unwrap();

    // -- Step 1: Parse a realistic HTML document --
    let html = r#"
    <html lang="en">
    <head>
        <title>Rust Programming Language — Official Site</title>
        <meta name="description" content="A language empowering everyone to build reliable and efficient software.">
        <meta property="og:title" content="Rust Programming Language">
    </head>
    <body>
        <nav>Navigation links here that should be filtered</nav>
        <article>
            <h1>Welcome to Rust</h1>
            <p>Rust is a systems programming language focused on safety, speed, and concurrency.
               It accomplishes these goals without having a garbage collector, making it useful
               for a number of use cases other languages are not good at. Rust provides memory
               safety guarantees through its ownership system. The borrow checker validates
               references at compile time, preventing data races and dangling pointers.
               Rust has been voted the most loved programming language for multiple years
               in the Stack Overflow developer survey.</p>
            <h2>Why Choose Rust?</h2>
            <p>Performance is comparable to C and C++ while providing memory safety without
               a garbage collector. Rust enables developers to write fast, reliable code
               with zero-cost abstractions. The type system and ownership model guarantee
               memory-safety and thread-safety at compile time. Fearless concurrency lets
               you write parallel code without the usual headaches of data races.</p>
            <h2>Getting Started</h2>
            <p>Install Rust using rustup, the Rust toolchain installer. The Cargo package
               manager handles building your code, downloading dependencies, and building
               libraries. The Rust community is welcoming and helpful, with extensive
               documentation available at doc.rust-lang.org.</p>
        </article>
        <footer>Footer content here</footer>
        <a href="/learn">Learn Rust</a>
        <a href="/tools">Tools</a>
        <a href="https://crates.io">Crates.io</a>
    </body>
    </html>
    "#;

    let parsed_doc = raithe_parser::parse_html(
        html.as_bytes(),
        "https://www.rust-lang.org/",
        200,
        "text/html",
    )
    .expect("Parse should succeed");

    assert_eq!(parsed_doc.title, "Rust Programming Language");
    assert_eq!(parsed_doc.language, "en");
    assert!(parsed_doc.word_count > 50, "Should have substantial content");
    assert!(
        parsed_doc.body.contains("memory safety"),
        "Body should contain main article content"
    );
    assert!(
        !parsed_doc.body.contains("Navigation links"),
        "Boilerplate should be removed"
    );
    assert!(!parsed_doc.outgoing_links.is_empty());

    eprintln!(
        "✓ Parsed: title='{}', words={}, links={}, simhash={:016x}",
        parsed_doc.title,
        parsed_doc.word_count,
        parsed_doc.outgoing_links.len(),
        parsed_doc.simhash
    );

    // -- Step 2: Emit to crawl log --
    let crawl_log = raithe_storage::crawl_log::FileCrawlLog::new(crawl_log_dir.clone())
        .await
        .expect("Crawl log init");

    crawl_log
        .emit(parsed_doc.clone())
        .await
        .expect("Emit should succeed");

    eprintln!("✓ Emitted to crawl log");

    // -- Step 3: Read it back from the crawl log --
    let log_files = crawl_log.list_log_files();
    assert!(!log_files.is_empty(), "Should have at least one log file");

    let read_docs = raithe_storage::crawl_log::FileCrawlLog::read_log_file(&log_files[0])
        .expect("Read log file");
    assert_eq!(read_docs.len(), 1);
    assert_eq!(read_docs[0].url, "https://www.rust-lang.org/");

    eprintln!("✓ Read back from crawl log: {} documents", read_docs.len());

    // -- Step 4: Index into Tantivy --
    let config = RaitheConfig::default();
    // M4a — RaitheIndex::open now requires a neural manager + HNSW handle.
    // In the test we use a no-model neural manager (graceful degradation:
    // embed returns zero vectors, indexer skips HNSW insert).
    let neural = std::sync::Arc::new(
        raithe_neural::GpuInferenceManager::new(&config.neural).expect("neural init"),
    );
    let hnsw = std::sync::Arc::new(std::sync::RwLock::new(
        raithe_semantic::HnswIndex::new(&config.semantic),
    ));
    let test_sidecar = index_dir.parent().unwrap().join("hnsw.bin");
    let index = raithe_indexer::RaitheIndex::open(
        &index_dir,
        &config.indexer,
        neural,
        hnsw,
        test_sidecar,
    )
    .expect("Index open");

    assert_eq!(index.doc_count(), 0, "Index should start empty");

    index
        .add_document(&read_docs[0])
        .expect("Add document should succeed");
    index.commit().expect("Commit should succeed");

    assert_eq!(
        index.doc_count(),
        1,
        "Index should have 1 document after commit"
    );

    eprintln!("✓ Indexed: {} documents in Tantivy", index.doc_count());

    // -- Step 5: Search and verify results --
    let (results, total_hits) = index
        .search("rust programming", 10, 0)
        .expect("Search should succeed");

    assert!(total_hits > 0, "Should have at least 1 hit");
    assert!(!results.is_empty(), "Results should not be empty");
    assert_eq!(results[0].url, "https://www.rust-lang.org/");
    assert!(results[0].title.contains("Rust"));
    assert!(results[0].score > 0.0, "Score should be positive");
    assert!(!results[0].snippet.is_empty(), "Snippet should be generated");
    assert_eq!(results[0].domain, "www.rust-lang.org");

    eprintln!(
        "✓ Search 'rust programming': {} hits, score={:.4}, title='{}'",
        total_hits, results[0].score, results[0].title
    );
    eprintln!(
        "  Snippet: {}...",
        &results[0].snippet[..results[0].snippet.len().min(120)]
    );

    // Body text search.
    let (results2, _) = index
        .search("memory safety", 10, 0)
        .expect("Search should succeed");
    assert!(!results2.is_empty(), "Should find document by body content");
    assert_eq!(results2[0].url, "https://www.rust-lang.org/");

    eprintln!("✓ Search 'memory safety': found via body text");

    // Negative search.
    let (results3, hits3) = index
        .search("javascript", 10, 0)
        .expect("Search should succeed");
    assert_eq!(hits3, 0, "Should not find unrelated query");
    assert!(results3.is_empty());

    eprintln!("✓ Search 'javascript': 0 results (correct)");

    // Specific term.
    let (results4, _) = index
        .search("borrow checker", 10, 0)
        .expect("Search should succeed");
    assert!(!results4.is_empty(), "Should find by specific body term");

    eprintln!("✓ Search 'borrow checker': found specific term");

    // -- Step 6: Verify query understanding --
    let query_engine = raithe_query::QueryEngine::new(&config.query);
    let parsed_query = query_engine.parse_query("how does rust borrow checker work");

    assert_eq!(
        parsed_query.intent,
        raithe_common::types::QueryIntent::Informational
    );
    assert!(parsed_query.terms.contains(&"rust".to_string()));
    assert!(parsed_query.terms.contains(&"borrow".to_string()));

    eprintln!(
        "✓ Query understanding: intent={:?}, terms={:?}",
        parsed_query.intent, parsed_query.terms
    );

    // -- Step 7: Verify instant answers --
    let instant_engine = raithe_instant::InstantAnswerEngine::new(&config.instant);

    let calc_query = query_engine.parse_query("2 + 3 * 4");
    let calc_answer = instant_engine
        .answer(&calc_query)
        .await
        .expect("Instant answer should not error");

    assert!(calc_answer.is_some(), "Calculator should produce an answer");
    let ca = calc_answer.unwrap();
    assert!(ca.answer.contains("14"), "2 + 3 * 4 = 14");

    eprintln!("✓ Instant answer (calculator): {}", ca.answer);

    let conv_query = query_engine.parse_query("100 km in miles");
    let conv_answer = instant_engine
        .answer(&conv_query)
        .await
        .expect("Instant answer should not error");

    assert!(conv_answer.is_some(), "Unit conversion should produce answer");
    let cv = conv_answer.unwrap();
    assert!(cv.answer.contains("62"), "100 km ≈ 62 miles");

    eprintln!("✓ Instant answer (conversion): {}", cv.answer);

    // -- Step 8: Add a second document and verify ranking --
    let html2 = r#"
    <html lang="en">
    <head>
        <title>Python Programming Language</title>
        <meta name="description" content="Python is a popular general-purpose programming language.">
    </head>
    <body>
        <article>
            <h1>Python Overview</h1>
            <p>Python is an interpreted, high-level, general-purpose programming language.
               Created by Guido van Rossum and first released in 1991, Python design
               philosophy emphasizes code readability. Python features a dynamic type
               system and automatic memory management via garbage collection. It supports
               multiple programming paradigms including structured, object-oriented, and
               functional programming. Python is often described as a batteries-included
               language due to its comprehensive standard library. It is widely used in
               web development, data science, artificial intelligence, and scientific
               computing.</p>
        </article>
    </body>
    </html>
    "#;

    let parsed_doc2 = raithe_parser::parse_html(
        html2.as_bytes(),
        "https://www.python.org/",
        200,
        "text/html",
    )
    .expect("Parse should succeed");

    index
        .add_document(&parsed_doc2)
        .expect("Add second document");
    index.commit().expect("Commit");

    assert_eq!(index.doc_count(), 2);

    // "rust" should rank rust-lang.org first.
    let (results_rust, _) = index.search("rust", 10, 0).expect("Search");
    assert_eq!(results_rust[0].domain, "www.rust-lang.org");

    // "python" should rank python.org first.
    let (results_python, _) = index.search("python", 10, 0).expect("Search");
    assert_eq!(results_python[0].domain, "www.python.org");

    // "programming language" should return both.
    let (results_both, _) = index
        .search("programming language", 10, 0)
        .expect("Search");
    assert!(results_both.len() >= 2, "Both docs should match");

    eprintln!(
        "✓ Ranking: 'rust' → {}, 'python' → {}, 'programming language' → {} results",
        results_rust[0].domain,
        results_python[0].domain,
        results_both.len()
    );

    eprintln!("\n✅ ALL PIPELINE TESTS PASSED — raithe-se end-to-end verified");
}
