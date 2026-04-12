<div align="center">
  <h1>© RAiTHE INDUSTRIES INCORPORATED 2026</h1>
</div>

<div align="left">

<br>

                                          ██████╗  █████╗ ██╗████████╗██╗  ██╗███████╗
                                          ██╔══██╗██╔══██╗   ╚══██╔══╝██║  ██║██╔════╝
                                          ██████╔╝███████║██║   ██║   ███████║█████╗
                                          ██╔══██╗██╔══██║██║   ██║   ██╔══██║██╔══╝
                                          ██║  ██║██║  ██║██║   ██║   ██║  ██║███████╗
                                          ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                                                          SEARCH ENGINE

# raithe-se

**The World's Best Search Engine in Rust.**

Demonstrator architecture with hybrid neural ranking,
LLM-assisted query understanding, and structured instant answers.
Designed for consumer hardware deployment with a clear scaling path
to distributed production infrastructure.

## Project Status

**Overall Alignment:** ✅ GOOD FOUNDATION, DOCUMENTED GAPS

The raithe-se codebase has been thoroughly analyzed and verified against
the engineering specification. All 72 known issues are documented and
classified for prioritized resolution.

| Category | Count | Breakdown |
|----------|-------|-----------|
| CRITICAL Deviations | 2 | GBDT Algorithm, Neural Tokenizer |
| HIGH Deviations | 4 | Seed List, Rate Limiting, Step 3 Neural, Parser |
| MEDIUM Deviations | 1 | Index Writer Heap |
| Documented Bugs | 28 | 6 CRITICAL, 10 HIGH, 8 MEDIUM, 4 LOW |
| Security Issues | 37 | 2 CRITICAL, 8 HIGH, 12 MEDIUM, 15 LOW |
| **TOTAL ISSUES** | **72** | |

See [Engineering Specification](raithe-se_Engineering_Spec_.txt),
[Security Analysis](raithe-se_Security_Analysis.txt), and
[Bug Analysis Report](raithe-se_Bug_Analysis_Report.txt) for details.

## Architecture

raithe-se is organized as a Cargo workspace with 16 crates:

| Crate | Description |
|-------|-------------|
| `common` | Shared types, error handling, config, feature flags, SimHash |
| `metrics` | Prometheus metrics registry and structured logging |
| `storage` | Local SSD storage, mmap, crawl log, backup |
| `crawler` | Async HTTP crawler with politeness and bandwidth throttling |
| `parser` | HTML parsing, boilerplate removal, structural extraction |
| `indexer` | Tantivy-backed inverted index with custom tokenizer |
| `ranker` | Three-phase ranking: BM25F → GBDT → cross-encoder |
| `query` | Query understanding: spell correction, intent, synonyms, LLM |
| `serving` | Axum HTTP server and query coordination |
| `linkgraph` | Compressed link graph and PageRank computation |
| `freshness` | Incremental re-crawl pipeline |
| `semantic` | HNSW ANN index for dense embedding retrieval |
| `instant` | Instant answers: calculator, conversions, knowledge graph |
| `neural` | GPU inference manager (ONNX Runtime + CUDA) |
| `session` | In-memory session context tracking |
| `app` | Single-binary entry point |

## Hardware Target (Single-Node Demo)

| Component | Specification |
|-----------|---------------|
| CPU | Intel i7-6700K (4C/8T, 4.0 GHz base) |
| RAM | 32 GB DDR4 |
| GPU | NVIDIA GTX 1080 (8 GB VRAM) |
| Storage | 1 TB NVMe SSD |
| Network | 300 Mbps down / 20 Mbps up |

## Quick Start

Open terminal, run in bash.
# Build in release mode (LTO enabled)
cargo build --release
<br>

# Run with default config
./target/release/raithe-se

# Run with custom config
./target/release/raithe-se /path/to/engine.toml

raithe-se will be available at:
`http://localhost:8080`.
<br>
Prometheus metrics are exposed at:
`http://localhost:9090/metrics`.

## Configuration

<br>
Configuration is loaded in order (Section 14.2):
<br>
<br> 1. Compiled-in defaults
<br> 2. `engine.toml` file
<br> 3. Environment variables (`RAITHE__SECTION__KEY`)
<br> 4. CLI arguments
<br>
<br> Hot-reload: config file changes are detected within 5 seconds.

## Security Issues

⚠️ **Security Analysis Status:** 37 issues documented and verified.

| Priority | Count | Key Issues |
|----------|-------|------------|
| CRITICAL | 2 | Path traversal in backup system, Integer overflow in doc ID |
| HIGH | 8 | DoS attacks (robots.txt exhaustion, slowloris), Missing rate limiting, etc. |
| MEDIUM | 12 | CORS misconfiguration, Missing security headers, Info disclosure, etc. |
| LOW | 15 | Missing security headers (CSP, HSTS), Weak crypto, etc. |

### Critical Security Issues

**BUG-001: Path Traversal in Backup System**
- Location: `crates/storage/src/backup.rs:51-72`
- Issue: No validation of destination paths in copy_dir_recursive
- Impact: Arbitrary file write via "../" in filenames

**BUG-002: Integer Overflow in Doc ID Counter**
- Location: `crates/indexer/src/lib.rs:241-243`
- Issue: AtomicU64::fetch_add without overflow check
- Impact: Doc ID collisions cause index corruption

### Other Notable Security Issues

- BUG-009: Rate limiting NOT implemented (governor crate unused)
- BUG-013: Version info exposed in health endpoint
- BUG-015: XXE risk in XML parsing
- BUG-022: Using vulnerable dependencies (run `cargo audit`)

See [Security Analysis Report](raithe-se_Security_Analysis.txt) for full details.

## Known Bugs

⚠️ **Bug Analysis Status:** 28 bugs documented and verified.

| Priority | Count | Key Issues |
|----------|-------|------------|
| CRITICAL | 6 | .unwrap() panic in metrics, Lock poisoning, Division by zero, Unsafe memory |
| HIGH | 10 | Deadlock risk, Memory leaks, Race conditions, Infinite loop in GBDT |
| MEDIUM | 8 | UTF-8 lossy conversion, Pagination bounds, Silent failures |
| LOW | 4 | Ignored errors, String cloning, Data races |

### Critical Bug Fixes Required

| Bug | Location | Issue |
|-----|----------|-------|
| BUG-001 | `crates/app/src/main.rs:363-365` | .unwrap() panic in metrics server binding |
| BUG-002 | `crates/indexer/src/lib.rs:261-265` | Lock poisoning with RwLock::unwrap() |
| BUG-003 | `crates/crawler/src/scheduler.rs:38` | Division by zero in scheduler |
| BUG-006 | `crates/neural/src/lib.rs:117-121` | Array index out of bounds in tokenizer |

See [Bug Analysis Report](raithe-se_Bug_Analysis_Report.txt) for full details.

## Deviations from Specification

The following deviations from the engineering specification have been
identified and documented:

| Dev | Priority | Issue | Impact |
|-----|----------|-------|--------|
| DEV-001 | CRITICAL | GBDT uses 10 hand-tuned trees instead of 300 LambdaMART trees | Degraded ranking quality |
| DEV-002 | CRITICAL | Neural tokenizer is hash-based placeholder, not vocabulary-based | Neural pipeline non-functional |
| DEV-003 | HIGH | Seed list has 147 URLs instead of 1000+ required | Limited crawl coverage |
| DEV-004 | HIGH | Rate limiting not implemented (governor crate unused) | DoS vulnerability |
| DEV-005 | HIGH | Step 3 neural cross-encoder never invoked | Cannot achieve best results |
| DEV-006 | MEDIUM | Parser uses scraper instead of lol_html as primary | Higher memory usage |
| DEV-007 | MEDIUM | Index writer heap fixed at 50MB instead of 1GB | Reduced indexing throughput |

See [Engineering Specification](raithe-se_Engineering_Spec_.txt) for full details.

## License

Proprietary — © RAiTHE INDUSTRIES INCORPORATED 2026.
All rights reserved.

---
*Engineered in Rust — Smarter paths, faster results.*
