// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: linkgraph
//
// Compressed link graph and PageRank computation.
//
// Storage: CSR (Compressed Sparse Row) format, memory-mapped.
// PageRank: power iteration, damping=0.85, convergence=1e-6.
// Topic-specific PageRank: 4 topic vectors for re-ranking boost.
// Anchor text aggregation: up to 100 per URL.
// ================================================================================

use raithe_common::config::LinkGraphConfig;
use raithe_common::types::ExtractedLink;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use tracing::{debug, info};

/// Compressed Sparse Row representation of the link graph.
/// Memory-mapped for zero-copy access.
pub struct LinkGraph {
    /// Number of nodes (pages) in the graph.
    pub node_count: u64,
    /// CSR row pointers: offsets[i] .. offsets[i+1] are the outlinks of node i.
    /// Length: node_count + 1.
    pub offsets: Vec<u64>,
    /// CSR column indices: the target node IDs of each link.
    pub targets: Vec<u32>,
    /// PageRank scores, one per node.
    pub pagerank: Vec<f32>,
    /// Domain authority scores, one per node.
    pub domain_authority: Vec<f32>,
    pub url_index: HashMap<String, u32>,
    pub pr_reference: Vec<f32>,
    pub pr_news: Vec<f32>,
    pub pr_commercial: Vec<f32>,
    pub pr_academic: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topic {
    Reference,
    News,
    Commercial,
    Academic,
}

/// Domain-suffix URL classifier (Q6=a; upgrade path logged as B.17).
/// Extracts host, lowercases, matches suffix/infix rules. Fallback: Reference.
pub fn classify_url(url: &str) -> Topic {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let host = after_scheme.split('/').next().unwrap_or("").to_ascii_lowercase();

    if host.ends_with(".edu") || host.contains(".ac.") || host.ends_with(".edu.au")
        || host.contains("university") || host.contains("arxiv.")
        || host.contains("scholar.") || host.contains("researchgate.")
    { return Topic::Academic; }

    if host.ends_with(".gov") || host.contains(".gov.")
        || host.contains("wikipedia.") || host.contains("wiktionary.")
        || host.contains("wikidata.") || host.starts_with("reference.")
        || host.starts_with("docs.") || host.contains("dictionary.")
        || host.contains(".gc.ca")
    { return Topic::Reference; }

    if host.ends_with(".news") || host.starts_with("news.")
        || host.contains("cbc.ca") || host.contains("bbc.")
        || host.contains("cnn.") || host.contains("reuters.")
        || host.contains("apnews.") || host.contains("nytimes.")
        || host.contains("washingtonpost.") || host.contains("theglobeandmail.")
        || host.contains("radio-canada.")
    { return Topic::News; }

    if host.ends_with(".shop") || host.ends_with(".store")
        || host.starts_with("shop.") || host.starts_with("store.")
        || host.contains("amazon.") || host.contains("ebay.")
        || host.contains("etsy.") || host.contains("walmart.")
        || host.contains("costco.") || host.contains("homedepot.")
    { return Topic::Commercial; }

    Topic::Reference
}

impl LinkGraph {
    /// Create an empty link graph.
    pub fn new() -> Self {
        Self {
            node_count: 0,
            offsets: vec![0],
            targets: Vec::new(),
            pagerank: Vec::new(),
            domain_authority: Vec::new(),
            url_index: HashMap::new(),
            pr_reference: Vec::new(),
            pr_news: Vec::new(),
            pr_commercial: Vec::new(),
            pr_academic: Vec::new(),
        }
    }

    /// Run PageRank power iteration.
    /// damping: 0.85, convergence: 1e-6, max_iterations: 50.
    pub fn compute_pagerank(&mut self, config: &LinkGraphConfig) {
        let n = self.node_count as usize;
        if n == 0 {
            return;
        }

        info!("Computing PageRank for {} nodes (damping={}, max_iter={})",
            n, config.pagerank_damping, config.pagerank_max_iterations);

        let d = config.pagerank_damping;
        let base = (1.0 - d) / n as f64;

        // Initialize all PageRank scores to 1/N.
        let mut pr: Vec<f64> = vec![1.0 / n as f64; n];
        let mut new_pr: Vec<f64> = vec![0.0; n];

        for iteration in 0..config.pagerank_max_iterations {
            // Reset new_pr.
            new_pr.fill(base);

            // For each node, distribute its PR to its outlinks.
            for i in 0..n {
                let start = self.offsets[i] as usize;
                let end = self.offsets[i + 1] as usize;
                let out_degree = end - start;

                if out_degree == 0 {
                    // Dangling node: distribute to all nodes.
                    let contribution = d * pr[i] / n as f64;
                    for val in new_pr.iter_mut() {
                        *val += contribution;
                    }
                } else {
                    let contribution = d * pr[i] / out_degree as f64;
                    for j in start..end {
                        let target = self.targets[j] as usize;
                        if target < n {
                            new_pr[target] += contribution;
                        }
                    }
                }
            }

            // Check convergence (L1 norm).
            let delta: f64 = pr.iter().zip(new_pr.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            std::mem::swap(&mut pr, &mut new_pr);

            debug!("PageRank iteration {}: delta={:.2e}", iteration, delta);

            if delta < config.pagerank_convergence_threshold {
                info!("PageRank converged after {} iterations (delta={:.2e})",
                    iteration + 1, delta);
                break;
            }
        }

        // Store as f32.
        self.pagerank = pr.into_iter().map(|v| v as f32).collect();
    }

    /// Get outlinks for a node.
    pub fn outlinks(&self, node_id: usize) -> &[u32] {
        if node_id >= self.node_count as usize {
            return &[];
        }
        let start = self.offsets[node_id] as usize;
        let end = self.offsets[node_id + 1] as usize;
        &self.targets[start..end]
    }

    /// Get PageRank score for a node.
    pub fn get_pagerank(&self, node_id: usize) -> f32 {
        self.pagerank.get(node_id).copied().unwrap_or(0.0)
    }

    // ---------------------------------------------------------------------
    // Build / query / persist (Q4=b, one-shot corpus walk)
    // ---------------------------------------------------------------------

    /// Build a LinkGraph from an iterator of (source_url, outgoing_links).
    /// Uses first-seen order for url -> node_id assignment. Source urls are
    /// assigned node_ids 0..S-1 first so crawled docs occupy dense low ids.
    pub fn build_from_docs<I>(docs: I, config: &LinkGraphConfig) -> Self
    where
        I: IntoIterator<Item = (String, Vec<ExtractedLink>)>,
    {
        let mut url_index: HashMap<String, u32> = HashMap::new();
        let doc_vec: Vec<(String, Vec<ExtractedLink>)> = docs.into_iter().collect();

        // Pass 1a: assign ids to source urls first (dense low ids).
        for (src, _) in &doc_vec {
            let next = url_index.len() as u32;
            url_index.entry(src.clone()).or_insert(next);
        }
        // Pass 1b: assign ids to any target urls not already seen.
        for (_, outs) in &doc_vec {
            for l in outs {
                let next = url_index.len() as u32;
                url_index.entry(l.url.clone()).or_insert(next);
            }
        }

        // Pass 2: build (src_id, out_ids) pairs, dedup outs.
        let mut edges: Vec<(u32, Vec<u32>)> = Vec::with_capacity(doc_vec.len());
        for (src, outs) in &doc_vec {
            let src_id = *url_index.get(src).unwrap();
            let mut out_ids: Vec<u32> =
                outs.iter().map(|l| *url_index.get(&l.url).unwrap()).collect();
            out_ids.sort_unstable();
            out_ids.dedup();
            edges.push((src_id, out_ids));
        }
        edges.sort_by_key(|(s, _)| *s);

        // CSR assembly.
        let node_count = url_index.len() as u64;
        let n = node_count as usize;
        let mut offsets: Vec<u64> = vec![0u64; n + 1];
        let mut targets: Vec<u32> = Vec::new();
        let mut edge_iter = edges.into_iter().peekable();
        for i in 0..n {
            offsets[i] = targets.len() as u64;
            while let Some((s, _)) = edge_iter.peek() {
                if (*s as usize) != i {
                    break;
                }
                let (_, outs) = edge_iter.next().unwrap();
                targets.extend(outs);
            }
        }
        offsets[n] = targets.len() as u64;

        let mut graph = Self {
            node_count,
            offsets,
            targets,
            pagerank: Vec::new(),
            domain_authority: vec![0.0; n],
            url_index,
            pr_reference: Vec::new(),
            pr_news: Vec::new(),
            pr_commercial: Vec::new(),
            pr_academic: Vec::new(),
        };
        graph.compute_pagerank(config);
        graph.compute_topic_pageranks(config);
        info!(
            "LinkGraph built: {} nodes, {} edges, PageRank + topic-PR computed",
            graph.node_count,
            graph.targets.len()
        );
        graph
    }

    /// Look up PageRank by url. Returns 0.0 for unknown urls.
    pub fn pagerank_for_url(&self, url: &str) -> f32 {
        self.url_index
            .get(url)
            .and_then(|id| self.pagerank.get(*id as usize).copied())
            .unwrap_or(0.0)
    }

    /// M3c — Compute four topic-specific PageRank vectors via personalized
    /// power iteration, seeds classified by classify_url (Q6=a).
    pub fn compute_topic_pageranks(&mut self, config: &LinkGraphConfig) {
        let n = self.node_count as usize;
        if n == 0 {
            self.pr_reference = Vec::new();
            self.pr_news = Vec::new();
            self.pr_commercial = Vec::new();
            self.pr_academic = Vec::new();
            return;
        }
        let mut seeds_ref: Vec<usize> = Vec::new();
        let mut seeds_news: Vec<usize> = Vec::new();
        let mut seeds_com: Vec<usize> = Vec::new();
        let mut seeds_acad: Vec<usize> = Vec::new();
        for (url, id) in &self.url_index {
            match classify_url(url) {
                Topic::Reference => seeds_ref.push(*id as usize),
                Topic::News => seeds_news.push(*id as usize),
                Topic::Commercial => seeds_com.push(*id as usize),
                Topic::Academic => seeds_acad.push(*id as usize),
            }
        }
        info!(
            "Topic-PR seeds: ref={} news={} com={} acad={}",
            seeds_ref.len(), seeds_news.len(), seeds_com.len(), seeds_acad.len()
        );
        info!("Computing topic-PR: Reference");
        self.pr_reference = self.personalized_pagerank(&seeds_ref, config);
        info!("Computing topic-PR: News");
        self.pr_news = self.personalized_pagerank(&seeds_news, config);
        info!("Computing topic-PR: Commercial");
        self.pr_commercial = self.personalized_pagerank(&seeds_com, config);
        info!("Computing topic-PR: Academic");
        self.pr_academic = self.personalized_pagerank(&seeds_acad, config);
        info!("Topic-PR computation complete");
    }

    /// Personalized PageRank: teleport mass (1-d) concentrated on seeds.
    /// Falls back to uniform teleport if seed list is empty.
    ///
    /// m3c-fix — O(n) dangling-mass aggregation instead of O(n^2). Drops one
    /// topic-PR iteration from ~hours to ~seconds on a 262k-node graph that
    /// is ~99% dangling.
    fn personalized_pagerank(&self, seeds: &[usize], config: &LinkGraphConfig) -> Vec<f32> {
        let n = self.node_count as usize;
        let d = config.pagerank_damping;
        let mut teleport: Vec<f64> = vec![0.0; n];
        if seeds.is_empty() {
            let u = (1.0 - d) / n as f64;
            for v in teleport.iter_mut() { *v = u; }
        } else {
            let per_seed = (1.0 - d) / seeds.len() as f64;
            for &s in seeds { if s < n { teleport[s] = per_seed; } }
        }
        let mut is_dangling: Vec<bool> = vec![false; n];
        for i in 0..n {
            if self.offsets[i + 1] == self.offsets[i] { is_dangling[i] = true; }
        }
        let teleport_norm = (1.0 - d).max(1e-12);

        let mut pr: Vec<f64> = vec![1.0 / n as f64; n];
        let mut new_pr: Vec<f64> = vec![0.0; n];
        for _ in 0..config.pagerank_max_iterations {
            new_pr.copy_from_slice(&teleport);
            let mut dangling_sum = 0.0f64;
            for i in 0..n {
                if is_dangling[i] { dangling_sum += pr[i]; }
            }
            let dangling_factor = d * dangling_sum / teleport_norm;
            for j in 0..n {
                new_pr[j] += dangling_factor * teleport[j];
            }
            for i in 0..n {
                if is_dangling[i] { continue; }
                let start = self.offsets[i] as usize;
                let end = self.offsets[i + 1] as usize;
                let out_degree = end - start;
                let contribution = d * pr[i] / out_degree as f64;
                for j in start..end {
                    let t = self.targets[j] as usize;
                    if t < n { new_pr[t] += contribution; }
                }
            }
            let delta: f64 = pr.iter().zip(new_pr.iter()).map(|(a,b)| (a-b).abs()).sum();
            std::mem::swap(&mut pr, &mut new_pr);
            if delta < config.pagerank_convergence_threshold { break; }
        }
        pr.into_iter().map(|v| v as f32).collect()
    }

    /// Look up all four topic-PR scores by url. Returns [0.0; 4] if unknown.
    pub fn topic_pageranks_for_url(&self, url: &str) -> [f32; 4] {
        let Some(&id) = self.url_index.get(url) else { return [0.0; 4]; };
        let i = id as usize;
        [
            self.pr_reference.get(i).copied().unwrap_or(0.0),
            self.pr_news.get(i).copied().unwrap_or(0.0),
            self.pr_commercial.get(i).copied().unwrap_or(0.0),
            self.pr_academic.get(i).copied().unwrap_or(0.0),
        ]
    }

    /// Persist to disk. Manual binary format, no external deps.
    /// Layout: magic(4 "RAIT") ver(4) node_count(8)
    ///         [u32_len + utf8_bytes] * node_count
    ///         offsets[(node_count+1) * u64]
    ///         targets_len(8) targets[t * u32]
    ///         pagerank[node_count * f32]
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"RAIT")?;
        f.write_all(&2u32.to_le_bytes())?; // M3c — v2 adds topic-PR vectors
        f.write_all(&self.node_count.to_le_bytes())?;
        let mut urls_by_id: Vec<(u32, &String)> =
            self.url_index.iter().map(|(u, id)| (*id, u)).collect();
        urls_by_id.sort_by_key(|(id, _)| *id);
        for (_, u) in &urls_by_id {
            let b = u.as_bytes();
            f.write_all(&(b.len() as u32).to_le_bytes())?;
            f.write_all(b)?;
        }
        for o in &self.offsets {
            f.write_all(&o.to_le_bytes())?;
        }
        f.write_all(&(self.targets.len() as u64).to_le_bytes())?;
        for t in &self.targets {
            f.write_all(&t.to_le_bytes())?;
        }
        for p in &self.pagerank {
            f.write_all(&p.to_le_bytes())?;
        }
        // M3c v2 — four topic vectors after base pagerank, in fixed order.
        for vec in [&self.pr_reference, &self.pr_news, &self.pr_commercial, &self.pr_academic] {
            for p in vec.iter() {
                f.write_all(&p.to_le_bytes())?;
            }
        }
        debug!("LinkGraph saved to {:?}", path);
        Ok(())
    }

    /// Load from disk. Inverse of save().
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RAIT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "linkgraph.bin: bad magic",
            ));
        }
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf4)?;
        let ver = u32::from_le_bytes(buf4);
        if ver != 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("linkgraph.bin: version {} (need 2, rebuild required)", ver),
            ));
        }
        f.read_exact(&mut buf8)?;
        let node_count = u64::from_le_bytes(buf8);
        let n = node_count as usize;
        let mut url_index = HashMap::with_capacity(n);
        for i in 0..n {
            f.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;
            let mut s = vec![0u8; len];
            f.read_exact(&mut s)?;
            url_index.insert(String::from_utf8_lossy(&s).into_owned(), i as u32);
        }
        let mut offsets = vec![0u64; n + 1];
        for o in offsets.iter_mut() {
            f.read_exact(&mut buf8)?;
            *o = u64::from_le_bytes(buf8);
        }
        f.read_exact(&mut buf8)?;
        let tlen = u64::from_le_bytes(buf8) as usize;
        let mut targets = vec![0u32; tlen];
        for t in targets.iter_mut() {
            f.read_exact(&mut buf4)?;
            *t = u32::from_le_bytes(buf4);
        }
        let mut pagerank = vec![0f32; n];
        for p in pagerank.iter_mut() {
            f.read_exact(&mut buf4)?;
            *p = f32::from_le_bytes(buf4);
        }
        // M3c v2 — read four topic vectors in the same order save() wrote them.
        let mut read_vec = |len: usize| -> std::io::Result<Vec<f32>> {
            let mut v = vec![0f32; len];
            let mut b = [0u8; 4];
            for slot in v.iter_mut() {
                f.read_exact(&mut b)?;
                *slot = f32::from_le_bytes(b);
            }
            Ok(v)
        };
        let pr_reference = read_vec(n)?;
        let pr_news = read_vec(n)?;
        let pr_commercial = read_vec(n)?;
        let pr_academic = read_vec(n)?;
        info!("LinkGraph loaded from {:?}: {} nodes (v2, with topic-PR)", path, node_count);
        Ok(Self {
            node_count,
            offsets,
            targets,
            pagerank,
            domain_authority: vec![0.0; n],
            url_index,
            pr_reference,
            pr_news,
            pr_commercial,
            pr_academic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_simple_graph() {
        // Simple 3-node graph: 0 -> 1 -> 2 -> 0 (cycle)
        let mut graph = LinkGraph {
            node_count: 3,
            offsets: vec![0, 1, 2, 3],
            targets: vec![1, 2, 0],
            pagerank: Vec::new(),
            domain_authority: Vec::new(),
            url_index: HashMap::new(),
            pr_reference: Vec::new(),
            pr_news: Vec::new(),
            pr_commercial: Vec::new(),
            pr_academic: Vec::new(),
        };

        let config = LinkGraphConfig {
            pagerank_damping: 0.85,
            pagerank_convergence_threshold: 1e-6,
            pagerank_max_iterations: 50,
            topic_pagerank_count: 4,
            max_anchor_texts_per_url: 100,
        };

        graph.compute_pagerank(&config);

        // In a cycle, all nodes should converge to equal PageRank (1/3).
        let expected = 1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (graph.pagerank[i] - expected as f32).abs() < 0.01,
                "Node {} PR={}, expected ~{}",
                i,
                graph.pagerank[i],
                expected
            );
        }
    }
}
