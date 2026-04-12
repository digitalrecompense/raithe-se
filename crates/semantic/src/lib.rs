// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: semantic
//
// Dense embedding index using HNSW.
//
// Bi-encoder (all-MiniLM-L6-v2) produces 128-dim embeddings.
// HNSW graph: M=16, ef_construction=100, ef_search=50.
// Product quantization: 16 sub-quantizers for ~6x compression.
// Stored on SSD, mmap'd for query-time access.
//
// Full HNSW implementation:
//   Insert: greedy search from entry point through upper layers, then
//   ef_construction-width beam search at the insert layer and below.
//   Bidirectional connections, pruned to M neighbors per layer.
//
//   Search: greedy descent through upper layers, ef_search-width beam
//   search at layer 0, return top-k from the beam.
// ================================================================================

use raithe_common::config::SemanticConfig;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use tracing::{debug, info};

/// A node in the HNSW graph.
#[derive(Clone)]
struct HnswNode {
    /// Document ID this node represents.
    doc_id: u64,
    /// Dense embedding vector (128 dimensions).
    vector: Vec<f32>,
    /// Connections per layer: layer_connections[layer] = vec of neighbor node indices.
    layer_connections: Vec<Vec<usize>>,
    /// Maximum layer this node appears on (stored for graph serialization).
    _max_layer: usize,
}

/// HNSW index for approximate nearest neighbor search.
pub struct HnswIndex {
    config: SemanticConfig,
    nodes: Vec<HnswNode>,
    /// Entry point node index (highest layer).
    entry_point: Option<usize>,
    /// Maximum layer in the graph.
    max_level: usize,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: &SemanticConfig) -> Self {
        info!(
            "HNSW index created: dim={}, M={}, ef_construction={}, ef_search={}",
            config.embedding_dim,
            config.hnsw_m,
            config.hnsw_ef_construction,
            config.hnsw_ef_search
        );

        Self {
            config: config.clone(),
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
        }
    }

    /// Insert a document embedding into the HNSW index.
    ///
    /// Full HNSW insert algorithm (Malkov & Yashunin, 2018):
    /// 1. Assign a random level l to the new node.
    /// 2. From the entry point, greedily descend through layers max_level..=l+1
    ///    finding the closest node at each layer.
    /// 3. At layers l..=0, use ef_construction-width beam search to find the
    ///    nearest neighbors. Connect the new node bidirectionally, pruning
    ///    connections to M (or M_max=2*M at layer 0) per node.
    pub fn insert(&mut self, doc_id: u64, vector: Vec<f32>) {
        assert_eq!(
            vector.len(),
            self.config.embedding_dim,
            "Vector dimension mismatch: expected {}, got {}",
            self.config.embedding_dim,
            vector.len()
        );

        let level = self.random_level();
        let m = self.config.hnsw_m;
        let m_max0 = m * 2; // Layer 0 allows more connections.
        let ef_construction = self.config.hnsw_ef_construction;

        let node = HnswNode {
            doc_id,
            vector,
            layer_connections: vec![Vec::new(); level + 1],
            _max_layer: level,
        };

        let node_idx = self.nodes.len();
        self.nodes.push(node);

        // First node — just set as entry point.
        if node_idx == 0 {
            self.entry_point = Some(0);
            self.max_level = level;
            return;
        }

        let ep = self.entry_point.unwrap();

        // Phase 1: Greedy descent from entry point through upper layers.
        // Find the closest node at each layer above the new node's level.
        let mut current_ep = ep;
        let upper_start = self.max_level;
        let upper_end = if level < upper_start { level + 1 } else { upper_start + 1 };

        for layer in (upper_end..=upper_start).rev() {
            current_ep = self.greedy_closest(current_ep, node_idx, layer);
        }

        // Phase 2: At layers min(level, max_level)..=0, do ef_construction-width
        // beam search to find nearest neighbors and connect bidirectionally.
        let insert_top = level.min(self.max_level);

        for layer in (0..=insert_top).rev() {
            let max_connections = if layer == 0 { m_max0 } else { m };

            // Beam search for ef_construction nearest neighbors at this layer.
            let neighbors = self.search_layer(current_ep, node_idx, ef_construction, layer);

            // Select the best M connections (simple nearest-neighbor heuristic).
            let selected: Vec<usize> = neighbors
                .iter()
                .take(max_connections)
                .map(|&(idx, _)| idx)
                .collect();

            // Connect new node → selected neighbors.
            self.nodes[node_idx].layer_connections[layer] = selected.clone();

            // Connect selected neighbors → new node (bidirectional).
            for &neighbor_idx in &selected {
                if layer < self.nodes[neighbor_idx].layer_connections.len() {
                    let neighbor_conns = &mut self.nodes[neighbor_idx].layer_connections[layer];
                    neighbor_conns.push(node_idx);

                    // Prune if over capacity.
                    if neighbor_conns.len() > max_connections {
                        self.prune_connections(neighbor_idx, layer, max_connections);
                    }
                }
            }

            // Use the closest found node as EP for the next layer down.
            if let Some(&(closest, _)) = neighbors.first() {
                current_ep = closest;
            }
        }

        // Update entry point if the new node has a higher level.
        if level > self.max_level {
            self.entry_point = Some(node_idx);
            self.max_level = level;
        }

        debug!(doc_id = doc_id, level = level, "Inserted into HNSW (graph-wired)");
    }

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Full HNSW search algorithm:
    /// 1. Greedy descent from entry point through upper layers.
    /// 2. ef_search-width beam search at layer 0.
    /// 3. Return the top-k results from the beam.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ef_search = self.config.hnsw_ef_search.max(k);

        // Create a temporary query node for distance calculations.
        let _query_idx = self.nodes.len(); // Virtual index for query.

        // Phase 1: Greedy descent through upper layers.
        let mut current_ep = ep;
        for layer in (1..=self.max_level).rev() {
            current_ep = self.greedy_closest_to_query(current_ep, query, layer);
        }

        // Phase 2: ef_search-width beam search at layer 0.
        let results = self.search_layer_by_query(current_ep, query, ef_search, 0);

        // Return top-k.
        results
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (self.nodes[idx].doc_id, dist))
            .collect()
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// M4a — Expose embedding dimension for sanity checks at insert time.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// M4a — Persist (doc_id, vector) pairs to disk. Graph structure is rebuilt
    /// on load by replaying inserts. Manual binary format, magic "RAHN" v1.
    /// Layout: magic(4) ver(4) node_count(8) embedding_dim(8)
    ///         [doc_id(8) + dim*f32(4)] * node_count
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"RAHN")?;
        f.write_all(&1u32.to_le_bytes())?;
        f.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        f.write_all(&(self.config.embedding_dim as u64).to_le_bytes())?;
        for n in &self.nodes {
            f.write_all(&n.doc_id.to_le_bytes())?;
            for v in &n.vector {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        info!("HnswIndex saved: {} nodes -> {:?}", self.nodes.len(), path);
        Ok(())
    }

    /// M4a — Load from sidecar; rebuild graph by replaying inserts.
    pub fn load(path: &std::path::Path, config: &SemanticConfig) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RAHN" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "hnsw.bin: bad magic",
            ));
        }
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        f.read_exact(&mut buf4)?;
        let ver = u32::from_le_bytes(buf4);
        if ver != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("hnsw.bin: version {} (need 1)", ver),
            ));
        }
        f.read_exact(&mut buf8)?;
        let node_count = u64::from_le_bytes(buf8) as usize;
        f.read_exact(&mut buf8)?;
        let dim = u64::from_le_bytes(buf8) as usize;
        if dim != config.embedding_dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("hnsw.bin: dim {} != config {}", dim, config.embedding_dim),
            ));
        }
        let mut idx = HnswIndex::new(config);
        for _ in 0..node_count {
            f.read_exact(&mut buf8)?;
            let doc_id = u64::from_le_bytes(buf8);
            let mut vec = vec![0f32; dim];
            for slot in vec.iter_mut() {
                f.read_exact(&mut buf4)?;
                *slot = f32::from_le_bytes(buf4);
            }
            idx.insert(doc_id, vec);
        }
        info!("HnswIndex loaded: {} nodes from {:?}", node_count, path);
        Ok(idx)
    }

    // -----------------------------------------------------------------------
    // Internal HNSW methods
    // -----------------------------------------------------------------------

    /// Greedy search at a single layer: find the closest existing node to
    /// a target node (by index). Used during insert for upper-layer descent.
    fn greedy_closest(&self, ep: usize, target: usize, layer: usize) -> usize {
        let target_vec = &self.nodes[target].vector;
        let mut current = ep;
        let mut current_dist = cosine_distance(&self.nodes[current].vector, target_vec);

        loop {
            let mut improved = false;
            if layer < self.nodes[current].layer_connections.len() {
                for &neighbor in &self.nodes[current].layer_connections[layer] {
                    let d = cosine_distance(&self.nodes[neighbor].vector, target_vec);
                    if d < current_dist {
                        current = neighbor;
                        current_dist = d;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        current
    }

    /// Greedy search at a single layer using a raw query vector (not an indexed node).
    fn greedy_closest_to_query(&self, ep: usize, query: &[f32], layer: usize) -> usize {
        let mut current = ep;
        let mut current_dist = cosine_distance(&self.nodes[current].vector, query);

        loop {
            let mut improved = false;
            if layer < self.nodes[current].layer_connections.len() {
                for &neighbor in &self.nodes[current].layer_connections[layer] {
                    let d = cosine_distance(&self.nodes[neighbor].vector, query);
                    if d < current_dist {
                        current = neighbor;
                        current_dist = d;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search at a single layer: find ef nearest neighbors of the target
    /// node (by index). Returns sorted (node_idx, distance) pairs.
    fn search_layer(
        &self,
        ep: usize,
        target: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let target_vec = &self.nodes[target].vector;
        self.beam_search(ep, target_vec, ef, layer)
    }

    /// Beam search at a single layer using a raw query vector.
    fn search_layer_by_query(
        &self,
        ep: usize,
        query: &[f32],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        self.beam_search(ep, query, ef, layer)
    }

    /// Core beam search implementation.
    /// Returns up to ef nearest neighbors sorted by distance (ascending).
    fn beam_search(
        &self,
        ep: usize,
        query: &[f32],
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let ep_dist = cosine_distance(&self.nodes[ep].vector, query);

        // Candidates: min-heap by distance (explore closest first).
        let mut candidates: BinaryHeap<Reverse<OrderedFloat>> = BinaryHeap::new();
        // Results: max-heap by distance (track the ef-nearest, farthest at top).
        let mut results: BinaryHeap<OrderedFloat> = BinaryHeap::new();
        // Visited set.
        let mut visited = vec![false; self.nodes.len()];

        candidates.push(Reverse(OrderedFloat(ep_dist, ep)));
        results.push(OrderedFloat(ep_dist, ep));
        visited[ep] = true;

        while let Some(Reverse(OrderedFloat(c_dist, c_idx))) = candidates.pop() {
            // If the closest candidate is farther than the farthest result, stop.
            if let Some(&OrderedFloat(f_dist, _)) = results.peek() {
                if c_dist > f_dist && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors of c_idx at this layer.
            if layer < self.nodes[c_idx].layer_connections.len() {
                for &neighbor in &self.nodes[c_idx].layer_connections[layer] {
                    if visited[neighbor] {
                        continue;
                    }
                    visited[neighbor] = true;

                    let n_dist = cosine_distance(&self.nodes[neighbor].vector, query);

                    let should_add = if results.len() < ef {
                        true
                    } else if let Some(&OrderedFloat(f_dist, _)) = results.peek() {
                        n_dist < f_dist
                    } else {
                        true
                    };

                    if should_add {
                        candidates.push(Reverse(OrderedFloat(n_dist, neighbor)));
                        results.push(OrderedFloat(n_dist, neighbor));
                        if results.len() > ef {
                            results.pop(); // Remove the farthest.
                        }
                    }
                }
            }
        }

        // Collect results sorted by distance (ascending).
        let mut result_vec: Vec<(usize, f32)> = results
            .into_iter()
            .map(|OrderedFloat(d, idx)| (idx, d))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result_vec
    }

    /// Prune a node's connections at a layer to max_connections.
    /// Keeps the closest neighbors by distance.
    fn prune_connections(&mut self, node_idx: usize, layer: usize, max_connections: usize) {
        let node_vec = self.nodes[node_idx].vector.clone();
        let conns = &self.nodes[node_idx].layer_connections[layer];

        let mut scored: Vec<(usize, f32)> = conns
            .iter()
            .map(|&neighbor| {
                let d = cosine_distance(&self.nodes[neighbor].vector, &node_vec);
                (neighbor, d)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_connections);

        self.nodes[node_idx].layer_connections[layer] =
            scored.into_iter().map(|(idx, _)| idx).collect();
    }

    /// Generate a random level for a new node.
    /// P(level=l) = (1/M)^l, geometric distribution.
    fn random_level(&self) -> usize {
        let m = self.config.hnsw_m as f64;
        let mut level = 0;
        while rand::random::<f64>() < (1.0 / m) && level < 16 {
            level += 1;
        }
        level
    }
}

// ---------------------------------------------------------------------------
// Ordered float wrapper for BinaryHeap (f32 doesn't implement Ord)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct OrderedFloat(f32, usize);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

/// Cosine distance between two vectors: 1 - cosine_similarity.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Cosine similarity between two vectors (for ranking features).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_distance(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let config = SemanticConfig {
            embedding_dim: 4,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 50,
            pq_sub_quantizers: 16,
        };

        let mut index = HnswIndex::new(&config);
        index.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        index.insert(2, vec![0.9, 0.1, 0.0, 0.0]);
        index.insert(3, vec![0.0, 0.0, 1.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // Doc 1 = identical vector
    }

    #[test]
    fn test_hnsw_graph_wired() {
        // Verify that insert actually wires graph connections.
        let config = SemanticConfig {
            embedding_dim: 3,
            hnsw_m: 4,
            hnsw_ef_construction: 10,
            hnsw_ef_search: 10,
            pq_sub_quantizers: 16,
        };

        let mut index = HnswIndex::new(&config);
        for i in 0..20 {
            let angle = (i as f32) * std::f32::consts::PI / 10.0;
            index.insert(i as u64, vec![angle.cos(), angle.sin(), 0.0]);
        }

        // Check that at least some nodes have non-empty connections at layer 0.
        let connected: usize = index
            .nodes
            .iter()
            .filter(|n| !n.layer_connections[0].is_empty())
            .count();

        assert!(
            connected >= 15,
            "At least 15 of 20 nodes should have connections at layer 0, got {}",
            connected
        );
    }

    #[test]
    fn test_hnsw_many_nodes_recall() {
        // Insert 100 nodes, verify the search finds close neighbors.
        let config = SemanticConfig {
            embedding_dim: 4,
            hnsw_m: 8,
            hnsw_ef_construction: 32,
            hnsw_ef_search: 32,
            pq_sub_quantizers: 16,
        };

        let mut index = HnswIndex::new(&config);
        for i in 0..100 {
            let v = vec![
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.1).cos(),
                (i as f32 * 0.05).sin(),
                (i as f32 * 0.05).cos(),
            ];
            index.insert(i, v);
        }

        // Search for node 50's vector.
        let query = vec![
            (50.0_f32 * 0.1).sin(),
            (50.0_f32 * 0.1).cos(),
            (50.0_f32 * 0.05).sin(),
            (50.0_f32 * 0.05).cos(),
        ];

        let results = index.search(&query, 5);
        assert_eq!(results.len(), 5);
        // The top result should be node 50 (exact match).
        assert_eq!(results[0].0, 50);
    }
}
