// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler :: bandwidth
//
// Bandwidth-aware throttling.
// 20 Mbps upload constraint — serving takes priority over crawling.
// Pause crawler when upload utilization exceeds 50% (10 Mbps).
// Resume when utilization drops below 30% (6 Mbps).
// ================================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Global bandwidth monitor tracking upload usage per second.
pub struct BandwidthMonitor {
    /// Current upload bytes per second (updated by the serving layer).
    upload_bytes_per_sec: Arc<AtomicU64>,
    /// Upload bandwidth capacity in bytes per second (~20 Mbps = 2.5 MB/s).
    capacity_bytes_per_sec: u64,
    /// Pause threshold as percentage (default: 50).
    pause_threshold_pct: u8,
    /// Resume threshold as percentage (default: 30).
    resume_threshold_pct: u8,
    /// Whether the crawler is currently paused.
    is_paused: std::sync::atomic::AtomicBool,
}

impl BandwidthMonitor {
    /// Create a new bandwidth monitor.
    /// Assumes 20 Mbps (~2.5 MB/s) upload capacity.
    pub fn new(pause_threshold_pct: u8, resume_threshold_pct: u8) -> Self {
        Self {
            upload_bytes_per_sec: Arc::new(AtomicU64::new(0)),
            capacity_bytes_per_sec: 2_500_000, // 20 Mbps
            pause_threshold_pct,
            resume_threshold_pct,
            is_paused: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Report current upload usage (called periodically by the serving layer).
    pub fn report_upload_bytes_per_sec(&self, bytes: u64) {
        self.upload_bytes_per_sec.store(bytes, Ordering::Relaxed);
    }

    /// Get a handle to the upload counter for the serving layer to update.
    pub fn upload_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.upload_bytes_per_sec)
    }

    /// Check if the crawler should pause based on current bandwidth usage.
    /// Implements hysteresis: pause at 50%, resume at 30%.
    pub fn should_pause(&self) -> bool {
        let current = self.upload_bytes_per_sec.load(Ordering::Relaxed);
        let utilization_pct =
            ((current as f64 / self.capacity_bytes_per_sec as f64) * 100.0) as u8;

        let currently_paused = self.is_paused.load(Ordering::Relaxed);

        if currently_paused {
            // Resume only when below resume threshold.
            if utilization_pct < self.resume_threshold_pct {
                self.is_paused.store(false, Ordering::Relaxed);
                false
            } else {
                true
            }
        } else {
            // Pause when above pause threshold.
            if utilization_pct >= self.pause_threshold_pct {
                self.is_paused.store(true, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
    }

    /// Get current utilization as a percentage.
    pub fn utilization_pct(&self) -> f64 {
        let current = self.upload_bytes_per_sec.load(Ordering::Relaxed);
        (current as f64 / self.capacity_bytes_per_sec as f64) * 100.0
    }
}
