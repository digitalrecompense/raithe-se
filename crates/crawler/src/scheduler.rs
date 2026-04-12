// ================================================================================
// © RAiTHE INDUSTRIES INCORPORATED 2026
// raithe-se :: crawler :: scheduler
//
// Adaptive re-crawl scheduling.
// Maintains per-URL change history (last 5 crawls).
// Fits exponential decay model to estimate change probability.
// Schedule next crawl when P(changed) crosses 0.5.
// Clamp re-crawl interval to [12 hours, 30 days].
// ================================================================================

use std::time::Duration;

/// Estimate the next re-crawl time for a URL based on its change history.
///
/// `change_intervals_hours`: the intervals (in hours) between content changes,
/// computed from the last 5 crawl timestamps and content hashes.
///
/// Returns the recommended delay before the next crawl, clamped to
/// [min_interval, max_interval].
pub fn estimate_recrawl_delay(
    change_intervals_hours: &[f64],
    min_interval_hours: u64,
    max_interval_days: u64,
) -> Duration {
    let min_hours = min_interval_hours as f64;
    let max_hours = (max_interval_days * 24) as f64;

    if change_intervals_hours.is_empty() {
        // No change history — use the midpoint.
        let default_hours = (min_hours + max_hours) / 2.0;
        return Duration::from_secs((default_hours * 3600.0) as u64);
    }

    // Fit a simple exponential decay model:
    // Mean change interval is the average of observed intervals.
    let mean_interval: f64 =
        change_intervals_hours.iter().sum::<f64>() / change_intervals_hours.len() as f64;

    // Schedule the next crawl at P(changed) = 0.5.
    // For exponential distribution: P(T <= t) = 1 - e^(-t/mean)
    // Solving for P = 0.5: t = -mean * ln(0.5) = mean * ln(2)
    let optimal_hours = mean_interval * 2.0_f64.ln();

    // Clamp to configured bounds.
    let clamped = optimal_hours.clamp(min_hours, max_hours);

    Duration::from_secs((clamped * 3600.0) as u64)
}

/// Compute the change probability for a URL given the time since last crawl
/// and the estimated mean change interval.
///
/// Uses exponential CDF: P(changed) = 1 - e^(-t / mean_interval)
pub fn change_probability(hours_since_last_crawl: f64, mean_interval_hours: f64) -> f64 {
    if mean_interval_hours <= 0.0 {
        return 1.0;
    }
    1.0 - (-hours_since_last_crawl / mean_interval_hours).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_recrawl_no_history() {
        let delay = estimate_recrawl_delay(&[], 12, 30);
        // Should be roughly the midpoint: (12 + 720) / 2 = 366 hours.
        let hours = delay.as_secs() as f64 / 3600.0;
        assert!(hours >= 12.0 && hours <= 720.0);
    }

    #[test]
    fn test_estimate_recrawl_frequent_changes() {
        // Page changes every 6 hours on average.
        let intervals = vec![6.0, 5.0, 7.0, 6.0];
        let delay = estimate_recrawl_delay(&intervals, 12, 30);
        let hours = delay.as_secs() as f64 / 3600.0;
        // Optimal would be ~4.2h (6 * ln2) but clamped to min 12h.
        assert!((hours - 12.0).abs() < 1.0);
    }

    #[test]
    fn test_change_probability() {
        // After 1 mean interval, probability should be ~0.632.
        let p = change_probability(24.0, 24.0);
        assert!((p - 0.632).abs() < 0.01);

        // After 0 time, probability should be 0.
        let p = change_probability(0.0, 24.0);
        assert!(p.abs() < 0.001);
    }
}
