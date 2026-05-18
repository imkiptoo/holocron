//! An embedding cache decorator.
//!
//! [`CachingEmbedder`] wraps any [`Embedder`] and memoizes exact-match
//! `text -> embedding` results, so repeated questions (and repeated training
//! text) skip the network round-trip. This is distinct from the engine's
//! *semantic* SQL cache — it only avoids re-embedding byte-identical input.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::Result;
use crate::traits::Embedder;

/// Wraps an [`Embedder`] with a bounded, dependency-free approximate-LRU cache.
///
/// The cache is segmented into two generations: lookups check the "hot" map
/// then the "cold" one (promoting on hit); when "hot" fills to `capacity` it is
/// demoted to "cold" and a fresh "hot" begins. Memory stays bounded at roughly
/// `2 * capacity` entries without needing a linked-list LRU.
pub struct CachingEmbedder<E> {
    inner: E,
    capacity: usize,
    cells: Mutex<Segments>,
}

#[derive(Default)]
struct Segments {
    hot: HashMap<String, Vec<f32>>,
    cold: HashMap<String, Vec<f32>>,
}

impl<E> CachingEmbedder<E> {
    /// Wrap `inner`, caching up to ~`2 * capacity` distinct embeddings.
    pub fn new(inner: E, capacity: usize) -> Self {
        Self { inner, capacity: capacity.max(1), cells: Mutex::new(Segments::default()) }
    }

    fn get(&self, text: &str) -> Option<Vec<f32>> {
        let mut seg = self.cells.lock().unwrap();
        if let Some(v) = seg.hot.get(text) {
            return Some(v.clone());
        }
        // Promote a cold hit into the hot generation.
        if let Some(v) = seg.cold.remove(text) {
            seg.hot.insert(text.to_string(), v.clone());
            return Some(v);
        }
        None
    }

    fn put(&self, text: &str, embedding: &[f32]) {
        let mut seg = self.cells.lock().unwrap();
        if seg.hot.len() >= self.capacity {
            // Demote the current hot generation; drop the previous cold one.
            let demoted = std::mem::take(&mut seg.hot);
            seg.cold = demoted;
        }
        seg.hot.insert(text.to_string(), embedding.to_vec());
    }
}

#[async_trait]
impl<E: Embedder> Embedder for CachingEmbedder<E> {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(hit) = self.get(text) {
            return Ok(hit);
        }
        let emb = self.inner.embed(text).await?;
        self.put(text, &emb);
        Ok(emb)
    }

    fn dims(&self) -> usize {
        self.inner.dims()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// An embedder that counts calls and returns the text length as a 1-vector.
    struct Counting {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Embedder for Counting {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(vec![text.len() as f32])
        }
        fn dims(&self) -> usize {
            1
        }
    }

    #[tokio::test]
    async fn caches_repeated_text() {
        let inner = Counting { calls: AtomicUsize::new(0) };
        let emb = CachingEmbedder::new(inner, 8);

        assert_eq!(emb.embed("hello").await.unwrap(), vec![5.0]);
        assert_eq!(emb.embed("hello").await.unwrap(), vec![5.0]);
        assert_eq!(emb.embed("hello").await.unwrap(), vec![5.0]);
        // Only one call reached the inner embedder.
        assert_eq!(emb.inner.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn survives_generation_rollover_via_cold_segment() {
        let inner = Counting { calls: AtomicUsize::new(0) };
        let emb = CachingEmbedder::new(inner, 1); // hot holds 1, cold holds 1

        emb.embed("a").await.unwrap(); // hot={a}
        emb.embed("b").await.unwrap(); // demote hot->cold={a}, hot={b}
        // "a" is now in the cold segment and should still hit (no new call).
        let before = emb.inner.calls.load(Ordering::Relaxed);
        emb.embed("a").await.unwrap();
        assert_eq!(emb.inner.calls.load(Ordering::Relaxed), before);
    }
}
