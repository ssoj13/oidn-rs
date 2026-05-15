//! TZA weight blob resolution — in-binary lookup + optional filesystem fallback.
//!
//! Each `embed-*` Cargo feature bakes a slice of the 23 reference TZA
//! blobs from `data/weights/` into the library via `include_bytes!`.
//! A default build embeds nothing, keeping the library small; the
//! consumer chooses which subsets it wants in its binary:
//!
//! | Feature             | Stems                                            | Approx size |
//! |---------------------|--------------------------------------------------|-------------|
//! | `embed-hdr`         | `rt_hdr[_small]`, `rt_hdr_alb[_small]`,          | ~10 MB      |
//! |                     | `rt_hdr_alb_nrm[_small]`                         |             |
//! | `embed-ldr`         | `rt_ldr[_small]`, `rt_ldr_alb[_small]`,          | ~10 MB      |
//! |                     | `rt_ldr_alb_nrm[_small]`                         |             |
//! | `embed-aov`         | `rt_alb[_large]`, `rt_nrm[_large]`               | ~5 MB       |
//! | `embed-aux-clean`   | `rt_hdr_calb_cnrm[_small/_large]`,               | ~12 MB      |
//! |                     | `rt_ldr_calb_cnrm[_small]`                       |             |
//! | `embed-lightmap`    | `rtlightmap_hdr`, `rtlightmap_dir`               | ~5 MB       |
//! | `embed-all`         | everything (umbrella over the five above)        | ~48 MB      |
//!
//! Consumers that prefer fully external weights (the historical
//! behaviour) just don't enable any `embed-*` feature and pass a
//! `weights_dir` path to [`resolve`].

use std::path::Path;

use crate::filter::Quality;
use crate::registry::{ModelKey, quality_candidates};

/// Lookup a TZA blob baked into the library by a Cargo `embed-*`
/// feature. Returns `None` when the stem isn't recognised *or* when
/// the relevant feature was disabled at build time.
///
/// The returned slice is `'static` — keep it around as long as the
/// process lives. `RtFilter::builder(...).weights(...)` takes
/// `impl Into<Vec<u8>>`, so callers typically `.to_vec()` once at
/// cache fill time and reuse the owned buffer afterwards.
pub fn embedded(stem: &str) -> Option<&'static [u8]> {
    // The match arms below are gated by Cargo features. With no
    // features enabled the function always falls through to `None`.
    // `include_bytes!` paths are relative to this file
    // (`crates/oidn-rs/src/weights.rs`); the weight blobs sit at
    // `oidn-rs/data/weights/*.tza`, i.e. three levels up.
    match stem {
        // ---------- embed-hdr ----------
        #[cfg(feature = "embed-hdr")]
        "rt_hdr" => Some(include_bytes!("../../../data/weights/rt_hdr.tza")),
        #[cfg(feature = "embed-hdr")]
        "rt_hdr_small" => Some(include_bytes!("../../../data/weights/rt_hdr_small.tza")),
        #[cfg(feature = "embed-hdr")]
        "rt_hdr_alb" => Some(include_bytes!("../../../data/weights/rt_hdr_alb.tza")),
        #[cfg(feature = "embed-hdr")]
        "rt_hdr_alb_small" => Some(include_bytes!("../../../data/weights/rt_hdr_alb_small.tza")),
        #[cfg(feature = "embed-hdr")]
        "rt_hdr_alb_nrm" => Some(include_bytes!("../../../data/weights/rt_hdr_alb_nrm.tza")),
        #[cfg(feature = "embed-hdr")]
        "rt_hdr_alb_nrm_small" => Some(include_bytes!(
            "../../../data/weights/rt_hdr_alb_nrm_small.tza"
        )),

        // ---------- embed-ldr ----------
        #[cfg(feature = "embed-ldr")]
        "rt_ldr" => Some(include_bytes!("../../../data/weights/rt_ldr.tza")),
        #[cfg(feature = "embed-ldr")]
        "rt_ldr_small" => Some(include_bytes!("../../../data/weights/rt_ldr_small.tza")),
        #[cfg(feature = "embed-ldr")]
        "rt_ldr_alb" => Some(include_bytes!("../../../data/weights/rt_ldr_alb.tza")),
        #[cfg(feature = "embed-ldr")]
        "rt_ldr_alb_small" => Some(include_bytes!("../../../data/weights/rt_ldr_alb_small.tza")),
        #[cfg(feature = "embed-ldr")]
        "rt_ldr_alb_nrm" => Some(include_bytes!("../../../data/weights/rt_ldr_alb_nrm.tza")),
        #[cfg(feature = "embed-ldr")]
        "rt_ldr_alb_nrm_small" => Some(include_bytes!(
            "../../../data/weights/rt_ldr_alb_nrm_small.tza"
        )),

        // ---------- embed-aov ----------
        #[cfg(feature = "embed-aov")]
        "rt_alb" => Some(include_bytes!("../../../data/weights/rt_alb.tza")),
        #[cfg(feature = "embed-aov")]
        "rt_alb_large" => Some(include_bytes!("../../../data/weights/rt_alb_large.tza")),
        #[cfg(feature = "embed-aov")]
        "rt_nrm" => Some(include_bytes!("../../../data/weights/rt_nrm.tza")),
        #[cfg(feature = "embed-aov")]
        "rt_nrm_large" => Some(include_bytes!("../../../data/weights/rt_nrm_large.tza")),

        // ---------- embed-aux-clean ----------
        #[cfg(feature = "embed-aux-clean")]
        "rt_hdr_calb_cnrm" => Some(include_bytes!("../../../data/weights/rt_hdr_calb_cnrm.tza")),
        #[cfg(feature = "embed-aux-clean")]
        "rt_hdr_calb_cnrm_small" => Some(include_bytes!(
            "../../../data/weights/rt_hdr_calb_cnrm_small.tza"
        )),
        #[cfg(feature = "embed-aux-clean")]
        "rt_hdr_calb_cnrm_large" => Some(include_bytes!(
            "../../../data/weights/rt_hdr_calb_cnrm_large.tza"
        )),
        #[cfg(feature = "embed-aux-clean")]
        "rt_ldr_calb_cnrm" => Some(include_bytes!("../../../data/weights/rt_ldr_calb_cnrm.tza")),
        #[cfg(feature = "embed-aux-clean")]
        "rt_ldr_calb_cnrm_small" => Some(include_bytes!(
            "../../../data/weights/rt_ldr_calb_cnrm_small.tza"
        )),

        // ---------- embed-lightmap ----------
        #[cfg(feature = "embed-lightmap")]
        "rtlightmap_hdr" => Some(include_bytes!("../../../data/weights/rtlightmap_hdr.tza")),
        #[cfg(feature = "embed-lightmap")]
        "rtlightmap_dir" => Some(include_bytes!("../../../data/weights/rtlightmap_dir.tza")),

        _ => None,
    }
}

/// Walk quality candidate stems and return the first weight blob we
/// can produce, trying embedded `include_bytes!` first and then
/// (optionally) reading `fallback_dir.join("{stem}.tza")` from disk.
///
/// Designed to replace the per-consumer candidate loops:
///
/// ```ignore
/// let base_key = oidn_rs::registry::select_rt(
///     /* has_color */ true, has_alb, has_nrm, hdr, srgb,
///     directional, clean_aux, quality,
/// )?;
/// let (stem, bytes) = oidn_rs::weights::resolve(
///     &base_key, quality, Some(weights_dir.as_path()),
/// )?;
/// let filter = RtFilter::<B>::builder(device, weights_dir)
///     .weights(bytes)
///     .build();
/// ```
///
/// Returns `None` only when *all* candidates fail — neither embedded
/// nor any provided filesystem path holds a matching blob.
pub fn resolve(
    base_key: &ModelKey,
    quality: Quality,
    fallback_dir: Option<&Path>,
) -> Option<(String, Vec<u8>)> {
    for stem in quality_candidates(base_key, quality) {
        if let Some(bytes) = embedded(&stem) {
            return Some((stem, bytes.to_vec()));
        }
        if let Some(dir) = fallback_dir {
            let path = dir.join(format!("{stem}.tza"));
            if let Ok(bytes) = std::fs::read(&path) {
                return Some((stem, bytes));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: any feature-gated stem we declare must resolve to bytes
    /// in the builds that enable it, and the blob must be non-empty
    /// (TZA headers are at least a few hundred bytes — a 0-byte slice
    /// indicates a broken `include_bytes!` path).
    #[test]
    fn embedded_blobs_are_non_empty_when_features_enabled() {
        // Cheap stems to spot-check from each feature gate.
        let stems_under_test: &[&str] = &[
            #[cfg(feature = "embed-hdr")] "rt_hdr",
            #[cfg(feature = "embed-ldr")] "rt_ldr",
            #[cfg(feature = "embed-aov")] "rt_alb",
            #[cfg(feature = "embed-aux-clean")] "rt_hdr_calb_cnrm",
            #[cfg(feature = "embed-lightmap")] "rtlightmap_hdr",
        ];
        for stem in stems_under_test {
            let bytes = embedded(stem)
                .unwrap_or_else(|| panic!("expected embedded weight for stem `{stem}`"));
            assert!(
                !bytes.is_empty() && bytes.len() > 256,
                "embedded `{stem}` looks suspiciously small ({} bytes)",
                bytes.len(),
            );
        }
    }

    /// Without any `embed-*` feature the lookup must return None for
    /// every known stem. (Tested implicitly by the build matrix; here
    /// we just check an unknown stem always returns None regardless of
    /// the feature flags so the fallthrough arm is exercised.)
    #[test]
    fn embedded_returns_none_for_unknown_stem() {
        assert!(embedded("not_a_real_stem").is_none());
        assert!(embedded("").is_none());
    }
}
