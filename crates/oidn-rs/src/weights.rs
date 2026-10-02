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

/// Which storage sources may satisfy a candidate, in priority order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SourcePolicy {
    #[default]
    EmbeddedFirst,
    DiskFirst,
    EmbeddedOnly,
    DiskOnly,
}

/// Provenance of resolved bytes; filesystem errors are never treated as absence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightSource {
    Embedded,
    File(std::path::PathBuf),
}

/// Selected identity, bytes and provenance travel together until schema validation.
#[derive(Debug, Clone)]
pub struct ResolvedWeights {
    pub stem: String,
    pub bytes: Vec<u8>,
    pub source: WeightSource,
}

/// Resolve sources in explicit priority order, then quality candidates within each source.
/// Returns `Ok(None)` only when all allowed sources are absent. Only
/// filesystem `NotFound` permits fallback; all other I/O errors propagate.
pub fn resolve(
    base_key: &ModelKey,
    quality: Quality,
    fallback_dir: Option<&Path>,
    policy: SourcePolicy,
) -> Result<Option<ResolvedWeights>, crate::error::OidnError> {
    let candidates = quality_candidates(base_key, quality);
    let sources: &[bool] = match policy {
        SourcePolicy::EmbeddedFirst => &[true, false],
        SourcePolicy::DiskFirst => &[false, true],
        SourcePolicy::EmbeddedOnly => &[true],
        SourcePolicy::DiskOnly => &[false],
    };
    // Source precedence dominates quality fallback: an explicit directory's
    // base model wins over an embedded Large when disk-first was requested.
    for &is_embedded in sources {
        for stem in candidates.iter().cloned() {
            if is_embedded {
                if let Some(bytes) = embedded(&stem) {
                    return Ok(Some(ResolvedWeights {
                        stem,
                        bytes: bytes.to_vec(),
                        source: WeightSource::Embedded,
                    }));
                }
            } else if let Some(dir) = fallback_dir {
                let path = dir.join(format!("{stem}.tza"));
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        return Ok(Some(ResolvedWeights {
                            stem,
                            bytes,
                            source: WeightSource::File(path),
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("oidn-resolver-{}-{nonce}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            for name in ["rt_hdr.tza", "rt_alb.tza"] {
                let file = self.0.join(name);
                if file.is_dir() {
                    let _ = std::fs::remove_dir(file);
                } else if file.is_file() {
                    let _ = std::fs::remove_file(file);
                }
            }
            let _ = std::fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn disk_only_absence_and_io_errors_are_distinct() {
        let fixture = Fixture::new();
        let key = ModelKey::new("rt_hdr");
        assert!(
            resolve(
                &key,
                Quality::Balanced,
                Some(&fixture.0),
                SourcePolicy::DiskOnly
            )
            .unwrap()
            .is_none()
        );
        std::fs::create_dir(fixture.0.join("rt_hdr.tza")).unwrap();
        assert!(matches!(
            resolve(
                &key,
                Quality::Balanced,
                Some(&fixture.0),
                SourcePolicy::DiskOnly
            ),
            Err(crate::error::OidnError::Io(_))
        ));
    }

    #[cfg(feature = "embed-hdr")]
    #[test]
    fn explicit_directory_content_wins_same_stem_conflict() {
        let fixture = Fixture::new();
        std::fs::write(fixture.0.join("rt_hdr.tza"), b"caller disk bytes").unwrap();
        let key = ModelKey::new("rt_hdr");
        let disk = resolve(
            &key,
            Quality::Balanced,
            Some(&fixture.0),
            SourcePolicy::DiskFirst,
        )
        .unwrap()
        .unwrap();
        assert_eq!(disk.bytes, b"caller disk bytes");
        assert!(matches!(disk.source, WeightSource::File(_)));
        let embedded = resolve(
            &key,
            Quality::Balanced,
            Some(&fixture.0),
            SourcePolicy::EmbeddedFirst,
        )
        .unwrap()
        .unwrap();
        assert_eq!(embedded.bytes, super::embedded("rt_hdr").unwrap());
        assert_eq!(embedded.source, WeightSource::Embedded);
    }

    #[cfg(feature = "embed-aov")]
    #[test]
    fn source_priority_precedes_quality_fallback() {
        let fixture = Fixture::new();
        std::fs::write(fixture.0.join("rt_alb.tza"), b"caller base").unwrap();
        let key = ModelKey::new("rt_alb");
        let disk = resolve(
            &key,
            Quality::High,
            Some(&fixture.0),
            SourcePolicy::DiskFirst,
        )
        .unwrap()
        .unwrap();
        assert_eq!(disk.stem, "rt_alb");
        assert_eq!(disk.bytes, b"caller base");
        let embedded = resolve(
            &key,
            Quality::High,
            Some(&fixture.0),
            SourcePolicy::EmbeddedFirst,
        )
        .unwrap()
        .unwrap();
        assert_eq!(embedded.stem, "rt_alb_large");
    }

    /// Sanity: any feature-gated stem we declare must resolve to bytes
    /// in the builds that enable it, and the blob must be non-empty
    /// (TZA headers are at least a few hundred bytes — a 0-byte slice
    /// indicates a broken `include_bytes!` path).
    #[test]
    fn embedded_blobs_are_non_empty_when_features_enabled() {
        // Cheap stems to spot-check from each feature gate.
        let stems_under_test: &[&str] = &[
            #[cfg(feature = "embed-hdr")]
            "rt_hdr",
            #[cfg(feature = "embed-ldr")]
            "rt_ldr",
            #[cfg(feature = "embed-aov")]
            "rt_alb",
            #[cfg(feature = "embed-aux-clean")]
            "rt_hdr_calb_cnrm",
            #[cfg(feature = "embed-lightmap")]
            "rtlightmap_hdr",
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
