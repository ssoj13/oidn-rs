//! Model picker — port of `UNetFilter::getWeights`
//! (`_ref/oidn/core/rt_filter.cpp` + `unet_filter.cpp:394-466`).

use crate::filter::Quality;

/// Identifies a model file in the `oidn-weights` archive. The inner string
/// is the file stem (e.g. `rt_hdr_alb_nrm`); use [`Self::filename`] to get
/// the full `.tza` filename.
///
/// Held as `String` rather than `&'static str` because quality-based
/// resolution may construct keys at runtime (e.g. `rt_alb` → `rt_alb_large`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelKey(pub String);

impl ModelKey {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
    pub fn name(&self) -> &str { &self.0 }
    pub fn filename(&self) -> String { format!("{}.tza", self.0) }
}

/// Pick the RT base model key for a feature combination — independent of
/// quality. Quality-based upgrading (`_large` / `_small`) happens in
/// [`quality_candidates`].
#[allow(clippy::too_many_arguments)]
pub fn select_rt(
    has_color: bool,
    has_albedo: bool,
    has_normal: bool,
    hdr: bool,
    srgb: bool,
    directional: bool,
    clean_aux: bool,
    quality: Quality,
) -> Option<ModelKey> {
    let _ = quality; // base key is independent of quality
    let base: &'static str = match (has_color, has_albedo, has_normal, hdr, srgb, directional, clean_aux) {
        (true, false, false, true,  _, false, _) => "rt_hdr",
        (true, false, false, false, true,  false, _) => "rt_ldr",
        (true, false, false, false, false, false, _) => "rt_ldr",
        (true, false, false, false, _, true, _) => "rtlightmap_dir",

        (true, true, false, true,  _, false, _) => "rt_hdr_alb",
        (true, true, false, false, _, false, _) => "rt_ldr_alb",

        (true, true, true, true,  _, false, false) => "rt_hdr_alb_nrm",
        (true, true, true, true,  _, false, true)  => "rt_hdr_calb_cnrm",
        (true, true, true, false, _, false, false) => "rt_ldr_alb_nrm",
        (true, true, true, false, _, false, true)  => "rt_ldr_calb_cnrm",

        (false, true,  false, false, _, _, _) => "rt_alb",
        (false, false, true,  false, false, _, _) => "rt_nrm",

        _ => return None,
    };
    Some(ModelKey::new(base))
}

/// Given a base model key + quality, return the list of preferred filenames
/// (stems) in order: try the first; if missing, fall back to the next.
///
/// Logic mirrors `_ref/oidn/core/unet_filter.cpp:446-459` (`getWeights`):
/// `Quality::High` prefers `_large`, falls back to base.
/// `Quality::Balanced` always uses base.
/// `Quality::Fast` prefers `_small`, falls back to base.
pub fn quality_candidates(base: &ModelKey, quality: Quality) -> Vec<String> {
    let s = base.name();
    match quality {
        Quality::High     => vec![format!("{s}_large"), s.to_string()],
        Quality::Balanced => vec![s.to_string()],
        Quality::Fast     => vec![format!("{s}_small"), s.to_string()],
    }
}
