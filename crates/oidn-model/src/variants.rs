/// UNet architecture variants supported by the Intel OIDN runtime.
///
/// `Base` / `Small` share the `UNet` topology (`model.py:UNet`).
/// `Large` / `XLarge` use the distinct `UNetLarge` topology with twice as
/// many convolutions per stage and wider channels (`model.py:UNetLarge`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Default UNet — `rt_hdr`, `rt_hdr_alb`, `rt_hdr_alb_nrm`, `rt_ldr*`,
    /// `rt_alb`, `rt_nrm`, lightmap base models. ~1.8 MB weights.
    Base,
    /// Compact UNet — `*_small` model variants. ~620 KB weights.
    Small,
    /// Large UNet (different topology) — `rt_alb_large`, `rt_nrm_large`,
    /// `rt_hdr_calb_cnrm_large`. ~7.4 MB weights.
    Large,
    /// Extra-large UNet (same topology as Large, wider channels). Reserved
    /// for future Intel weight releases — no shipped `.tza` exists yet.
    XLarge,
}

impl Variant {
    /// Detect the variant from a TZA tensor map.
    ///
    /// Logic mirrors `_ref/oidn/core/unet_filter.cpp:263`:
    /// presence of `enc_conv1b.weight` ⇒ `Large` (or XL — same topology,
    /// channel widths inferred from shapes).
    pub fn from_tensor_names<I: IntoIterator<Item = S>, S: AsRef<str>>(names: I) -> Self {
        for n in names {
            if n.as_ref() == "enc_conv1b.weight" { return Variant::Large; }
        }
        Variant::Base
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelConfig {
    pub ec1: usize,
    pub ec2: usize,
    pub ec3: usize,
    pub ec4: usize,
    pub ec5: usize,
    pub dc4: usize,
    pub dc3: usize,
    pub dc2a: usize,
    pub dc2b: usize,
    pub dc1a: usize,
    pub dc1b: usize,
}

impl ChannelConfig {
    /// Channel widths for `UNet` (base / small). Panics if called with a
    /// Large variant — callers should branch on `Variant` first.
    pub const fn for_variant(v: Variant) -> Self {
        match v {
            Variant::Base => Self {
                ec1: 32, ec2: 48, ec3: 64, ec4: 80, ec5: 96,
                dc4: 112, dc3: 96, dc2a: 64, dc2b: 64, dc1a: 64, dc1b: 32,
            },
            Variant::Small => Self {
                ec1: 32, ec2: 32, ec3: 32, ec4: 32, ec5: 32,
                dc4: 64, dc3: 64, dc2a: 64, dc2b: 32, dc1a: 32, dc1b: 32,
            },
            Variant::Large | Variant::XLarge => {
                panic!("ChannelConfig::for_variant called with Large/XLarge — use UNetLarge instead")
            }
        }
    }
}
