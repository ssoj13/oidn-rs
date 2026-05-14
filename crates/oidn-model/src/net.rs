//! `Net` — backend-generic enum over the two U-Net topologies.
//!
//! Lets the runtime call `forward` without caring whether the loaded weights
//! follow the base/small `UNet` topology or the larger `UNetLarge` topology.

use burn::tensor::{Tensor, backend::Backend};

use crate::{UNet, UNetLarge};

// UNet and UNetLarge differ significantly in size (Large has 19 vs 16 convs
// with wider channels). Boxing one variant would force an extra heap alloc on
// every commit. We hold exactly one filter alive at a time, so the size delta
// is irrelevant in practice.
#[allow(clippy::large_enum_variant)]
pub enum Net<B: Backend> {
    Base(UNet<B>),
    Large(UNetLarge<B>),
}

impl<B: Backend> Net<B> {
    pub fn in_channels(&self) -> usize {
        match self {
            Net::Base(u)  => u.in_channels(),
            Net::Large(u) => u.in_channels(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Net::Base(u)  => u.forward(x),
            Net::Large(u) => u.forward(x),
        }
    }
}

impl<B: Backend> std::fmt::Debug for Net<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Net::Base(_)  => write!(f, "Net::Base(UNet)"),
            Net::Large(_) => write!(f, "Net::Large(UNetLarge)"),
        }
    }
}
