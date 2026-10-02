//! Validated U-Net architecture inferred from archive tensors.
//! The native graph also derives channels from weights (LOCAL2.5 core/graph.cpp).
use crate::{ChannelConfig, ChannelConfigLarge, LoadError, Net, UNet, UNetLarge, Variant};
use burn::tensor::Device;
use oidn_tza::{Layout, TensorMap};

/// Spatial geometry of the supported native OIDN topologies.
pub const RECEPTIVE_FIELD_BASE: i32 = 174;
pub const RECEPTIVE_FIELD_LARGE: i32 = 202;
pub const MIN_TILE_ALIGNMENT: i32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channels {
    Base(ChannelConfig),
    Large(ChannelConfigLarge),
}

/// Complete validated model geometry. Filenames do not determine executable widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDescriptor {
    variant: Variant,
    input: usize,
    output: usize,
    channels: Channels,
    parameters_bytes: usize,
}

impl ModelDescriptor {
    /// Validate every parameter and channel edge before any model/device allocation.
    pub fn from_tza(map: &TensorMap) -> Result<Self, LoadError> {
        let large = map.contains_key("enc_conv1b.weight");
        let first = if large { "enc_conv1a" } else { "enc_conv0" };
        let last = if large { "dec_conv1c" } else { "dec_conv0" };
        let input = kernel(map, first)?[1];
        let output = kernel(map, last)?[0];
        let channels = if large {
            Channels::Large(ChannelConfigLarge {
                ec1: kernel(map, "enc_conv1a")?[0],
                ec2: kernel(map, "enc_conv2a")?[0],
                ec3: kernel(map, "enc_conv3a")?[0],
                ec4: kernel(map, "enc_conv4a")?[0],
                ec5: kernel(map, "enc_conv5a")?[0],
                dc4: kernel(map, "dec_conv4a")?[0],
                dc3: kernel(map, "dec_conv3a")?[0],
                dc2: kernel(map, "dec_conv2a")?[0],
                dc1: kernel(map, "dec_conv1a")?[0],
            })
        } else {
            Channels::Base(ChannelConfig {
                ec1: kernel(map, "enc_conv0")?[0],
                ec2: kernel(map, "enc_conv2")?[0],
                ec3: kernel(map, "enc_conv3")?[0],
                ec4: kernel(map, "enc_conv4")?[0],
                ec5: kernel(map, "enc_conv5a")?[0],
                dc4: kernel(map, "dec_conv4a")?[0],
                dc3: kernel(map, "dec_conv3a")?[0],
                dc2a: kernel(map, "dec_conv2a")?[0],
                dc2b: kernel(map, "dec_conv2b")?[0],
                dc1a: kernel(map, "dec_conv1a")?[0],
                dc1b: kernel(map, "dec_conv1b")?[0],
            })
        };
        let variant = match channels {
            Channels::Base(c) if c == ChannelConfig::for_variant(Variant::Small) => Variant::Small,
            Channels::Base(_) => Variant::Base,
            Channels::Large(c) if c == ChannelConfigLarge::XL => Variant::XLarge,
            Channels::Large(_) => Variant::Large,
        };
        let parameters_bytes = map.values().try_fold(0usize, |sum, tensor| {
            let bytes = tensor
                .desc
                .num_elements()?
                .checked_mul(4)
                .ok_or(oidn_tza::TzaError::SizeOverflow)?;
            sum.checked_add(bytes)
                .ok_or(oidn_tza::TzaError::SizeOverflow)
        })?;
        let desc = Self {
            variant,
            input,
            output,
            channels,
            parameters_bytes,
        };
        desc.validate(map)?;
        Ok(desc)
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }
    pub fn in_channels(&self) -> usize {
        self.input
    }
    pub fn out_channels(&self) -> usize {
        self.output
    }
    pub fn receptive_field(&self) -> i32 {
        match self.channels {
            Channels::Base(_) => RECEPTIVE_FIELD_BASE,
            Channels::Large(_) => RECEPTIVE_FIELD_LARGE,
        }
    }
    pub fn alignment(&self) -> i32 {
        MIN_TILE_ALIGNMENT
    }

    /// Logical f32 parameter payload size, excluding device packing/allocator overhead.
    pub fn parameters_bytes(&self) -> usize {
        self.parameters_bytes
    }

    /// Conservative logical activation storage per aligned tile pixel.
    ///
    /// Sums layer input/output tensors at their spatial resolutions, even where
    /// lifetimes do not overlap. Includes input packing and output storage, but
    /// excludes convolution workspaces, allocator overhead and whole-image buffers.
    /// This is a planning estimate, not a backend memory limit.
    pub fn activation_bytes_per_pixel(&self) -> usize {
        let divisors: &[u64] = match self.channels {
            Channels::Base(_) => &[1, 1, 4, 16, 64, 256, 256, 64, 64, 16, 16, 4, 4, 1, 1, 1],
            Channels::Large(_) => &[
                1, 1, 4, 4, 16, 16, 64, 64, 256, 256, 64, 64, 16, 16, 4, 4, 1, 1, 1,
            ],
        };
        // Validated channel sums fit usize; u64 arithmetic also covers 32-bit targets.
        let mut channels = 2 * self.input as u64 + self.output as u64;
        for ((_, input, output), divisor) in self
            .layers()
            .expect("validated channel sums")
            .iter()
            .zip(divisors)
        {
            channels += (*input as u64 + *output as u64).div_ceil(*divisor);
        }
        usize::try_from(channels * 4).unwrap_or(usize::MAX)
    }

    /// Construct and populate the topology from the same validated descriptor.
    /// Revalidate because TensorMap and its public payloads may have changed.
    pub fn load(&self, map: &TensorMap, device: &Device) -> Result<Net, LoadError> {
        if Self::from_tza(map)? != *self {
            return Err(LoadError::InvalidSchema(
                "archive no longer matches model descriptor".into(),
            ));
        }
        match self.channels {
            Channels::Base(c) => Ok(Net::Base(
                UNet::new_with(self.input, self.output, c, device).load(map, device)?,
            )),
            Channels::Large(c) => Ok(Net::Large(
                UNetLarge::new_with(self.input, self.output, c, device).load(map, device)?,
            )),
        }
    }

    fn layers(&self) -> Result<Vec<(&'static str, usize, usize)>, LoadError> {
        Ok(match self.channels {
            Channels::Base(c) => vec![
                ("enc_conv0", self.input, c.ec1),
                ("enc_conv1", c.ec1, c.ec1),
                ("enc_conv2", c.ec1, c.ec2),
                ("enc_conv3", c.ec2, c.ec3),
                ("enc_conv4", c.ec3, c.ec4),
                ("enc_conv5a", c.ec4, c.ec5),
                ("enc_conv5b", c.ec5, c.ec5),
                ("dec_conv4a", add(c.ec5, c.ec3)?, c.dc4),
                ("dec_conv4b", c.dc4, c.dc4),
                ("dec_conv3a", add(c.dc4, c.ec2)?, c.dc3),
                ("dec_conv3b", c.dc3, c.dc3),
                ("dec_conv2a", add(c.dc3, c.ec1)?, c.dc2a),
                ("dec_conv2b", c.dc2a, c.dc2b),
                ("dec_conv1a", add(c.dc2b, self.input)?, c.dc1a),
                ("dec_conv1b", c.dc1a, c.dc1b),
                ("dec_conv0", c.dc1b, self.output),
            ],
            Channels::Large(c) => vec![
                ("enc_conv1a", self.input, c.ec1),
                ("enc_conv1b", c.ec1, c.ec1),
                ("enc_conv2a", c.ec1, c.ec2),
                ("enc_conv2b", c.ec2, c.ec2),
                ("enc_conv3a", c.ec2, c.ec3),
                ("enc_conv3b", c.ec3, c.ec3),
                ("enc_conv4a", c.ec3, c.ec4),
                ("enc_conv4b", c.ec4, c.ec4),
                ("enc_conv5a", c.ec4, c.ec5),
                ("enc_conv5b", c.ec5, c.ec5),
                ("dec_conv4a", add(c.ec5, c.ec3)?, c.dc4),
                ("dec_conv4b", c.dc4, c.dc4),
                ("dec_conv3a", add(c.dc4, c.ec2)?, c.dc3),
                ("dec_conv3b", c.dc3, c.dc3),
                ("dec_conv2a", add(c.dc3, c.ec1)?, c.dc2),
                ("dec_conv2b", c.dc2, c.dc2),
                ("dec_conv1a", add(c.dc2, self.input)?, c.dc1),
                ("dec_conv1b", c.dc1, c.dc1),
                ("dec_conv1c", c.dc1, self.output),
            ],
        })
    }

    fn validate(&self, map: &TensorMap) -> Result<(), LoadError> {
        let edges = self.layers()?;
        // All expected pairs are mandatory; unknown layers/parameters cannot be silently ignored.
        for &(layer, input, output) in &edges {
            let name = format!("{layer}.weight");
            let got = kernel(map, layer)?;
            if got != [output, input, 3, 3] {
                return Err(LoadError::ShapeMismatch {
                    name,
                    expected: vec![output, input, 3, 3],
                    got: map[&format!("{layer}.weight")].desc.dims.clone(),
                });
            }
            let bias_name = format!("{layer}.bias");
            let bias = map
                .get(&bias_name)
                .ok_or_else(|| LoadError::MissingTensor(bias_name.clone()))?;
            if bias.desc.layout != Layout::X {
                return Err(LoadError::BadLayout {
                    name: bias_name,
                    expected: "x",
                    got: bias.desc.layout,
                });
            }
            if bias.desc.dims != [output as u32] {
                return Err(LoadError::ShapeMismatch {
                    name: bias_name,
                    expected: vec![output],
                    got: bias.desc.dims.clone(),
                });
            }
        }
        if map.len() != edges.len() * 2 {
            let unexpected = map.keys().find(|name| {
                !edges.iter().any(|(layer, _, _)| {
                    **name == format!("{layer}.weight") || **name == format!("{layer}.bias")
                })
            });
            return Err(LoadError::InvalidSchema(format!(
                "unexpected parameter: {unexpected:?}"
            )));
        }
        for (name, tensor) in map {
            if tensor.iter_f32()?.any(|v| !v.is_finite()) {
                return Err(LoadError::NonFiniteTensor(name.clone()));
            }
        }
        Ok(())
    }
}

fn kernel(map: &TensorMap, layer: &str) -> Result<[usize; 4], LoadError> {
    let name = format!("{layer}.weight");
    let tensor = map
        .get(&name)
        .ok_or_else(|| LoadError::MissingTensor(name.clone()))?;
    if tensor.desc.layout != Layout::Oihw {
        return Err(LoadError::BadLayout {
            name,
            expected: "oihw",
            got: tensor.desc.layout,
        });
    }
    tensor.validate()?;
    let dims = &tensor.desc.dims;
    Ok([
        dims[0] as usize,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] as usize,
    ])
}

fn add(a: usize, b: usize) -> Result<usize, LoadError> {
    a.checked_add(b)
        .ok_or_else(|| LoadError::InvalidSchema("channel sum overflow".into()))
}
