//! U-Net network — direct port of `_ref/oidn/training/model.py:UNet`.

use burn::{
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{
        Device,
        Tensor,
        activation::relu,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use crate::variants::{ChannelConfig, Variant};

/// 3×3 convolution with padding=1 on all sides (matches `Conv` in Python
/// reference, which uses `nn.Conv2d(..., 3, padding=1)`).
fn conv3(in_ch: usize, out_ch: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([in_ch, out_ch], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(true)
        .init(device)
}

/// U-Net denoiser network. Sixteen 3×3 convolutions, four pool/upsample
/// stages, four skip-connection concatenations.
///
/// Topology mirrors `model.py:UNet.forward` exactly (same op order, same
/// channel widths). The two extra `concat(x, input)` skip means the input is
/// re-injected at the highest decoder level.
#[derive(Module, Debug)]
pub struct UNet {
    pub enc_conv0: Conv2d,
    pub enc_conv1: Conv2d,
    pub enc_conv2: Conv2d,
    pub enc_conv3: Conv2d,
    pub enc_conv4: Conv2d,
    pub enc_conv5a: Conv2d,
    pub enc_conv5b: Conv2d,
    pub dec_conv4a: Conv2d,
    pub dec_conv4b: Conv2d,
    pub dec_conv3a: Conv2d,
    pub dec_conv3b: Conv2d,
    pub dec_conv2a: Conv2d,
    pub dec_conv2b: Conv2d,
    pub dec_conv1a: Conv2d,
    pub dec_conv1b: Conv2d,
    pub dec_conv0: Conv2d,
    pub pool: MaxPool2d,
    pub in_channels: usize,
}

impl UNet {
    /// Construct a U-Net with the given variant, input/output channel counts.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        variant: Variant,
        device: &Device,
    ) -> Self {
        let c = ChannelConfig::for_variant(variant);
        let ic = in_channels;
        let oc = out_channels;

        Self {
            enc_conv0: conv3(ic, c.ec1, device),
            enc_conv1: conv3(c.ec1, c.ec1, device),
            enc_conv2: conv3(c.ec1, c.ec2, device),
            enc_conv3: conv3(c.ec2, c.ec3, device),
            enc_conv4: conv3(c.ec3, c.ec4, device),
            enc_conv5a: conv3(c.ec4, c.ec5, device),
            enc_conv5b: conv3(c.ec5, c.ec5, device),
            dec_conv4a: conv3(c.ec5 + c.ec3, c.dc4, device),
            dec_conv4b: conv3(c.dc4, c.dc4, device),
            dec_conv3a: conv3(c.dc4 + c.ec2, c.dc3, device),
            dec_conv3b: conv3(c.dc3, c.dc3, device),
            dec_conv2a: conv3(c.dc3 + c.ec1, c.dc2a, device),
            dec_conv2b: conv3(c.dc2a, c.dc2b, device),
            dec_conv1a: conv3(c.dc2b + ic, c.dc1a, device),
            dec_conv1b: conv3(c.dc1a, c.dc1b, device),
            dec_conv0: conv3(c.dc1b, oc, device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            in_channels: ic,
        }
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Forward pass.
    ///
    /// Input  shape `[N, in_channels,  H, W]`,
    /// Output shape `[N, out_channels, H, W]`. Spatial dimensions must be a
    /// multiple of 16 (the network's alignment requirement, see
    /// `_ref/oidn/training/model.py:117`).
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        // Encoder
        let x = relu(self.enc_conv0.forward(input.clone()));
        let x = relu(self.enc_conv1.forward(x));
        let pool1 = self.pool.forward(x);

        let x = relu(self.enc_conv2.forward(pool1.clone()));
        let pool2 = self.pool.forward(x);

        let x = relu(self.enc_conv3.forward(pool2.clone()));
        let pool3 = self.pool.forward(x);

        let x = relu(self.enc_conv4.forward(pool3.clone()));
        let x = self.pool.forward(x);

        // Bottleneck
        let x = relu(self.enc_conv5a.forward(x));
        let x = relu(self.enc_conv5b.forward(x));

        // Decoder — each upsample is 2× nearest, concat is along channel dim (1)
        let x = upsample2x(x);
        let x = Tensor::cat(vec![x, pool3], 1);
        let x = relu(self.dec_conv4a.forward(x));
        let x = relu(self.dec_conv4b.forward(x));

        let x = upsample2x(x);
        let x = Tensor::cat(vec![x, pool2], 1);
        let x = relu(self.dec_conv3a.forward(x));
        let x = relu(self.dec_conv3b.forward(x));

        let x = upsample2x(x);
        let x = Tensor::cat(vec![x, pool1], 1);
        let x = relu(self.dec_conv2a.forward(x));
        let x = relu(self.dec_conv2b.forward(x));

        let x = upsample2x(x);
        let x = Tensor::cat(vec![x, input], 1);
        let x = relu(self.dec_conv1a.forward(x));
        let x = relu(self.dec_conv1b.forward(x));

        relu(self.dec_conv0.forward(x))
    }
}

/// 2× nearest-neighbor upsample, matching `F.interpolate(x, scale_factor=2, mode='nearest')`.
fn upsample2x(x: Tensor<4>) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}
