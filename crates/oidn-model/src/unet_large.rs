//! UNetLarge — direct port of `_ref/oidn/training/model.py:UNetLarge`.
//!
//! Differs from the base `UNet`:
//! - Each encoder/decoder stage has TWO convs (suffix `a`/`b`) instead of one
//!   between pools.
//! - 19 convs total (10 encoder + 8 decoder + 1 output).
//! - Output layer is `dec_conv1c` (not `dec_conv0`).
//! - Channel widths roughly double the base UNet.

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

fn conv3(in_ch: usize, out_ch: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([in_ch, out_ch], [3, 3])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(true)
        .init(device)
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelConfigLarge {
    pub ec1: usize,
    pub ec2: usize,
    pub ec3: usize,
    pub ec4: usize,
    pub ec5: usize,
    pub dc4: usize,
    pub dc3: usize,
    pub dc2: usize,
    pub dc1: usize,
}

impl ChannelConfigLarge {
    pub const BASE: Self = Self {
        ec1: 64,
        ec2: 96,
        ec3: 128,
        ec4: 192,
        ec5: 256,
        dc4: 192,
        dc3: 128,
        dc2: 96,
        dc1: 64,
    };

    pub const XL: Self = Self {
        ec1: 96,
        ec2: 128,
        ec3: 192,
        ec4: 256,
        ec5: 384,
        dc4: 256,
        dc3: 192,
        dc2: 128,
        dc1: 96,
    };
}

/// UNetLarge denoiser network.
#[derive(Module, Debug)]
pub struct UNetLarge {
    pub enc_conv1a: Conv2d,
    pub enc_conv1b: Conv2d,
    pub enc_conv2a: Conv2d,
    pub enc_conv2b: Conv2d,
    pub enc_conv3a: Conv2d,
    pub enc_conv3b: Conv2d,
    pub enc_conv4a: Conv2d,
    pub enc_conv4b: Conv2d,
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
    pub dec_conv1c: Conv2d,
    pub pool: MaxPool2d,
    pub in_channels: usize,
}

impl UNetLarge {
    pub fn new_with(
        in_channels: usize,
        out_channels: usize,
        c: ChannelConfigLarge,
        device: &Device,
    ) -> Self {
        let ic = in_channels;
        let oc = out_channels;
        Self {
            enc_conv1a: conv3(ic, c.ec1, device),
            enc_conv1b: conv3(c.ec1, c.ec1, device),
            enc_conv2a: conv3(c.ec1, c.ec2, device),
            enc_conv2b: conv3(c.ec2, c.ec2, device),
            enc_conv3a: conv3(c.ec2, c.ec3, device),
            enc_conv3b: conv3(c.ec3, c.ec3, device),
            enc_conv4a: conv3(c.ec3, c.ec4, device),
            enc_conv4b: conv3(c.ec4, c.ec4, device),
            enc_conv5a: conv3(c.ec4, c.ec5, device),
            enc_conv5b: conv3(c.ec5, c.ec5, device),
            dec_conv4a: conv3(c.ec5 + c.ec3, c.dc4, device),
            dec_conv4b: conv3(c.dc4, c.dc4, device),
            dec_conv3a: conv3(c.dc4 + c.ec2, c.dc3, device),
            dec_conv3b: conv3(c.dc3, c.dc3, device),
            dec_conv2a: conv3(c.dc3 + c.ec1, c.dc2, device),
            dec_conv2b: conv3(c.dc2, c.dc2, device),
            dec_conv1a: conv3(c.dc2 + ic, c.dc1, device),
            dec_conv1b: conv3(c.dc1, c.dc1, device),
            dec_conv1c: conv3(c.dc1, oc, device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            in_channels: ic,
        }
    }

    pub fn new(in_channels: usize, out_channels: usize, device: &Device) -> Self {
        Self::new_with(in_channels, out_channels, ChannelConfigLarge::BASE, device)
    }

    pub fn new_xl(in_channels: usize, out_channels: usize, device: &Device) -> Self {
        Self::new_with(in_channels, out_channels, ChannelConfigLarge::XL, device)
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Forward pass — mirrors `model.py:UNetLarge.forward` op order.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        // Encoder
        let x = relu(self.enc_conv1a.forward(input.clone()));
        let x = relu(self.enc_conv1b.forward(x));
        let pool1 = self.pool.forward(x);

        let x = relu(self.enc_conv2a.forward(pool1.clone()));
        let x = relu(self.enc_conv2b.forward(x));
        let pool2 = self.pool.forward(x);

        let x = relu(self.enc_conv3a.forward(pool2.clone()));
        let x = relu(self.enc_conv3b.forward(x));
        let pool3 = self.pool.forward(x);

        let x = relu(self.enc_conv4a.forward(pool3.clone()));
        let x = relu(self.enc_conv4b.forward(x));
        let x = self.pool.forward(x);

        // Bottleneck
        let x = relu(self.enc_conv5a.forward(x));
        let x = relu(self.enc_conv5b.forward(x));

        // Decoder
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
        relu(self.dec_conv1c.forward(x))
    }
}

fn upsample2x(x: Tensor<4>) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}
