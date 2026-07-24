//! Load TZA tensor data into a `UNet` Burn `Module`.
//!
//! Tensor names in TZA shipped by Intel match PyTorch state_dict names exactly
//! (`enc_conv0.weight`, `dec_conv4a.bias`, …). Layout `oihw` is the same order
//! Burn `Conv2d` weights expect, so the only conversion we do is `f16 → f32`.

use burn::{
    module::{Param, ParamId},
    nn::conv::Conv2d,
    tensor::{Device, Tensor, TensorData},
};
use oidn_tza::{DType, Layout, Tensor as TzaTensor, TensorMap};
use thiserror::Error;

use crate::unet::UNet;
use crate::unet_large::UNetLarge;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("missing tensor in archive: {0:?}")]
    MissingTensor(String),

    #[error("tensor {name:?} expected layout {expected:?}, got {got:?}")]
    BadLayout {
        name: String,
        expected: &'static str,
        got: Layout,
    },

    #[error("tensor {name:?} expected shape {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<u32>,
    },
}

fn fetch<'a>(map: &'a TensorMap, name: &str) -> Result<&'a TzaTensor, LoadError> {
    map.get(name)
        .ok_or_else(|| LoadError::MissingTensor(name.to_owned()))
}

/// Build a 4-D Burn `Param` from an `oihw` TZA tensor.
fn into_param4(
    name: &str,
    src: &TzaTensor,
    expected_shape: [usize; 4],
    device: &Device,
) -> Result<Param<Tensor<4>>, LoadError> {
    if src.desc.layout != Layout::Oihw {
        return Err(LoadError::BadLayout {
            name: name.to_owned(),
            expected: "oihw",
            got: src.desc.layout,
        });
    }
    if src.desc.dims.len() != 4 || (0..4).any(|i| src.desc.dims[i] as usize != expected_shape[i]) {
        return Err(LoadError::ShapeMismatch {
            name: name.to_owned(),
            expected: expected_shape.to_vec(),
            got: src.desc.dims.clone(),
        });
    }
    let data: Vec<f32> = match src.desc.dtype {
        DType::Float32 => src.as_f32().unwrap().to_vec(),
        DType::Float16 => src.as_f16().unwrap().iter().map(|h| h.to_f32()).collect(),
    };
    let tensor = Tensor::<4>::from_data(TensorData::new(data, expected_shape), device);
    Ok(Param::initialized(ParamId::new(), tensor))
}

/// Build a 1-D Burn `Param` (bias) from an `x` TZA tensor.
fn into_param1(
    name: &str,
    src: &TzaTensor,
    expected_len: usize,
    device: &Device,
) -> Result<Param<Tensor<1>>, LoadError> {
    if src.desc.layout != Layout::X {
        return Err(LoadError::BadLayout {
            name: name.to_owned(),
            expected: "x",
            got: src.desc.layout,
        });
    }
    if src.desc.dims.len() != 1 || src.desc.dims[0] as usize != expected_len {
        return Err(LoadError::ShapeMismatch {
            name: name.to_owned(),
            expected: vec![expected_len],
            got: src.desc.dims.clone(),
        });
    }
    let data: Vec<f32> = match src.desc.dtype {
        DType::Float32 => src.as_f32().unwrap().to_vec(),
        DType::Float16 => src.as_f16().unwrap().iter().map(|h| h.to_f32()).collect(),
    };
    let tensor = Tensor::<1>::from_data(TensorData::new(data, [expected_len]), device);
    Ok(Param::initialized(ParamId::new(), tensor))
}

/// Overwrite `conv`'s weight and bias from `<layer>.weight` / `<layer>.bias`.
fn load_conv(
    mut conv: Conv2d,
    layer: &str,
    map: &TensorMap,
    device: &Device,
) -> Result<Conv2d, LoadError> {
    // The Conv2d's `weight` shape is [out, in, kH, kW] — same order as TZA `oihw`.
    let w_shape: [usize; 4] = conv.weight.shape().dims();
    let b_len = conv
        .bias
        .as_ref()
        .map(|b| b.shape().dims::<1>()[0])
        .unwrap_or(w_shape[0]);

    let w_name = format!("{layer}.weight");
    let b_name = format!("{layer}.bias");

    let w_src = fetch(map, &w_name)?;
    let b_src = fetch(map, &b_name)?;

    conv.weight = into_param4(&w_name, w_src, w_shape, device)?;
    conv.bias = Some(into_param1(&b_name, b_src, b_len, device)?);
    Ok(conv)
}

/// Load all weights from a TZA `TensorMap` into the given U-Net.
pub fn load_tza(
    unet: UNet,
    map: &TensorMap,
    device: &Device,
) -> Result<UNet, LoadError> {
    Ok(UNet {
        enc_conv0: load_conv(unet.enc_conv0, "enc_conv0", map, device)?,
        enc_conv1: load_conv(unet.enc_conv1, "enc_conv1", map, device)?,
        enc_conv2: load_conv(unet.enc_conv2, "enc_conv2", map, device)?,
        enc_conv3: load_conv(unet.enc_conv3, "enc_conv3", map, device)?,
        enc_conv4: load_conv(unet.enc_conv4, "enc_conv4", map, device)?,
        enc_conv5a: load_conv(unet.enc_conv5a, "enc_conv5a", map, device)?,
        enc_conv5b: load_conv(unet.enc_conv5b, "enc_conv5b", map, device)?,
        dec_conv4a: load_conv(unet.dec_conv4a, "dec_conv4a", map, device)?,
        dec_conv4b: load_conv(unet.dec_conv4b, "dec_conv4b", map, device)?,
        dec_conv3a: load_conv(unet.dec_conv3a, "dec_conv3a", map, device)?,
        dec_conv3b: load_conv(unet.dec_conv3b, "dec_conv3b", map, device)?,
        dec_conv2a: load_conv(unet.dec_conv2a, "dec_conv2a", map, device)?,
        dec_conv2b: load_conv(unet.dec_conv2b, "dec_conv2b", map, device)?,
        dec_conv1a: load_conv(unet.dec_conv1a, "dec_conv1a", map, device)?,
        dec_conv1b: load_conv(unet.dec_conv1b, "dec_conv1b", map, device)?,
        dec_conv0: load_conv(unet.dec_conv0, "dec_conv0", map, device)?,
        pool: unet.pool,
        in_channels: unet.in_channels,
    })
}

/// Same as `load_tza`, but for the `UNetLarge` topology.
///
/// Layer naming matches `_ref/oidn/training/model.py:UNetLarge` —
/// `enc_conv{1..5}{a,b}`, `dec_conv{4..2}{a,b}`, `dec_conv1{a,b,c}`.
pub fn load_tza_large(
    unet: UNetLarge,
    map: &TensorMap,
    device: &Device,
) -> Result<UNetLarge, LoadError> {
    Ok(UNetLarge {
        enc_conv1a: load_conv(unet.enc_conv1a, "enc_conv1a", map, device)?,
        enc_conv1b: load_conv(unet.enc_conv1b, "enc_conv1b", map, device)?,
        enc_conv2a: load_conv(unet.enc_conv2a, "enc_conv2a", map, device)?,
        enc_conv2b: load_conv(unet.enc_conv2b, "enc_conv2b", map, device)?,
        enc_conv3a: load_conv(unet.enc_conv3a, "enc_conv3a", map, device)?,
        enc_conv3b: load_conv(unet.enc_conv3b, "enc_conv3b", map, device)?,
        enc_conv4a: load_conv(unet.enc_conv4a, "enc_conv4a", map, device)?,
        enc_conv4b: load_conv(unet.enc_conv4b, "enc_conv4b", map, device)?,
        enc_conv5a: load_conv(unet.enc_conv5a, "enc_conv5a", map, device)?,
        enc_conv5b: load_conv(unet.enc_conv5b, "enc_conv5b", map, device)?,
        dec_conv4a: load_conv(unet.dec_conv4a, "dec_conv4a", map, device)?,
        dec_conv4b: load_conv(unet.dec_conv4b, "dec_conv4b", map, device)?,
        dec_conv3a: load_conv(unet.dec_conv3a, "dec_conv3a", map, device)?,
        dec_conv3b: load_conv(unet.dec_conv3b, "dec_conv3b", map, device)?,
        dec_conv2a: load_conv(unet.dec_conv2a, "dec_conv2a", map, device)?,
        dec_conv2b: load_conv(unet.dec_conv2b, "dec_conv2b", map, device)?,
        dec_conv1a: load_conv(unet.dec_conv1a, "dec_conv1a", map, device)?,
        dec_conv1b: load_conv(unet.dec_conv1b, "dec_conv1b", map, device)?,
        dec_conv1c: load_conv(unet.dec_conv1c, "dec_conv1c", map, device)?,
        pool: unet.pool,
        in_channels: unet.in_channels,
    })
}
