// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving a model with the low-precision activation numerics it was
// trained with for its clients. If your team needs expertise in NVFP4 and MXFP8 microscaled inference then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The declared activation quantization of a product, as an opt-in numerics tier.
//!
//! A release may declare that its runtime quantizes the activation operand of some products
//! (`hf_quant_config.json`'s `required_activation_quantization`, parsed by
//! [`checkpoint::packed::ActivationQuant::scheme`]): a W4A4 NVFP4 or W8A8 MXFP8 product, the numerics the model was
//! trained with. brain's reference numerics keep activations in f32 (the published reference does), so this tier is
//! taken only when asked for (`--linear-activations declared`, [`crate::policy::LinearActivations::Declared`]) and is
//! measured as KL and top-1 against the reference.
//!
//! The tier is applied as a round trip of the activation rows through the declared format ([`qdq_step`]) feeding the
//! product the weights already have: the decoded activation values are exactly what a quantized product multiplies,
//! so the result is that product up to the f32 accumulation order. The tradeoff is speed: the round trip writes the
//! rows back as f32 and the product reads f32, where a native W4A4 / W8A8 product would read the packed codes. It is
//! the faithful, simplest form; a packed-operand product is a later optimisation with this as its parity oracle.
//!
//! The rounding is piecewise constant, so its gradient is the straight-through identity (dx = dy): there is no backward
//! kernel.

use checkpoint::packed::ActivationScheme;
use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

/// The pipeline [`qdq_step`] dispatches, for a model's kernel list.
pub const KERNELS: [(&str, &str); 1] = [("act_qdq_nvfp4_mxfp8", kernels::ACT_QDQ_NVFP4_MXFP8)];

/// `out = dequant(quant(x))` of `rows` activation rows of width `k` in `scheme` (`act_qdq_nvfp4_mxfp8.wgsl`; host
/// reference [`crate::hostmath::nvfp4_four_over_six_qdq`] / [`crate::hostmath::mxfp8_qdq`], bit for bit). `k` must be a
/// multiple of the scheme's block; the kernel must be registered ([`KERNELS`]).
pub fn qdq_step(g: &Gpu, scheme: ActivationScheme, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32, k: u32) -> Result<Step, String> {
    if !k.is_multiple_of(scheme.block()) {
        return Err(format!("{scheme:?} activations need rows of a multiple of {} values, these have {k}", scheme.block()));
    }
    let idx = g.kernel_index(KERNELS[0].0).ok_or("kernel 'act_qdq_nvfp4_mxfp8' is not registered on this Gpu: add model::act_quant::KERNELS to the kernel list")?;
    let code = match scheme {
        ActivationScheme::Nvfp4FourOverSixMae => 0,
        ActivationScheme::Mxfp8E4m3 => 1,
    };
    Ok(g.dispatch(idx, &[x, out], &[rows, k, code], Dispatch::Workgroups(rows)))
}
