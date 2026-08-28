// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a VAE graph actually costs on the device, and whether the number a
//! placement decision is made from tells the truth about it.
//!
//! Swedish Embedded AB implements memory-bounded inference for teams shipping
//! onto hardware with a hard ceiling. If your team needs expertise in sizing a
//! model to the card it has to run on, you can procure our services by sending
//! an email to info@swedishembedded.com.
//!
//! An estimate that is too SMALL is the bug: the plan says the card has room,
//! every stage before the last one completes, and then the driver reports out
//! of memory. Too large is merely conservative. So the never-under-report
//! direction is asserted at every size, and tightness only at the sizes real
//! images are actually generated at - a graph with a size-independent scratch
//! floor cannot be bracketed tightly at thumbnail sizes, and over-reserving a
//! few hundred MiB there costs nothing.
//!
//! **These need a real GPU.** The CPU JIT keeps the reference kernels rather
//! than the GEMM/im2col lowerings (`blocks::Builder::coop`), so its graph is a
//! different and smaller one; calibrating against it would under-report the
//! path that actually runs, which is precisely the bug.

mod zeros;

use vae::{VaeConfig, VaeDecoder, VaeEncoder};

const MIB: f64 = (1u64 << 20) as f64;

fn mib(b: u64) -> f64 {
    b as f64 / MIB
}

/// A card with room, or `None`. Reads free VRAM directly rather than taking
/// the ambient default: this file builds multi-GiB graphs, and the ambient
/// default in a bare test binary is card 0 whether or not anything else is
/// already living there.
fn card_with_room(want_mib: u64) -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut it = l.split(',').map(str::trim);
            Some((it.next()?.parse::<u32>().ok()?, it.next()?.parse::<u64>().ok()?))
        })
        .max_by_key(|&(_, free)| free)
        .filter(|&(_, free)| free > want_mib)
        .map(|(i, _)| format!("gpu{i}"))
}

/// The decode estimate must never promise a decode is smaller than it is, and
/// must be close at the sizes placement decisions are actually made at.
#[test]
fn the_decode_estimate_brackets_what_the_decode_really_allocates() {
    let Some(dev) = card_with_room(6144) else {
        eprintln!("skip: no GPU with room");
        return;
    };
    let c = VaeConfig::flux2();
    let ts = zeros::decoder(&c);
    let mut widest = None;
    for (lh, lw) in [(16u32, 16u32), (32, 24), (64, 48)] {
        let real = VaeDecoder::from_diffusers(c.clone(), &ts, lh, lw, Some(&dev)).device_bytes();
        let pred = vae::decoder_device_bytes(&c, lh, lw);
        assert!(
            pred >= real,
            "estimate must never under-report: latent {lh}x{lw} predicted {:.1} MiB < real {:.1} MiB",
            mib(pred),
            mib(real)
        );
        widest = Some((lh, lw, real, pred));
    }
    // 512x384 out - a real frame, where being 4x short is the field failure.
    let (lh, lw, real, pred) = widest.expect("at least one size");
    assert!(
        pred <= real * 5 / 4,
        "at latent {lh}x{lw} the estimate must be close, not merely safe: predicted {:.1} MiB vs real {:.1} MiB",
        mib(pred),
        mib(real)
    );
}

/// The same contract for the encoder, which is what a reference image goes
/// through - once per reference.
#[test]
fn the_encode_estimate_brackets_what_the_encode_really_allocates() {
    let Some(dev) = card_with_room(4096) else {
        eprintln!("skip: no GPU with room");
        return;
    };
    let c = VaeConfig::flux2();
    let ts = zeros::encoder(&c);
    let mut widest = None;
    for (h, w) in [(128u32, 128u32), (256, 192), (512, 512)] {
        let real = VaeEncoder::from_diffusers(c.clone(), &ts, h, w, Some(&dev)).device_bytes();
        let pred = vae::encoder_device_bytes(&c, h, w);
        assert!(pred >= real, "under-report at {h}x{w}: {:.1} < {:.1} MiB", mib(pred), mib(real));
        widest = Some((h, w, real, pred));
    }
    let (h, w, real, pred) = widest.expect("at least one size");
    assert!(pred <= real * 3 / 2, "at {h}x{w}: {:.1} MiB vs real {:.1} MiB", mib(pred), mib(real));
}

/// The estimate has to SCALE with the image, not merely be right once. A flat
/// constant passes any single-point bracket, and a flat constant is exactly
/// what reserved 2 GiB for a decode that needed four times that.
#[test]
fn the_decode_estimate_grows_with_the_image_it_decodes() {
    let c = VaeConfig::flux2();
    let fixed = vae::decoder_weight_bytes(&c);
    // Compare the SIZE-DEPENDENT part: the weights are a constant that would
    // otherwise mask a predictor that ignores its input entirely.
    let small = vae::decoder_device_bytes(&c, 32, 24) - fixed;
    let big = vae::decoder_device_bytes(&c, 64, 48) - fixed;
    assert!(
        big > small,
        // perf-number: pixel-count ratio in the assertion message (64x48 vs 32x24), not a measured runtime speedup
        "4x the pixels must cost more ({:.1} -> {:.1} MiB above weights)",
        mib(small),
        mib(big)
    );
    let quadruple = vae::decoder_device_bytes(&c, 128, 96) - vae::decoder_device_bytes(&c, 64, 48);
    let single = vae::decoder_device_bytes(&c, 64, 48) - vae::decoder_device_bytes(&c, 32, 24);
    assert!(
        quadruple >= single * 7 / 2,
        "the per-pixel term must be linear in pixels: {:.1} vs {:.1} MiB",
        mib(quadruple),
        mib(single)
    );
}

/// At a real output size the decode is dominated by activations, not weights.
/// This is the shape of the field failure in one assertion: an estimate built
/// from the checkpoint alone is wrong by multiples exactly when it matters.
#[test]
fn activations_dominate_weights_at_a_real_output_size() {
    let c = VaeConfig::flux2();
    // latent 128x96 -> 1024x768, the size the reported failure decoded at.
    let full = vae::decoder_device_bytes(&c, 128, 96);
    let weights = vae::decoder_weight_bytes(&c);
    assert!(
        full > weights * 4,
        "at 1024x768 the decode must be dominated by activations, not the {:.1} MiB of weights (total {:.1} MiB)",
        mib(weights),
        mib(full)
    );
    // ...and specifically by the SIZE-DEPENDENT activations, not by whatever
    // fixed scratch the estimate also carries. A one-pixel latent is the
    // estimate with its per-pixel term switched off; a full frame must dwarf
    // it, or the estimate is a constant wearing a formula.
    let floor = vae::decoder_device_bytes(&c, 1, 1);
    assert!(
        full > floor * 4,
        "the per-pixel term must dominate at a real size: {:.1} MiB at 1024x768 vs {:.1} MiB floor",
        mib(full),
        mib(floor)
    );
}

/// The weight figures are summed from the same schedule the builder uploads,
/// so they must agree with what a built graph reports for its weights - the
/// whole-graph total minus what the activations of a deliberately tiny graph
/// can account for. Cheap, and it catches a schedule that has moved.
#[test]
fn the_weight_sum_matches_a_built_graph() {
    let Some(dev) = card_with_room(2048) else {
        eprintln!("skip: no GPU with room");
        return;
    };
    let c = VaeConfig::flux2();
    for (name, real, weights) in [
        (
            "decoder",
            VaeDecoder::from_diffusers(c.clone(), &zeros::decoder(&c), 4, 4, Some(&dev)).device_bytes(),
            vae::decoder_weight_bytes(&c),
        ),
        (
            "encoder",
            VaeEncoder::from_diffusers(c.clone(), &zeros::encoder(&c), 32, 32, Some(&dev)).device_bytes(),
            vae::encoder_weight_bytes(&c),
        ),
    ] {
        assert!(weights <= real, "{name}: weight sum {:.1} MiB exceeds the whole graph {:.1} MiB", mib(weights), mib(real));
        // At this size activations are a rounding error, so the weights are
        // essentially the whole graph. A schedule that gained or lost a block
        // breaks this by megabytes.
        assert!(
            real - weights < real / 4,
            "{name}: weight sum {:.1} MiB does not account for a tiny graph's {:.1} MiB",
            mib(weights),
            mib(real)
        );
    }
}
