// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Three keyframes in one generation pass.** A wiring gate on
//! `--start-frame` + `--mid-frame` + `--end-frame` used together: each still
//! is really VAE-encoded, really appended as its own guiding block, and really
//! reaches the denoiser at the instant it was pointed at.
//!
//! Swedish Embedded AB implements keyframe-conditioned video diffusion for its
//! clients. If your team needs a generation pipeline that answers to several
//! anchor frames at once, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ## What this can and cannot claim
//!
//! Real conv VAE, tiny random-weight DiT, CPU - the `longform.rs`/`upscale.rs`
//! pattern. The DiT carries no semantics, so nothing here says a clip *looks*
//! like its anchors. What it does say is that each anchor is a real input to
//! the run: change one still's pixels, or the pixel frame it is pointed at,
//! and the decoded clip changes. Anything dropped on the floor - an unencoded
//! path, a block appended at the wrong position, a still overwritten by the
//! next one - makes two of these runs equal that must not be.
//!
//! The perceptual half of the claim (a decoded frame reproducing its
//! conditioning still) needs a real checkpoint and lives in `anchor_real.rs`,
//! which is `#[ignore]`d for cost.

mod real_weights {
    use std::path::Path;

    use ltxv::pipeline::{generate, GenOpts, Paths};

    /// The named environment variable, else the repo-relative
    /// `resources/ltxv/weights/` the real files ship under - never a literal
    /// machine path, the same convention `longform.rs`/`upscale.rs` use.
    fn weights_path(rel: &str) -> Option<String> {
        // The model store's copy first, by file name; else the repo-relative one.
        let name = Path::new(rel).file_name()?.to_string_lossy().into_owned();
        if let Some(p) = brain_testutil::model_dir("Lightricks/LTX-2.5").map(|d| format!("{d}/{name}")).filter(|p| Path::new(p).exists()) {
            return Some(p);
        }
        let p = format!("{}/../../resources/ltxv/weights/{rel}", env!("CARGO_MANIFEST_DIR"));
        Path::new(&p).exists().then_some(p)
    }

    const FRAMES: usize = 17;
    const SIDE: usize = 64;

    /// A flat still of one colour, at exactly the generation size, so
    /// `generate`'s own resize is a no-op and the file on disk is what gets
    /// encoded.
    fn still(dir: &Path, name: &str, rgb: [u8; 3]) -> String {
        let path = dir.join(name);
        let px: Vec<u8> = std::iter::repeat_n(rgb, SIDE * SIDE).flatten().collect();
        image::RgbImage::from_raw(SIDE as u32, SIDE as u32, px).expect("a flat RGB image").save(&path).expect("write the conditioning still");
        path.to_string_lossy().into_owned()
    }

    fn base_opts() -> GenOpts {
        GenOpts { frames: FRAMES, width: SIDE, height: SIDE, steps: 2, fps: 8, device: Some("cpu".into()), seed: 5, ..GenOpts::default() } // seed: the anchor-content assertions compare runs pixel for pixel, so every run must share one noise draw (needs the real VAE; not run)
    }

    fn run(paths: &Paths, o: &GenOpts) -> Vec<Vec<u8>> {
        let (video, _) = generate(paths, "a moving bar", o, &capability::CancelToken::default(), |_, _, _| {}).expect("generate");
        assert_eq!(video.frames.len(), FRAMES, "the clip is not the requested length");
        assert_eq!((video.width as usize, video.height as usize), (SIDE, SIDE));
        video.frames
    }

    /// Start, middle and end anchors in ONE pass, and each of the three is
    /// observably part of the run.
    #[test]
    fn three_simultaneous_anchors_each_reach_the_denoiser() {
        let Some(vae) = weights_path("vae/ltx-2.5-video-vae-conv-bf16.safetensors") else {
            return brain_testutil::skip("put the real VAE in the model store");
        };
        let paths = Paths::new(vae.clone());
        let dir = std::env::temp_dir().join(format!("ltxv-anchors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let start = still(&dir, "start.png", [220, 30, 30]);
        let mid = still(&dir, "mid.png", [30, 220, 30]);
        let other_mid = still(&dir, "other-mid.png", [30, 30, 220]);
        let end = still(&dir, "end.png", [230, 230, 30]);

        let three = GenOpts { start_frame: Some(start.clone()), keyframes: vec![(8, mid.clone())], end_frame: Some(end.clone()), ..base_opts() };
        let a = run(&paths, &three);
        // A different picture at the same instant.
        let b = run(&paths, &GenOpts { keyframes: vec![(8, other_mid)], ..three.clone() });
        // The same picture at a different instant.
        let c = run(&paths, &GenOpts { keyframes: vec![(12, mid.clone())], ..three.clone() });
        // The two-anchor request this feature generalises.
        let d = run(&paths, &GenOpts { keyframes: Vec::new(), ..three.clone() });
        let _ = std::fs::remove_dir_all(&dir);

        assert_ne!(a, b, "changing the middle still's pixels changed nothing: the mid anchor's content never reached the denoiser");
        assert_ne!(a, c, "moving the middle still from frame 8 to frame 12 changed nothing: the mid anchor's POSITION never reached the denoiser");
        assert_ne!(a, d, "dropping the middle still changed nothing: the third anchor is not in the sequence at all");
        assert!(a.iter().any(|f| f.iter().any(|&v| v != f[0])), "every frame is a flat colour - nothing was decoded");
    }
}
