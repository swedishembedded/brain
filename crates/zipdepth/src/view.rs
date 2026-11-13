// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a depth map is shown as: the colorized map, the photo beside it, a
//! depth effect on the photo (fog, depth of field), or a stereogram.
//!
//! One renderer for every surface that shows depth: the served `infer`
//! action's `view` output and a live window draw the same pixels from the
//! same map, so a frame saved from one matches the other.
//!
//! Swedish Embedded AB implements depth sensing and its visualization on
//! embedded cameras for its clients. If your team needs expertise in
//! monocular depth on its own hardware then you can procure our services by
//! sending an email to info@swedishembedded.com.

use imaging::viz::{colorize, composite_side_by_side, Bounds, Colormap};

use crate::StereoOpts;

/// A way of showing one frame's depth. Each renders at its natural size
/// ([`View::canvas`]): the side-by-side views are twice the frame's width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    /// The photo | colorized depth, side by side.
    Side,
    /// Colorized depth alone.
    Depth,
    /// A random-dot autostereogram of the depth.
    Stereo,
    /// An autostereogram patterned with the photo itself, so its textures
    /// appear in depth.
    StereoImage,
    /// A cross-eye stereo pair of the photo (left | right), offset by depth.
    StereoDual,
    /// The photo with depth fog: far things fade into haze.
    Fog,
    /// The photo with depth-of-field blur: far things are softer.
    Blur,
}

impl View {
    /// Every view, in the order a viewer cycles them: the photo effects,
    /// then the map, then the stereograms.
    pub const ALL: [View; 7] = [View::Side, View::Fog, View::Blur, View::Depth, View::Stereo, View::StereoImage, View::StereoDual];

    /// The name a parameter or flag spells this view with.
    pub fn name(self) -> &'static str {
        match self {
            View::Side => "side",
            View::Depth => "depth",
            View::Stereo => "stereo",
            View::StereoImage => "stereo-image",
            View::StereoDual => "stereo-dual",
            View::Fog => "fog",
            View::Blur => "blur",
        }
    }

    pub fn parse(name: &str) -> Result<View, String> {
        View::ALL.into_iter().find(|v| v.name() == name).ok_or_else(|| {
            let names: Vec<&str> = View::ALL.iter().map(|v| v.name()).collect();
            format!("unknown view '{name}' (expected one of {})", names.join(", "))
        })
    }

    /// The next view in [`View::ALL`], wrapping.
    pub fn cycle(self) -> View {
        let i = View::ALL.iter().position(|&v| v == self).unwrap_or(0);
        View::ALL[(i + 1) % View::ALL.len()]
    }

    /// The canvas a `w x h` frame renders to in this view.
    pub fn canvas(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            View::Side | View::StereoDual => (2 * w, h),
            _ => (w, h),
        }
    }
}

/// How a frame's depth is drawn: the window it is normalized against, the
/// colormap, and the stereogram geometry.
pub struct Style {
    pub bounds: Bounds,
    pub colormap: Colormap,
    pub stereo: StereoOpts,
}

/// `depth` (one value per pixel of the `w x h` frame `rgb8`) shown as `view`:
/// RGB8 at [`View::canvas`]'s size.
pub fn render(rgb8: &[u8], depth: &[f32], w: u32, h: u32, view: View, style: &Style) -> Vec<u8> {
    let (bounds, near_is_high) = (style.bounds, style.stereo.near_is_high);
    match view {
        View::Side => {
            let colored = colorize(depth, bounds, style.colormap);
            composite_side_by_side(rgb8, w, h, &colored, w, h).0
        }
        View::Depth => colorize(depth, bounds, style.colormap),
        View::Stereo => crate::autostereogram(depth, w, h, bounds, &style.stereo),
        View::StereoImage => crate::autostereogram_textured(depth, w, h, bounds, &style.stereo, rgb8),
        View::StereoDual => {
            // Peak disparity about 1/25 of the width: enough relief to fuse,
            // small enough that disocclusion holes stay tiny.
            let max_disparity = (w / 25).clamp(8, 40);
            crate::stereo_pair(rgb8, depth, w, h, bounds, max_disparity, near_is_high)
        }
        View::Fog => crate::fog(rgb8, depth, w, h, bounds, [210, 216, 226], 3.5, near_is_high),
        View::Blur => {
            let max_radius = (w / 40).clamp(3, 20);
            crate::depth_blur(rgb8, depth, w, h, bounds, max_radius, near_is_high)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_view_round_trips_its_name_and_cycles_through_all_of_them() {
        let mut seen = vec![View::Side];
        for v in View::ALL {
            assert_eq!(View::parse(v.name()), Ok(v));
        }
        let mut v = View::Side.cycle();
        while v != View::Side {
            seen.push(v);
            v = v.cycle();
        }
        assert_eq!(seen, View::ALL.to_vec());
        assert!(View::parse("nope").unwrap_err().contains("stereo-dual"));
    }

    #[test]
    fn each_view_fills_its_canvas() {
        let (w, h) = (64u32, 32u32);
        let rgb8 = vec![128u8; (w * h * 3) as usize];
        let depth: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let style = Style { bounds: Bounds::from_percentiles(&depth, 0.02, 0.98), colormap: Colormap::Turbo, stereo: StereoOpts::with_stripes(w, 5) };
        for v in View::ALL {
            let (cw, ch) = v.canvas(w, h);
            assert_eq!(render(&rgb8, &depth, w, h, v, &style).len(), (cw * ch * 3) as usize, "{}", v.name());
        }
    }
}
