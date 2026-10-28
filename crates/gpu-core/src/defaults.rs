// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for getting the most out of the
// hardware a team already owns for its clients. If your team needs expertise
// in inference serving that configures itself for the machine it lands on,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Defaults chosen from the hardware.
//!
//! A setting nobody chose on the command line - the backend, a model's KV-cache
//! tier, whether speculation or a prompt cache gets memory - is decided here
//! from what the machine has, by one mechanism:
//!
//! * [`Hardware`] is what was detected: the cards, their memory and compute
//!   capability, and whether the CUDA backend can run.
//! * A setting's defaults are a list of [`Rule`]s. The FIRST rule whose `when`
//!   holds on this hardware supplies the candidates, ranked by expected
//!   throughput, best first. The last rule of a list holds everywhere, so every
//!   machine gets an answer.
//! * [`choose`] walks those candidates and keeps the first one the caller's
//!   `fits` accepts - typically "the memory plan still fits" - so a candidate
//!   the hardware could run but whose bytes do not fit falls through to the
//!   next instead of failing. Two 8 GB cards and two 24 GB cards are not two
//!   tables: they are the same list, with the planner deciding where each stops.
//! * A ranked list ends in its leanest candidate, which [`choose_or_leanest`]
//!   settles on when none fits, so a default never refuses a machine on its own.
//!
//! Adding a default is adding a rule (or a candidate) where its throughput
//! ranks it. What was chosen and why is returned, so a server reports it.

use crate::devices::Backend;
use std::fmt::Display;

/// One card, as far as a default may depend on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Card {
    /// Canonical index (`gpu<i>`).
    pub index: u32,
    pub name: String,
    /// PCI vendor id: `0x10de` NVIDIA, `0x1002` AMD, `0x8086` Intel.
    pub vendor_id: u32,
    /// Device memory, bytes.
    pub vram_bytes: u64,
    /// CUDA compute capability, when the CUDA backend can run on the card.
    pub compute_capability: Option<(u32, u32)>,
}

/// What the machine offers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hardware {
    pub cards: Vec<Card>,
    /// Why the CUDA backend cannot run here; `None` when it can.
    pub cuda_unusable: Option<String>,
}

/// PCI vendor id of NVIDIA.
pub const NVIDIA: u32 = 0x10de;

impl Hardware {
    /// The machine this process runs on, probed once.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn detected() -> &'static Hardware {
        static HW: std::sync::OnceLock<Hardware> = std::sync::OnceLock::new();
        HW.get_or_init(|| {
            let cuda = backend_cuda::usable_gpus();
            // A card is matched by its UUID, the identity both enumerations share.
            let cc_of = |uuid: Option<[u8; 16]>| {
                let uuid = uuid?;
                cuda.as_ref().ok()?.iter().find(|g| g.identity.uuid == Some(uuid)).map(|g| g.compute_capability)
            };
            let cards = crate::devices::gpus()
                .iter()
                .map(|d| Card {
                    index: d.index,
                    name: d.identity.name.clone(),
                    vendor_id: d.identity.vendor_id,
                    vram_bytes: d.identity.vram_bytes,
                    compute_capability: cc_of(d.identity.uuid),
                })
                .collect();
            Hardware { cards, cuda_unusable: cuda.err() }
        })
    }

    /// The warning for a machine that has an NVIDIA card and is driving it through
    /// `backend` only because CUDA cannot load: `None` when CUDA drives it, when no
    /// NVIDIA card is present, or when CUDA works and the portable backend was asked
    /// for. The reason is the loader's own, so it names the missing library.
    pub fn cuda_fallback_notice(&self, backend: Backend) -> Option<String> {
        if !matches!(backend, Backend::Wgpu | Backend::Vulkan) {
            return None;
        }
        let why = self.cuda_unusable.as_ref()?;
        let card = self.cards.iter().find(|c| c.vendor_id == NVIDIA)?;
        Some(format!(
            "NVIDIA {} is present but the CUDA backend cannot load ({why}); falling back to the {backend} backend, which runs slower and lacks the CUDA-only kernels (speculation among them). Make the CUDA driver library (libcuda.so.1) and NVRTC (libnvrtc.so) findable, for example on LD_LIBRARY_PATH, to run on CUDA",
            card.name
        ))
    }

    /// Every card is NVIDIA's and the CUDA backend can run on each.
    pub fn cuda_everywhere(&self) -> bool {
        self.cuda_unusable.is_none() && !self.cards.is_empty() && self.cards.iter().all(|c| c.vendor_id == NVIDIA && c.compute_capability.is_some())
    }

    /// Whether the native fused kernel `which` will run on every one of the
    /// cards `indices` under the backend this process serves with - what a
    /// memory plan asks before any device is opened, so it charges only what
    /// loading will allocate.
    pub fn offers_fused(&self, which: crate::Fused, indices: &[u32]) -> bool {
        crate::backend_name() == "cuda"
            && !indices.is_empty()
            && indices.iter().all(|i| self.cards.iter().find(|c| c.index == *i).and_then(|c| c.compute_capability).is_some_and(|cc| which.offered_at(cc)))
    }

    /// The lowest compute capability among the cards, when every card has one.
    pub fn min_compute_capability(&self) -> Option<(u32, u32)> {
        self.cards.iter().map(|c| c.compute_capability).collect::<Option<Vec<_>>>()?.into_iter().min()
    }
}

/// One rule of a setting's defaults: on hardware where `when` holds, try
/// `ranked` in order (best expected throughput first). `why` says what makes
/// the first candidate right for that hardware.
pub struct Rule<T: 'static> {
    pub when: fn(&Hardware) -> bool,
    pub ranked: &'static [T],
    pub why: &'static str,
}

/// What [`choose`] settled on.
#[derive(Clone, Debug, PartialEq)]
pub struct Chosen<T> {
    pub value: T,
    /// The rule's reason, and why each better-ranked candidate was passed over.
    pub why: String,
}

/// The candidates of the first rule that holds on `hw`, with its reason.
///
/// # Panics
/// If no rule holds: a list must end in a rule that holds everywhere.
pub fn candidates<T>(rules: &[Rule<T>], hw: &Hardware) -> (&'static [T], &'static str) {
    let rule = rules.iter().find(|r| (r.when)(hw)).expect("a defaults list ends in a rule that holds on every machine");
    (rule.ranked, rule.why)
}

/// The best-ranked candidate of the rule that holds on `hw` that `fits`
/// accepts; `Err` lists why each was refused.
pub fn choose<T: Copy + Display>(rules: &[Rule<T>], hw: &Hardware, mut fits: impl FnMut(T) -> Result<(), String>) -> Result<Chosen<T>, String> {
    let (ranked, why) = candidates(rules, hw);
    let mut passed = Vec::new();
    for &value in ranked {
        match fits(value) {
            Ok(()) if passed.is_empty() => return Ok(Chosen { value, why: why.to_string() }),
            Ok(()) => return Ok(Chosen { value, why: format!("{why}; {}", passed.join("; ")) }),
            Err(e) => passed.push(format!("{value} passed over: {e}")),
        }
    }
    Err(passed.join("; "))
}

/// [`choose`], settling on the rule's last (leanest) candidate when none fits,
/// with why each was refused - the caller's own checks then say what to change.
pub fn choose_or_leanest<T: Copy + Display>(rules: &[Rule<T>], hw: &Hardware, fits: impl FnMut(T) -> Result<(), String>) -> Chosen<T> {
    choose(rules, hw, fits).unwrap_or_else(|refused| {
        let (ranked, why) = candidates(rules, hw);
        let value = *ranked.last().expect("a rule ranks at least one candidate");
        Chosen { value, why: format!("{why}; none fits, so the leanest: {refused}") }
    })
}

/// The backend that drives the GPUs when none was asked for: the native
/// kernels and the urgent lane are CUDA's, so CUDA wherever every card can run
/// it; the portable backend on any other GPU; the CPU without one.
pub const BACKEND: &[Rule<Backend>] = &[
    Rule { when: Hardware::cuda_everywhere, ranked: &[Backend::Cuda], why: "every card is NVIDIA's and CUDA runs on it: the native kernels and the urgent lane are CUDA's" },
    Rule { when: |hw| !hw.cards.is_empty(), ranked: &[Backend::Wgpu], why: "the portable GPU backend" },
    Rule { when: |_| true, ranked: &[Backend::Cpu], why: "no GPU" },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn card(index: u32, vendor_id: u32, gib: u64, cc: Option<(u32, u32)>) -> Card {
        Card { index, name: format!("card{index}"), vendor_id, vram_bytes: gib << 30, compute_capability: cc }
    }

    #[test]
    fn the_backend_follows_what_the_cards_can_run() {
        let backend = |hw: &Hardware| choose(BACKEND, hw, |_| Ok(())).unwrap().value;
        let p40s = Hardware { cards: vec![card(0, NVIDIA, 24, Some((6, 1))), card(1, NVIDIA, 24, Some((6, 1)))], cuda_unusable: None };
        assert_eq!(backend(&p40s), Backend::Cuda);
        let no_toolkit = Hardware { cuda_unusable: Some("no NVRTC".into()), ..p40s.clone() };
        assert_eq!(backend(&no_toolkit), Backend::Wgpu, "NVIDIA cards without a way to build kernels stay portable");
        let mixed = Hardware { cards: vec![card(0, NVIDIA, 24, Some((6, 1))), card(1, 0x1002, 16, None)], cuda_unusable: None };
        assert_eq!(backend(&mixed), Backend::Wgpu, "one card CUDA cannot drive keeps every card on the portable backend");
        assert_eq!(backend(&Hardware::default()), Backend::Cpu);
    }

    #[test]
    fn an_nvidia_card_driven_without_cuda_is_reported_with_the_reason() {
        let nvidia = vec![card(0, NVIDIA, 96, None)];
        let broken = Hardware { cards: nvidia.clone(), cuda_unusable: Some("libcuda.so.1 could not be loaded: not found".into()) };
        let notice = broken.cuda_fallback_notice(Backend::Wgpu).expect("silent degradation");
        assert!(notice.contains("libcuda.so.1") && notice.contains("wgpu") && notice.contains("card0"), "{notice}");
        // Nothing to report: CUDA drives it, the operator chose the CPU, no NVIDIA card, or CUDA works and wgpu was asked for.
        assert_eq!(broken.cuda_fallback_notice(Backend::Cuda), None);
        assert_eq!(broken.cuda_fallback_notice(Backend::Cpu), None);
        let amd = Hardware { cards: vec![card(0, 0x1002, 16, None)], cuda_unusable: Some("no driver".into()) };
        assert_eq!(amd.cuda_fallback_notice(Backend::Wgpu), None);
        let healthy = Hardware { cards: nvidia, cuda_unusable: None };
        assert_eq!(healthy.cuda_fallback_notice(Backend::Wgpu), None);
    }

    #[test]
    fn a_candidate_that_does_not_fit_falls_through_and_says_why() {
        const RULES: &[Rule<u32>] = &[
            Rule { when: |hw| hw.cards.len() > 1, ranked: &[16, 8, 4], why: "several cards" },
            Rule { when: |_| true, ranked: &[4, 2, 1], why: "one card" },
        ];
        let two = Hardware { cards: vec![card(0, NVIDIA, 8, None), card(1, NVIDIA, 8, None)], cuda_unusable: None };
        let chosen = choose(RULES, &two, |v| if v <= 8 { Ok(()) } else { Err(format!("{v} does not fit")) }).unwrap();
        assert_eq!(chosen.value, 8);
        assert!(chosen.why.starts_with("several cards") && chosen.why.contains("16 passed over: 16 does not fit"), "{}", chosen.why);
        assert_eq!(choose(RULES, &Hardware::default(), |_| Ok(())).unwrap(), Chosen { value: 4, why: "one card".into() });
        let none = choose(RULES, &two, |v| Err(format!("{v} no"))).unwrap_err();
        assert!(none.contains("16 passed over") && none.contains("4 passed over"), "{none}");
        let leanest = choose_or_leanest(RULES, &two, |v| Err(format!("{v} no")));
        assert_eq!(leanest.value, 4, "nothing fits: the leanest candidate, never a refusal");
        assert!(leanest.why.contains("none fits") && leanest.why.contains("16 passed over"), "{}", leanest.why);
    }

    #[test]
    fn the_lowest_compute_capability_needs_every_card_to_have_one() {
        let hw = Hardware { cards: vec![card(0, NVIDIA, 24, Some((8, 6))), card(1, NVIDIA, 24, Some((6, 1)))], cuda_unusable: None };
        assert_eq!(hw.min_compute_capability(), Some((6, 1)));
        let partial = Hardware { cards: vec![card(0, NVIDIA, 24, Some((8, 6))), card(1, 0x1002, 16, None)], cuda_unusable: None };
        assert_eq!(partial.min_compute_capability(), None);
    }
}
