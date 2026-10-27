// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multi-GPU placement for document-OCR serving.
// If your team needs expertise in fitting vision-language models onto the cards
// it owns then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Where the two halves of an OCR model go: the vision encoder on one card, the
//! decoder on another card, the same card or the host - and how many whole
//! replicas the offered cards can run at once.

/// What one OCR model holds while hot. Every figure is a measurement of the
/// served build (`nvidia-smi memory.used` per card, `/proc/self/status` VmHWM),
/// recorded with its workload in the model's roadmap ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Device bytes of the vision encoder while hot.
    pub vision_device: u64,
    /// Device bytes of the decoder engine (weights, paged KV pool, scratch).
    pub decoder_device: u64,
    /// Host bytes once the decoder is on a card (tokenizer, preprocessing, per-token work).
    pub host_gpu_decoder: u64,
    /// Host bytes with the decoder on the CPU backend.
    pub host_cpu_decoder: u64,
}

impl Footprint {
    /// Peak footprint of the WHOLE model with the decoder on the host: what a
    /// GPU-less box pays entirely out of RAM (a software rasteriser's buffers are host RAM).
    pub fn composite_peak(&self) -> u64 {
        self.host_cpu_decoder + self.vision_device
    }
}

/// Where the decoder is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecoderDevice {
    /// The CPU Cranelift JIT (`gpu_core::Gpu::new_cpu`).
    Cpu,
    /// The ambient card - the one the vision encoder resolves to.
    SameCard,
    /// Card `i`, so the two halves come out of two different pools.
    Card(u32),
}

/// Where the two halves of the model go.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Placement {
    /// The card the vision encoder is built on, or `None` when no offered card can hold it.
    pub vision: Option<u32>,
    /// Where the decoder is built.
    pub decoder: DecoderDevice,
}

/// Place the model on the `cards` it may use, each `(index, usable bytes)`.
///
/// The caller owns which cards those are: a server passes the cards it
/// schedules (`--device`), so a placement can never name a card the
/// residency budgets do not hold - a cost naming an unbudgeted card is
/// refused by the scheduler.
///
/// 1. The vision encoder goes on the card with the most usable bytes that can
///    hold it at all (a single-image graph cannot be spread).
/// 2. The decoder goes on another card that can hold it (largest first) so the
///    two pools do not compete, else on the vision card when it holds both,
///    else on the CPU backend - a decoder that does not fit is not one that
///    runs slowly, it is one that fails to allocate.
/// 3. With no card for the encoder, everything is host-side.
pub fn plan_placement(cards: &[(u32, u64)], fp: &Footprint) -> Placement {
    let vision = cards.iter().filter(|&&(_, usable)| usable >= fp.vision_device).max_by_key(|&&(_, usable)| usable).map(|&(i, _)| i);
    let Some(v) = vision else { return Placement { vision: None, decoder: DecoderDevice::Cpu } };
    let other = cards.iter().filter(|&&(i, usable)| i != v && usable >= fp.decoder_device).max_by_key(|&&(_, usable)| usable);
    let own = cards.iter().find(|&&(i, _)| i == v).map_or(0, |&(_, usable)| usable);
    let decoder = match other {
        Some(&(i, _)) => DecoderDevice::Card(i),
        None if own >= fp.vision_device + fp.decoder_device => DecoderDevice::SameCard,
        None => DecoderDevice::Cpu,
    };
    Placement { vision: Some(v), decoder }
}

/// The placements of every replica the offered cards can run: one self-contained replica (encoder and decoder on the same card) on
/// each card that holds both, when there are two or more such cards - the spare capacity is then used, and two replicas never wait on
/// each other's card. With fewer, the one placement of [`plan_placement`] over the free memory. Never empty.
///
/// `cards` are `(index, free bytes)` now; `capacities` are `(index, total bytes)`. A replica is planned on every card whose CAPACITY
/// holds the model, free now or not: the scheduler runs a job on a replica whose card has the budget at that moment, so a card a
/// neighbouring process holds when the server starts gets its replica as soon as the capacity probe sees the card free.
pub fn plan_replicas(cards: &[(u32, u64)], capacities: &[(u32, u64)], fp: &Footprint) -> Vec<Placement> {
    let need = fp.vision_device + fp.decoder_device;
    let capacity_of = |i: u32, free: u64| capacities.iter().find(|&&(c, _)| c == i).map_or(free, |&(_, total)| total.max(free));
    let whole: Vec<Placement> = cards.iter().filter(|&&(i, free)| capacity_of(i, free) >= need).map(|&(i, _)| Placement { vision: Some(i), decoder: DecoderDevice::SameCard }).collect();
    if whole.len() >= 2 {
        whole
    } else {
        vec![plan_placement(cards, fp)]
    }
}

/// [`plan_placement`] for a one-shot process that has no scheduler: the cards
/// the process may use are the ambient pin when `--device gpuN` set one, else
/// every discrete card. Capacity, not free memory, because the backends expose
/// no portable free-VRAM query; a server uses its budgets instead.
pub fn placement_for_process(fp: &Footprint) -> Placement {
    let pinned = gpu_core::devices::ambient_gpu();
    let cards: Vec<(u32, u64)> = gpu_core::devices::gpus()
        .iter()
        .filter(|d| d.identity.class == gpu_core::DeviceClass::DiscreteGpu && pinned.is_none_or(|p| p == d.index))
        .map(|d| (d.index, d.identity.vram_bytes))
        .collect();
    plan_placement(&cards, fp)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const FP: Footprint = Footprint { vision_device: 7 * GIB, decoder_device: 14 * GIB, host_gpu_decoder: 3 * GIB, host_cpu_decoder: 14 * GIB };

    /// The placement only ever names cards it was offered. A server offers the
    /// cards `--device` schedules; naming any other card made the residency
    /// claim wait for a budget that does not exist.
    #[test]
    fn placement_names_only_offered_cards() {
        assert_eq!(plan_placement(&[(0, 90 * GIB)], &FP), Placement { vision: Some(0), decoder: DecoderDevice::SameCard });
        assert_eq!(plan_placement(&[(1, 90 * GIB)], &FP), Placement { vision: Some(1), decoder: DecoderDevice::SameCard }, "an offered card keeps its own index");
    }

    /// Spare cards are used: a replica on every card that holds the whole model, but a single
    /// replica when only one does (its decoder may then go to a second card).
    #[test]
    fn a_replica_runs_on_every_card_that_holds_the_whole_model() {
        let whole = Placement { vision: Some(0), decoder: DecoderDevice::SameCard };
        let whole_cards = [(0, 90 * GIB), (1, 90 * GIB)];
        assert_eq!(plan_replicas(&whole_cards, &whole_cards, &FP), vec![whole, Placement { vision: Some(1), decoder: DecoderDevice::SameCard }]);
        assert_eq!(plan_replicas(&[(0, 90 * GIB), (1, 12 * GIB)], &[(0, 90 * GIB), (1, 12 * GIB)], &FP), vec![plan_placement(&[(0, 90 * GIB), (1, 12 * GIB)], &FP)], "one whole card is one replica");
        let three = [(0, 90 * GIB), (1, 60 * GIB), (3, 8 * GIB)];
        assert_eq!(plan_replicas(&three, &three, &FP).len(), 2, "a card that cannot hold the model is left out");
        assert_eq!(plan_replicas(&[], &[], &FP), vec![Placement { vision: None, decoder: DecoderDevice::Cpu }]);
        // A neighbour holds most of card 1 when the server starts, but the card is big enough: its replica is planned, to be used once it frees.
        let held = [(0, 90 * GIB), (1, 12 * GIB)];
        let big = [(0, 96 * GIB), (1, 96 * GIB)];
        assert_eq!(plan_replicas(&held, &big, &FP).len(), 2, "capacity, not the free memory of the moment, decides the replicas");
    }

    #[test]
    fn the_decoder_prefers_a_second_offered_card_then_the_vision_card_then_the_cpu() {
        assert_eq!(plan_placement(&[(0, 24 * GIB), (1, 30 * GIB)], &FP), Placement { vision: Some(1), decoder: DecoderDevice::Card(0) });
        assert_eq!(plan_placement(&[(0, 24 * GIB)], &FP), Placement { vision: Some(0), decoder: DecoderDevice::SameCard });
        assert_eq!(plan_placement(&[(0, 10 * GIB)], &FP).decoder, DecoderDevice::Cpu, "the encoder fits, the decoder does not");
        assert_eq!(plan_placement(&[(0, 2 * GIB)], &FP), Placement { vision: None, decoder: DecoderDevice::Cpu });
        assert_eq!(plan_placement(&[], &FP), Placement { vision: None, decoder: DecoderDevice::Cpu });
    }
}
