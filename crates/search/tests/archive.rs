// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a quality-diversity archive must do for a search whose objective is a
//! speedrun, stated as behaviour rather than as implementation.
//!
//! The properties here are the ones the doom campaign's correctness rests on:
//! an elite is replaced by a faster one at equal achievement (that is the
//! whole speedrun mechanism), achievement still dominates speed, selection
//! favours the frontier without starving anything, and a full archive evicts
//! the cell it is least likely to draw rather than the best one it holds.

use data::rng::Rng;
use search::archive::{Admission, Archive, Niche, Worth};

fn niche(parts: &[i32]) -> Niche {
    Niche::new(parts)
}

/// Tolerance below which two achievements count as the same, so that time can
/// break the tie. Wider than the float noise on a score, narrower than the
/// step between two genuinely different outcomes.
const TOL: f32 = 1e-3;

#[test]
fn a_fresh_niche_is_admitted() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    assert_eq!(a.offer(niche(&[0, 0]), Worth::new(0.5, 100), "first"), Admission::Fresh);
    assert_eq!(a.len(), 1);
}

/// The speedrun mechanism. Same niche, same achievement, fewer tics: the
/// archive keeps the faster one. Without this the archive is a coverage map
/// and the campaign can never improve a time.
#[test]
fn at_equal_achievement_the_faster_run_wins() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[1, 1]), Worth::new(0.5, 500), "slow");
    assert_eq!(a.offer(niche(&[1, 1]), Worth::new(0.5, 200), "fast"), Admission::Improved);
    assert_eq!(a.get(&niche(&[1, 1])).unwrap().what, "fast");
    assert_eq!(a.get(&niche(&[1, 1])).unwrap().worth.cost, 200);
}

#[test]
fn at_equal_achievement_a_slower_run_is_rejected() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[1, 1]), Worth::new(0.5, 200), "fast");
    assert_eq!(a.offer(niche(&[1, 1]), Worth::new(0.5, 500), "slow"), Admission::Rejected);
    assert_eq!(a.get(&niche(&[1, 1])).unwrap().what, "fast");
}

/// Achievement dominates time, and it has to: a run that killed more of the
/// level is ahead of one that killed less however quickly it got there. Time
/// is the tiebreaker, never the objective on its own.
#[test]
fn a_greater_achievement_beats_a_faster_one() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[2, 2]), Worth::new(0.50, 100), "quick but short");
    assert_eq!(a.offer(niche(&[2, 2]), Worth::new(0.80, 900), "slow but far"), Admission::Improved);
    assert_eq!(a.get(&niche(&[2, 2])).unwrap().what, "slow but far");
}

#[test]
fn a_lesser_achievement_is_rejected_however_fast() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[2, 2]), Worth::new(0.80, 900), "slow but far");
    assert_eq!(a.offer(niche(&[2, 2]), Worth::new(0.50, 1), "instant but short"), Admission::Rejected);
    assert_eq!(a.get(&niche(&[2, 2])).unwrap().what, "slow but far");
}

/// Achievement within `tol` counts as equal so that time can decide. Without
/// a tolerance, a score that wobbles by a thousandth between two runs never
/// lets the faster one in, and the time never improves.
#[test]
fn achievement_within_tolerance_counts_as_equal_and_time_decides() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[3, 3]), Worth::new(0.500_0, 800), "slow");
    // +0.0002: inside TOL, so this is the SAME achievement and it is faster.
    assert_eq!(a.offer(niche(&[3, 3]), Worth::new(0.500_2, 300), "fast"), Admission::Improved);
    assert_eq!(a.get(&niche(&[3, 3])).unwrap().what, "fast");
    // -0.0002 and slower: same achievement, worse time, refused.
    assert_eq!(a.offer(niche(&[3, 3]), Worth::new(0.499_8, 900), "slower"), Admission::Rejected);
    assert_eq!(a.get(&niche(&[3, 3])).unwrap().what, "fast");
}

/// Two different niches are two different solutions, even when one scores
/// worse. That is the whole point of keeping an archive instead of a
/// leaderboard: the weaker one is a stepping stone the best-of-N would delete.
#[test]
fn different_niches_coexist_regardless_of_score() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[0, 0]), Worth::new(0.94, 100), "greedy");
    a.offer(niche(&[9, 9]), Worth::new(0.10, 100), "strange recursive trick");
    assert_eq!(a.len(), 2);
    assert_eq!(a.get(&niche(&[9, 9])).unwrap().what, "strange recursive trick");
}

/// Selection has to favour what has rarely been set off from, or the search
/// spends its whole budget on the hundreds of cells in the opening corridor.
#[test]
fn selection_favours_the_rarely_visited() {
    let mut a: Archive<u32> = Archive::new(64, TOL);
    for i in 0..8 {
        a.offer(niche(&[i, 0]), Worth::new(0.5, 100), i as u32);
    }
    let mut rng = Rng::new(11);
    let mut drawn = [0u32; 8];
    for _ in 0..4_000 {
        let n = a.pick(&mut rng).expect("archive is not empty");
        drawn[n.niche.parts()[0] as usize] += 1;
    }
    // With visits feeding back into the weight, 4000 draws over 8 equal cells
    // spread out instead of piling onto one. No cell may take more than a
    // third of the budget, and none may be starved.
    let most = *drawn.iter().max().unwrap();
    let least = *drawn.iter().min().unwrap();
    assert!(most < 4_000 / 3, "one cell took {most} of 4000 draws");
    assert!(least > 0, "a cell was never drawn at all");
}

/// A twofold preference for the best cell cannot push a frontier. An archive
/// that has been running a while holds thousands of cells of which a handful
/// are near the goal, so a flat preference spends almost the whole budget on
/// ground already covered - measured on the doom campaign, two runs held
/// their best score for 800 seconds while the archive grew from 649 to 1028
/// cells, finding new places the whole time and advancing not at all.
#[test]
fn selection_strongly_prefers_what_has_achieved_most() {
    let mut a: Archive<u32> = Archive::new(2048, TOL);
    // One cell at the goal, and two hundred that got a quarter of the way -
    // roughly the shape a real campaign settles into.
    a.offer(niche(&[0, 0]), Worth::new(1.0, 100), 0);
    for i in 1..=200 {
        a.offer(niche(&[i, 1]), Worth::new(0.25, 100), i as u32);
    }
    let mut rng = Rng::new(5);
    let mut leader = 0u32;
    for _ in 0..4_000 {
        if a.pick(&mut rng).expect("not empty").niche.parts()[1] == 0 {
            leader += 1;
        }
    }
    // Uniform-by-worth would give the single deep cell about 1 draw in 130.
    // It has to do far better than that to be a frontier the search pushes.
    assert!(
        leader > 4_000 / 40,
        "the one cell near the goal took only {leader} of 4000 draws against 200 shallow ones"
    );

    // The same archive with the preference turned off, as the control - so
    // this test demonstrates the difference rather than asserting a number
    // that some other weighting might also happen to clear.
    let mut flat: Archive<u32> = Archive::new(2048, TOL).focused(0.0);
    flat.offer(niche(&[0, 0]), Worth::new(1.0, 100), 0);
    for i in 1..=200 {
        flat.offer(niche(&[i, 1]), Worth::new(0.25, 100), i as u32);
    }
    let mut rng = Rng::new(5);
    let mut flat_leader = 0u32;
    for _ in 0..4_000 {
        if flat.pick(&mut rng).expect("not empty").niche.parts()[1] == 0 {
            flat_leader += 1;
        }
    }
    assert!(
        leader > flat_leader * 3,
        "preferring achievement barely changed anything: {leader} against a flat {flat_leader}"
    );
}

/// ...and never at the cost of cutting the shallow ones off. A stepping stone
/// that looks worthless is exactly what an archive like this exists to keep.
#[test]
fn a_far_worse_cell_is_still_drawn() {
    let mut a: Archive<u32> = Archive::new(16, TOL);
    a.offer(niche(&[0]), Worth::new(1.0, 100), 0);
    a.offer(niche(&[1]), Worth::new(0.01, 100), 1);
    let mut rng = Rng::new(7);
    let mut poor = 0;
    for _ in 0..4_000 {
        if a.pick(&mut rng).expect("not empty").niche.parts()[0] == 1 {
            poor += 1;
        }
    }
    assert!(poor > 20, "the weak cell was effectively cut off: {poor} of 4000");
}

/// A cell that led somewhere good keeps priority over one that achieved
/// nothing - but the one that achieved nothing stays in the draw, because
/// that is where a search that has not started yet has to begin.
#[test]
fn selection_prefers_worth_without_cutting_anything_off() {
    let mut a: Archive<u32> = Archive::new(64, TOL);
    a.offer(niche(&[0, 0]), Worth::new(1.0, 100), 0);
    a.offer(niche(&[1, 0]), Worth::new(0.0, 100), 1);
    let mut rng = Rng::new(3);
    let (mut rich, mut poor) = (0u32, 0u32);
    for _ in 0..2_000 {
        match a.pick(&mut rng).expect("not empty").niche.parts()[0] {
            0 => rich += 1,
            _ => poor += 1,
        }
    }
    assert!(rich > poor, "the worthwhile cell was not preferred: {rich} vs {poor}");
    assert!(poor > 0, "the cell that has achieved nothing was cut off entirely");
}

/// An archive that cannot grow is a search that has finished. When the slots
/// run out the weakest cell gives up its slot - and the slot NUMBER is reused,
/// because on a real environment a slot is a snapshot the host has to address.
#[test]
fn a_full_archive_evicts_the_weakest_and_reuses_its_slot() {
    let mut a: Archive<&str> = Archive::new(3, TOL);
    a.offer(niche(&[0, 0]), Worth::new(1.0, 100), "best");
    a.offer(niche(&[1, 0]), Worth::new(0.5, 100), "middle");
    a.offer(niche(&[2, 0]), Worth::new(0.01, 100), "weakest");
    let doomed = a.get(&niche(&[2, 0])).unwrap().slot;
    assert_eq!(a.offer(niche(&[3, 0]), Worth::new(0.7, 100), "newcomer"), Admission::Fresh);
    assert_eq!(a.len(), 3, "the archive grew past its capacity");
    assert!(a.get(&niche(&[2, 0])).is_none(), "the weakest cell survived");
    assert!(a.get(&niche(&[0, 0])).is_some(), "the BEST cell was evicted");
    assert_eq!(a.get(&niche(&[3, 0])).unwrap().slot, doomed, "the freed slot was not reused");
}

/// Every cell holds a distinct slot for as long as the archive is not full.
/// Two cells sharing one means one of them is restoring the other's snapshot,
/// which is a search that silently explores the wrong place.
#[test]
fn slots_are_unique() {
    let mut a: Archive<u32> = Archive::new(32, TOL);
    for i in 0..32 {
        a.offer(niche(&[i, 0]), Worth::new(0.5, 100), i as u32);
    }
    let mut slots: Vec<usize> = a.iter().map(|e| e.slot).collect();
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(slots.len(), 32, "two cells shared a slot");
}

/// An improved elite keeps its slot. Handing it a new one leaks the old
/// snapshot and burns through the host's slots for no reason.
#[test]
fn an_improved_elite_keeps_its_slot() {
    let mut a: Archive<&str> = Archive::new(8, TOL);
    a.offer(niche(&[5, 5]), Worth::new(0.5, 900), "slow");
    let slot = a.get(&niche(&[5, 5])).unwrap().slot;
    a.offer(niche(&[5, 5]), Worth::new(0.5, 100), "fast");
    assert_eq!(a.get(&niche(&[5, 5])).unwrap().slot, slot);
}

/// A cell that is improved says so, so that a caller which recorded a
/// trajectory as "resume at this cell, then..." can tell that the cell it
/// recorded against is gone. Without it the caller reconstructs a
/// plausible-looking sequence that no longer describes anything.
#[test]
fn an_improved_cell_reports_a_new_generation() {
    let mut a: Archive<&str> = Archive::new(8, TOL);
    a.offer(niche(&[1]), Worth::new(0.5, 900), "slow");
    assert_eq!(a.get(&niche(&[1])).unwrap().generation, 0);
    a.offer(niche(&[1]), Worth::new(0.5, 100), "fast");
    assert_eq!(a.get(&niche(&[1])).unwrap().generation, 1);
    // A REFUSED offer is not a change, so it must not move the generation -
    // or every chained trajectory is invalidated by an offer that changed
    // nothing.
    a.offer(niche(&[1]), Worth::new(0.5, 999), "slower");
    assert_eq!(a.get(&niche(&[1])).unwrap().generation, 1);
}

/// The best cell by achievement, then by time. What a campaign reports and
/// what the compression phase clones first.
#[test]
fn best_is_by_achievement_then_by_time() {
    let mut a: Archive<&str> = Archive::new(16, TOL);
    a.offer(niche(&[0, 0]), Worth::new(0.5, 100), "middling");
    a.offer(niche(&[1, 0]), Worth::new(0.9, 900), "furthest, slow");
    a.offer(niche(&[2, 0]), Worth::new(0.9, 400), "furthest, fast");
    assert_eq!(a.best().unwrap().what, "furthest, fast");
}

/// The archive outlives the process, or a campaign cannot compound. What
/// comes back has to be what went in - including visit counts, or a reloaded
/// archive re-explores its own opening corridor.
#[test]
fn an_archive_round_trips_through_json() {
    let mut a: Archive<String> = Archive::new(16, TOL);
    a.offer(niche(&[1, 2, 3]), Worth::new(0.75, 640), "a trail".to_string());
    a.offer(niche(&[4, 5, 6]), Worth::new(0.25, 120), "another".to_string());
    let mut rng = Rng::new(1);
    a.pick(&mut rng);
    let text = a.to_json().expect("an archive serialises");
    let back: Archive<String> = Archive::from_json(&text).expect("and comes back");
    assert_eq!(back.len(), a.len());
    assert_eq!(back.capacity(), a.capacity());
    for e in a.iter() {
        let there = back.get(&e.niche).expect("every cell survived");
        assert_eq!(there.what, e.what);
        assert_eq!(there.worth.cost, e.worth.cost);
        assert_eq!(there.visits, e.visits, "visit counts did not survive");
        assert_eq!(there.slot, e.slot);
    }
}

/// Coverage per axis, which is how a campaign says whether the search is
/// still finding new KINDS of situation or only refining what it has.
#[test]
fn coverage_counts_distinct_values_per_axis() {
    let mut a: Archive<u32> = Archive::new(16, TOL);
    a.offer(niche(&[0, 0, 7]), Worth::new(0.1, 1), 0);
    a.offer(niche(&[1, 0, 7]), Worth::new(0.1, 1), 1);
    a.offer(niche(&[2, 0, 7]), Worth::new(0.1, 1), 2);
    assert_eq!(a.coverage(), vec![3, 1, 1]);
}

/// A cell on the edge of the ground already reached is drawn more often than
/// one hemmed in by cells the archive already holds.
///
/// Go-Explore's frontier term. Without it a search spends its budget in
/// proportion to how much of a region it has already covered, which is
/// exactly backwards: the middle of a swept room is the least likely place
/// for anything new to be.
#[test]
fn the_edge_of_what_has_been_reached_is_preferred_to_the_middle_of_it() {
    // A row of cells along axis 0, plus one standing on its own far away.
    // Everything has achieved the same amount and been visited the same
    // number of times, so selection has nothing else to go on.
    let mut a: Archive<&str> = Archive::new(64, 1e-4).exploring(&[0], search::archive::EDGE);
    for x in 0..5 {
        a.offer(Niche::new(&[x, 0]), Worth::new(1.0, 10), "in the row");
    }
    a.offer(Niche::new(&[40, 0]), Worth::new(1.0, 10), "on its own");

    let hemmed_in = Niche::new(&[2, 0]);
    let alone = Niche::new(&[40, 0]);
    assert_eq!(a.get(&hemmed_in).map(|e| e.visits), Some(0));
    assert_eq!(a.get(&alone).map(|e| e.visits), Some(0));

    let mut rng = Rng::new(7);
    let (mut edge, mut middle) = (0, 0);
    for _ in 0..4_000 {
        let drawn = a.pick(&mut rng).map(|e| e.niche.clone()).expect("the archive is not empty");
        if drawn == alone {
            edge += 1;
        } else if drawn == hemmed_in {
            middle += 1;
        }
    }
    assert!(
        edge > middle,
        "the lone cell was drawn {edge} times against the hemmed-in one's {middle}"
    );
}

/// And an archive told of no axes selects exactly as it did before there was
/// a way to tell it: this is an opt-in, not a change to every caller.
#[test]
fn naming_no_axes_leaves_selection_alone() {
    let build = |axes: &[usize]| {
        let mut a: Archive<&str> = Archive::new(64, 1e-4).exploring(axes, search::archive::EDGE);
        for x in 0..5 {
            a.offer(Niche::new(&[x, 0]), Worth::new(1.0, 10), "row");
        }
        a.offer(Niche::new(&[40, 0]), Worth::new(1.0, 10), "alone");
        a
    };
    let mut plain = build(&[]);
    let mut rng = Rng::new(7);
    let mut lone = 0;
    for _ in 0..4_000 {
        if plain.pick(&mut rng).map(|e| e.niche.clone()) == Some(Niche::new(&[40, 0])) {
            lone += 1;
        }
    }
    // Six cells, all equal, no frontier term: about a sixth of the draws.
    let share = lone as f32 / 4_000.0;
    assert!((share - 1.0 / 6.0).abs() < 0.03, "share was {share}");
}
