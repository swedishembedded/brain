// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cost-sensitive decision theory: turning a calibrated probability into an
//! action.
//!
//! A repo-wide search before this module was written found no cost matrix,
//! no expected-loss minimization, and no value of information anywhere in
//! brain. `decide::routing::Router` is the nearest neighbour and is not this:
//! it is a fixed confidence-threshold cascade fitted against *correctness*,
//! with no notion that being wrong in different ways can cost different
//! amounts.
//!
//! The separation this module is built around: **the probability is not the
//! action**. `argmax p` answers "what outcome is most likely", not "what
//! should I do about it" - those coincide only when every wrong action costs
//! the same. [`bayes_action`] is the deterministic, non-learned function that
//! turns a probability distribution plus an explicit [`CostMatrix`] into the
//! action that minimizes expected cost - no policy gradient needed, because
//! this is a closed-form `argmin` over a distribution the model already
//! produced.
//!
//! Swedish Embedded AB builds decision systems that act on calibrated
//! probabilities according to the ACTUAL cost of being wrong, not merely the
//! most likely label. If your team needs expertise in cost-sensitive
//! decision-making under uncertainty, you can procure our services by
//! sending an email to info@swedishembedded.com.

/// `cost[action][outcome]` - the cost of taking `action` when the true
/// outcome is `outcome`. Actions and outcomes are independent index spaces
/// (an inspect action has no corresponding "outcome", for instance), so the
/// matrix need not be square.
#[derive(Clone, Debug)]
pub struct CostMatrix {
    n_actions: usize,
    n_outcomes: usize,
    // Row-major: cost of (action, outcome) at `data[action * n_outcomes + outcome]`.
    data: Vec<f32>,
}

impl CostMatrix {
    /// `rows[a][y]` is the cost of action `a` when the outcome is `y`. Every
    /// row must have the same length (asserted, since a ragged cost matrix
    /// has no well-defined `n_outcomes`).
    pub fn from_rows(rows: &[Vec<f32>]) -> CostMatrix {
        let n_actions = rows.len();
        let n_outcomes = rows.first().map(|r| r.len()).unwrap_or(0);
        for (a, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                n_outcomes,
                "action {a} has {} outcomes, row 0 has {n_outcomes}",
                row.len()
            );
        }
        let mut data = Vec::with_capacity(n_actions * n_outcomes);
        for row in rows {
            data.extend_from_slice(row);
        }
        CostMatrix {
            n_actions,
            n_outcomes,
            data,
        }
    }

    /// The binary decision cost matrix behind the worked example this module
    /// is gated against: two actions (`0` = the conservative one, e.g.
    /// "block"; `1` = the permissive one, e.g. "release") over two outcomes
    /// (`0` = the benign one, e.g. "healthy"; `1` = the costly one, e.g.
    /// "faulty"). `cost_fp` is the cost of the conservative action on a
    /// benign outcome (a false positive); `cost_fn` is the cost of the
    /// permissive action on the costly outcome (a false negative). Acting
    /// correctly costs nothing either way.
    pub fn binary(cost_fp: f32, cost_fn: f32) -> CostMatrix {
        CostMatrix::from_rows(&[vec![cost_fp, 0.0], vec![0.0, cost_fn]])
    }

    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    pub fn n_outcomes(&self) -> usize {
        self.n_outcomes
    }

    pub fn get(&self, action: usize, outcome: usize) -> f32 {
        self.data[action * self.n_outcomes + outcome]
    }

    /// `rows[a][y]`, the exact inverse of [`CostMatrix::from_rows`] - what a
    /// caller needs to write a matrix into a checkpoint and read it back.
    pub fn rows(&self) -> Vec<Vec<f32>> {
        (0..self.n_actions).map(|a| (0..self.n_outcomes).map(|y| self.get(a, y)).collect()).collect()
    }
}

/// `E_Y[cost(action, Y)]` under the distribution `p` over outcomes.
pub fn expected_cost(p: &[f32], costs: &CostMatrix, action: usize) -> f32 {
    assert_eq!(
        p.len(),
        costs.n_outcomes(),
        "p has {} entries, cost matrix has {} outcomes",
        p.len(),
        costs.n_outcomes()
    );
    (0..costs.n_outcomes())
        .map(|y| p[y] * costs.get(action, y))
        .sum()
}

/// The action-selection result: the minimizer, its expected cost (the Bayes
/// risk), and every OTHER action within floating tolerance of the minimum -
/// reported explicitly rather than silently broken by index order, since a
/// tie is exactly the boundary case a cost sweep is built to cross.
#[derive(Clone, Debug)]
pub struct BayesAction {
    pub action: usize,
    pub risk: f32,
    pub ties: Vec<usize>,
}

const TIE_TOLERANCE: f32 = 1e-6;

/// `argmin_a E_Y[cost(a, Y)]` under `p` - the deterministic, non-learned
/// mapping from a calibrated probability to an action. See this module's doc
/// for why this is a closed-form computation and not a policy.
pub fn bayes_action(p: &[f32], costs: &CostMatrix) -> BayesAction {
    assert!(
        costs.n_actions() > 0,
        "a cost matrix with no actions has no Bayes action"
    );
    let per_action: Vec<f32> = (0..costs.n_actions())
        .map(|a| expected_cost(p, costs, a))
        .collect();
    let (best, &risk) = per_action
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("cost is never NaN"))
        .unwrap();
    let ties = (0..costs.n_actions())
        .filter(|&a| a != best && (per_action[a] - risk).abs() <= TIE_TOLERANCE)
        .collect();
    BayesAction {
        action: best,
        risk,
        ties,
    }
}

/// `min_a E_Y[cost(a, Y)]` under `p` - the Bayes risk, i.e. the best
/// achievable expected cost given only the belief `p` (before any further
/// evidence).
pub fn bayes_risk(p: &[f32], costs: &CostMatrix) -> f32 {
    bayes_action(p, costs).risk
}

/// How much worse `taken_action` is than the Bayes-optimal action, under the
/// TRUE outcome distribution `p_true` - the number that distinguishes a
/// model that learned a calibrated belief from one that merely learned to
/// act plausibly. Always >= 0; `0` exactly iff `taken_action` is (one of) the
/// Bayes-optimal action(s).
pub fn regret(p_true: &[f32], costs: &CostMatrix, taken_action: usize) -> f32 {
    expected_cost(p_true, costs, taken_action) - bayes_risk(p_true, costs)
}

/// Value of information for a query `m` with `query_cost = c(m)`, computed
/// exactly as
///
/// ```text
/// VOI(m | x) = R*(x; L) - E_z[R*(x, z; L)] - c(m)
/// ```
///
/// `outcome_probs[i]` is the probability of the query's `i`-th possible
/// result, and `posteriors[i]` is the belief over outcomes GIVEN that result
/// (`sum_i outcome_probs[i] * posteriors[i] == prior`, the information-
/// refinement identity [`crate::atlas`] checks on a whole generated world).
/// Positive VOI means querying first is worth it; the caller compares this
/// against acting immediately, not against zero in isolation, since a
/// negative VOI still has to be weighed against the (still nonzero) risk of
/// acting now.
pub fn voi(
    prior: &[f32],
    costs: &CostMatrix,
    outcome_probs: &[f32],
    posteriors: &[Vec<f32>],
    query_cost: f32,
) -> f32 {
    assert_eq!(
        outcome_probs.len(),
        posteriors.len(),
        "one posterior per possible query outcome"
    );
    let risk_before = bayes_risk(prior, costs);
    let risk_after: f32 = outcome_probs
        .iter()
        .zip(posteriors)
        .map(|(&pz, post)| pz * bayes_risk(post, costs))
        .sum();
    risk_before - risk_after - query_cost
}

#[cfg(test)]
mod tests {
    /// A cost matrix has to survive being written down and read back, or a
    /// saved decision model cannot describe the costs it was evaluated
    /// under - see `brain::RlcdPipeline`'s head checkpoint.
    #[test]
    fn rows_round_trips_through_from_rows() {
        for m in [CostMatrix::binary(1.0, 10.0), CostMatrix::from_rows(&[vec![0.0, 3.0, 1.0], vec![2.0, 0.0, 5.0]])] {
            let back = CostMatrix::from_rows(&m.rows());
            assert_eq!(back.n_actions(), m.n_actions());
            assert_eq!(back.n_outcomes(), m.n_outcomes());
            for a in 0..m.n_actions() {
                for y in 0..m.n_outcomes() {
                    assert_eq!(back.get(a, y), m.get(a, y));
                }
            }
        }
    }

    use super::*;

    // The device-diagnosis worked example this crate's design was built
    // around: P(fault) = 0.2, P(+ | fault) = 0.8, P(+ | healthy) = 0.1.
    // Actions 0 = block, 1 = release; outcomes 0 = healthy, 1 = faulty.
    const P_NONE: [f32; 2] = [0.8, 0.2];
    const P_POSITIVE: [f32; 2] = [1.0 / 3.0, 2.0 / 3.0];
    const P_NEGATIVE: [f32; 2] = [18.0 / 19.0, 1.0 / 19.0];
    const P_POSITIVE_RESULT: f32 = 0.24; // P(+) = 0.24*P(F|+) ... = 0.2*0.8 + 0.8*0.1
    const P_NEGATIVE_RESULT: f32 = 0.76;

    #[test]
    fn the_boundary_action_matches_the_worked_example() {
        let costs = CostMatrix::binary(1.0, 10.0); // C_FP = 1, C_FN = 10, boundary = 1/11
        assert_eq!(
            bayes_action(&P_NONE, &costs).action,
            0,
            "block: below the 1/11 boundary"
        );
        assert_eq!(
            bayes_action(&P_POSITIVE, &costs).action,
            0,
            "block: 2/3 is above 1/11"
        );
        assert_eq!(
            bayes_action(&P_NEGATIVE, &costs).action,
            1,
            "release: 1/19 is below 1/11"
        );
    }

    #[test]
    fn bayes_risk_matches_the_worked_example() {
        let costs = CostMatrix::binary(1.0, 10.0);
        let risk_before = bayes_risk(&P_NONE, &costs);
        assert!(
            (risk_before - 0.8).abs() <= 1e-6,
            "risk before the test: {risk_before} vs 0.8"
        );

        let risk_after = P_POSITIVE_RESULT * bayes_risk(&P_POSITIVE, &costs)
            + P_NEGATIVE_RESULT * bayes_risk(&P_NEGATIVE, &costs);
        assert!(
            (risk_after - 0.48).abs() <= 1e-6,
            "risk after the test: {risk_after} vs 0.48"
        );
    }

    #[test]
    fn voi_matches_the_worked_example() {
        let costs = CostMatrix::binary(1.0, 10.0);
        let value = voi(
            &P_NONE,
            &costs,
            &[P_POSITIVE_RESULT, P_NEGATIVE_RESULT],
            &[P_POSITIVE.to_vec(), P_NEGATIVE.to_vec()],
            0.05,
        );
        assert!(
            (value - 0.27).abs() <= 1e-6,
            "VOI: {value} vs 0.27 (inspect-first should be worth it)"
        );
    }

    /// Raising the cost of a false positive from 1 to 4 moves the boundary
    /// from 1/11 to 4/14 and flips the no-evidence action from block to
    /// release - WITHOUT moving the probability at all. This is the whole
    /// point of separating belief from action: the same `P(fault) = 0.2`
    /// supports either action depending on what a mistake costs.
    #[test]
    fn a_cost_sweep_changes_the_action_not_the_probability() {
        let cheap_fp = CostMatrix::binary(1.0, 10.0);
        let expensive_fp = CostMatrix::binary(4.0, 10.0);
        assert_eq!(
            bayes_action(&P_NONE, &cheap_fp).action,
            0,
            "cheap false positive: block"
        );
        assert_eq!(
            bayes_action(&P_NONE, &expensive_fp).action,
            1,
            "expensive false positive: release"
        );
        // P_NONE itself is a fixed input to both calls - the belief never changed.
    }

    #[test]
    fn regret_is_zero_at_the_bayes_action_and_positive_elsewhere() {
        let costs = CostMatrix::binary(1.0, 10.0);
        let optimal = bayes_action(&P_POSITIVE, &costs).action;
        assert!((regret(&P_POSITIVE, &costs, optimal)).abs() <= 1e-6);
        let suboptimal = 1 - optimal;
        assert!(
            regret(&P_POSITIVE, &costs, suboptimal) > 0.0,
            "acting against the belief must cost strictly more"
        );
    }

    #[test]
    fn a_three_way_tie_is_reported_not_silently_broken() {
        // Three actions with identical expected cost under a uniform belief.
        let costs = CostMatrix::from_rows(&[vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]]);
        let result = bayes_action(&[0.5, 0.5], &costs);
        assert_eq!(result.action, 0, "the first minimizer wins ties by index");
        assert_eq!(
            result.ties,
            vec![1, 2],
            "the other two are reported as ties, not hidden"
        );
    }

    #[test]
    fn expected_cost_of_a_certain_outcome_reads_the_matrix_directly() {
        let costs = CostMatrix::binary(1.0, 10.0);
        assert_eq!(
            expected_cost(&[1.0, 0.0], &costs, 0),
            1.0,
            "blocking a known-healthy device: C_FP"
        );
        assert_eq!(
            expected_cost(&[0.0, 1.0], &costs, 0),
            0.0,
            "blocking a known-faulty device: correct, no cost"
        );
        assert_eq!(
            expected_cost(&[0.0, 1.0], &costs, 1),
            10.0,
            "releasing a known-faulty device: C_FN"
        );
        assert_eq!(
            expected_cost(&[1.0, 0.0], &costs, 1),
            0.0,
            "releasing a known-healthy device: correct, no cost"
        );
    }
}
