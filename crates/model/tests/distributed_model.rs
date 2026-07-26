// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end distributed training against the real [`Model`] trait, over both
//! transports. A minimal but genuine `Model` (interior-mutable linear regression,
//! one weight vector, MSE to a per-replica target) is trained data-parallel by
//! `DdpOptimizer`: each rank all-reduces its gradient through a `Collective` and
//! applies the identical AdamW step, so the replicas converge to the mean target
//! and stay bit-identical. Run once through `HostCollective` (threads, the
//! single-box path) and once through `NetworkCollective` (loopback TCP, the
//! cluster path) — same model, same driver, only the transport differs — then a
//! `federated_average` round. This is the "works for any model, scales from box
//! to cluster to federated" claim, exercised end to end.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use model::collective::Collective;
use model::{federated_average, Batch, DdpOptimizer, HostCollective, Model, ModelConfig, NetworkCollective};
use std::net::TcpListener;

// ---- a minimal real Model: w ∈ R^D, loss = Σ (w - target)², grad = 2(w - target) ----

#[derive(Clone)]
struct LinCfg {
    d: usize,
}
impl ModelConfig for LinCfg {
    fn param_list(&self) -> Vec<(String, usize)> {
        vec![("w".into(), self.d)]
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "d": self.d })
    }
    fn from_json(v: &serde_json::Value) -> Self {
        LinCfg { d: v["d"].as_u64().unwrap() as usize }
    }
    fn vocab(&self) -> u32 {
        0
    }
    fn block_size(&self) -> u32 {
        0
    }
    fn finalize_for_dataset(self, _v: u32, _b: u32) -> Self {
        self
    }
}

struct Lin {
    cfg: LinCfg,
    w: RefCell<Vec<f32>>,
    grad: RefCell<Vec<f32>>,
    target: Vec<f32>,
}
impl Lin {
    fn with_target(w0: Vec<f32>, target: Vec<f32>) -> Lin {
        let d = w0.len();
        Lin { cfg: LinCfg { d }, w: RefCell::new(w0), grad: RefCell::new(vec![0.0; d]), target }
    }
}
impl Model for Lin {
    type Config = LinCfg;
    fn new(cfg: LinCfg, _b: u32, _t: u32, init: &HashMap<String, Vec<f32>>) -> Self {
        let w = init.get("w").cloned().unwrap_or_else(|| vec![0.0; cfg.d]);
        Lin { cfg: cfg.clone(), w: RefCell::new(w), grad: RefCell::new(vec![0.0; cfg.d]), target: vec![0.0; cfg.d] }
    }
    fn init_weights(cfg: &LinCfg, _seed: u64) -> HashMap<String, Vec<f32>> {
        HashMap::from([("w".to_string(), vec![0.0; cfg.d])])
    }
    fn config(&self) -> &LinCfg {
        &self.cfg
    }
    fn set_batch(&self, _b: Batch) {}
    fn forward(&self) -> f32 {
        self.w.borrow().iter().zip(&self.target).map(|(&w, &t)| (w - t) * (w - t)).sum()
    }
    fn backward(&self) {
        let w = self.w.borrow();
        let mut g = self.grad.borrow_mut();
        for i in 0..w.len() {
            g[i] += 2.0 * (w[i] - self.target[i]);
        }
    }
    fn zero_grads(&self) {
        self.grad.borrow_mut().iter_mut().for_each(|x| *x = 0.0);
    }
    fn adamw_step(&self, _t: u32, _lr: f32, _wd: f32, _c: Option<f32>, _s: f32) {}
    fn poll_wait(&self) {}
    fn param_names(&self) -> Vec<String> {
        vec!["w".into()]
    }
    fn read_weight(&self, _n: &str) -> Vec<f32> {
        self.w.borrow().clone()
    }
    fn write_weight(&self, _n: &str, data: &[f32]) {
        self.w.borrow_mut().copy_from_slice(data);
    }
    fn read_grad(&self, _n: &str) -> Vec<f32> {
        self.grad.borrow().clone()
    }
    fn logits_all(&self, _t: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, _p: &str) {}
    fn config_json(&self) -> serde_json::Value {
        self.cfg.to_json()
    }
}

/// Train `world` replicas data-parallel through `make_coll(rank)`; each replica
/// targets a distinct vector. Returns each rank's final weights.
fn train_dp(world: usize, targets: Vec<Vec<f32>>, make_coll: impl Fn(usize) -> Arc<dyn Collective> + Send + Sync) -> Vec<Vec<f32>> {
    let d = targets[0].len();
    let out: Vec<std::sync::Mutex<Vec<f32>>> = (0..world).map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let make_coll = &make_coll;
    let targets = &targets;
    let out = &out;
    std::thread::scope(|s| {
        for r in 0..world {
            s.spawn(move || {
                let coll = make_coll(r);
                let model = Lin::with_target(vec![0.0; d], targets[r].clone());
                let mut opt = DdpOptimizer::new(&model);
                for t in 1..=400u32 {
                    model.zero_grads();
                    model.forward();
                    model.backward();
                    opt.step(&model, &*coll, r, t, 0.05, 0.0, None);
                }
                *out[r].lock().unwrap() = model.read_weight("w");
            });
        }
    });
    out.iter().map(|m| m.lock().unwrap().clone()).collect()
}

#[test]
fn data_parallel_converges_to_mean_target_over_host() {
    // 3 replicas, targets t0,t1,t2 → DDP drives w to their mean.
    let targets = vec![vec![1.0f32, 4.0], vec![3.0, 0.0], vec![2.0, 2.0]];
    let coll = HostCollective::new(3);
    let ws = train_dp(3, targets.clone(), move |_r| coll.clone() as Arc<dyn Collective>);
    let mean: Vec<f32> = (0..2).map(|i| targets.iter().map(|t| t[i]).sum::<f32>() / 3.0).collect();
    for w in &ws {
        assert_eq!(*w, ws[0], "replicas diverged");
        for (a, b) in w.iter().zip(&mean) {
            assert!((a - b).abs() < 1e-3, "host DDP: {a} vs mean {b}");
        }
    }
}

#[test]
fn data_parallel_converges_over_network() {
    // Same model + driver, but the collective is TCP (cluster path).
    let targets = vec![vec![1.0f32, 4.0], vec![3.0, 0.0]];
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let listener = Arc::new(std::sync::Mutex::new(Some(listener)));
    let ws = train_dp(2, targets.clone(), move |r| {
        if r == 0 {
            let l = listener.lock().unwrap().take().unwrap();
            Arc::new(NetworkCollective::coordinator(&l, 2).unwrap()) as Arc<dyn Collective>
        } else {
            // brief retry so the coordinator has time to start accepting.
            loop {
                if let Ok(c) = NetworkCollective::worker(r, 2, &addr) {
                    break Arc::new(c) as Arc<dyn Collective>;
                }
                std::thread::yield_now();
            }
        }
    });
    let mean: Vec<f32> = (0..2).map(|i| (targets[0][i] + targets[1][i]) / 2.0).collect();
    for w in &ws {
        assert_eq!(*w, ws[0], "replicas diverged over network");
        for (a, b) in w.iter().zip(&mean) {
            assert!((a - b).abs() < 1e-3, "net DDP: {a} vs mean {b}");
        }
    }
}

#[test]
fn federated_round_averages_node_models() {
    // Two nodes trained to different local optima, then one FedAvg round with
    // equal sample counts → the mean of their weights.
    let a = Lin::with_target(vec![10.0, -2.0], vec![0.0, 0.0]);
    let b = Lin::with_target(vec![0.0, 6.0], vec![0.0, 0.0]);
    let coll = HostCollective::new(2);
    let (wa, wb) = std::thread::scope(|s| {
        let coll = &coll;
        let h0 = s.spawn(move || {
            federated_average(&a, &**coll, 0, 1.0);
            a.read_weight("w")
        });
        let h1 = s.spawn(move || {
            federated_average(&b, &**coll, 1, 1.0);
            b.read_weight("w")
        });
        (h0.join().unwrap(), h1.join().unwrap())
    });
    assert_eq!(wa, vec![5.0, 2.0]);
    assert_eq!(wb, vec![5.0, 2.0]);
}
