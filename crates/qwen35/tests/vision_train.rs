// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language agents on a client's own verified trajectories. If
// your team needs expertise in training a model on the images it will be served, then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Training on records with images reads them as serving does.
//!
//! A tiny decoder and a tiny vision tower with random weights. The training forward on a record whose user turn
//! holds an image has the logits serving produces for the same prompt (the tower's rows spliced where serving
//! splices them, at the rotary positions serving gives them); a text batch after an image batch carries nothing over;
//! a batch of several rows treats each as it would alone; and a record the tower cannot read is refused by name
//! rather than trained as text.

mod common;

use common::vision_fixture::{picture, ppm, tiny_spec, tiny_tower_weights, add_vision_release, TOKENS};
use data::chat::ImageReader;
use data::chat_media;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::{Batch, Model, Split};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};
use qwen35::vision_frontend::{prompt_embeds, prompt_embeds_remembering, EncodedImages};
use qwen35::vision_sft::{attach_for_dataset, check_prepared, VisionFeed, VisionReader};
use qwen35::vision_tower::VisionTower;
use qwen3vl::mrope::PromptRope;

/// The two forwards run the same kernels on the CPU backend over the same rows, so they agree to rounding. The
/// tiny random model moves its logits by about 1e-4 when its positions or its picture change; the tolerance is a tenth
/// of that.
const TOLERANCE: f32 = 1e-5;

/// An 8 x 8 picture is four decoder rows in the tiny geometry.
const ROWS: usize = 4;

/// `[user text, image, user text]`: three text tokens, the image block, three text tokens.
fn prompt() -> Vec<u32> {
    let mut ids = vec![1, 2, 3, TOKENS.start];
    ids.extend([TOKENS.image; ROWS]);
    ids.extend([TOKENS.end, 4, 5, 6]);
    ids
}

fn placeholder(token: u32) -> bool {
    [TOKENS.start, TOKENS.image, TOKENS.end].contains(&token)
}

/// Next-token targets of `ids` with the placeholder block (the image's rows and the markers around it) unsupervised.
fn targets_of(ids: &[u32]) -> Vec<u32> {
    (0..ids.len())
        .map(|i| match ids.get(i + 1) {
            Some(&next) if !placeholder(next) && !placeholder(ids[i]) => next,
            Some(_) => model::IGNORE,
            None => 7,
        })
        .collect()
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-vision-train-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn head(head_w: &[f32], hidden: &[f32]) -> Vec<f32> {
    head_w.chunks_exact(hidden.len()).map(|r| r.iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

struct Rig {
    cfg: Qwen35Config,
    init: std::collections::HashMap<String, Vec<f32>>,
    gpu: Gpu,
}

impl Rig {
    fn new() -> Rig {
        let cfg = Qwen35Config::tiny();
        let init = qwen35::init::init_weights(&cfg, 11);
        Rig { cfg, init, gpu: Gpu::new_cpu(pipelines()) }
    }

    fn tower(&self) -> VisionTower {
        VisionTower::from_weights(tiny_spec(), &tiny_tower_weights(3), &self.gpu, self.cfg.d_model, 1 << 20).unwrap()
    }

    /// A training build of `rows` rows of `len` tokens, reading the image of example 0 of a dataset holding `image`.
    fn trainer(&self, rows: u32, len: u32, image: &imaging::pixels::Rgb8, tag: &str) -> Qwen35 {
        let dir = scratch(tag);
        chat_media::write(&dir, "train", &[vec![ppm(image).into()], vec![]]).unwrap();
        let m = Qwen35::new_train_on(self.gpu.share(), self.cfg.clone(), rows, len, &self.init);
        m.attach_vision(VisionFeed::open(&dir, self.tower()).unwrap().expect("a dataset with images"));
        m
    }
}

/// What serving computes for `ids` with `image`: the embedding gather with the tower's rows over the placeholders, at
/// the rotary positions of the prompt, through the inference build's chunked prefill. The last position's logits.
fn served_logits(rig: &Rig, ids: &[u32], image: &imaging::pixels::Rgb8, rotary: bool) -> Vec<f32> {
    let m = Qwen35::new_on_dt(rig.gpu.share(), rig.cfg.clone(), 1, rig.cfg.block_size, &rig.init, &TierPolicy::uniform(Dtype::F32));
    let d = rig.cfg.d_model as usize;
    let embeds = prompt_embeds(&rig.tower(), ids, std::slice::from_ref(image), &|| false).expect("the prompt and its image agree");
    let mut rows = m.embed_tokens(ids);
    for (k, &p) in embeds.positions().iter().enumerate() {
        rows[p as usize * d..(p as usize + 1) * d].copy_from_slice(&embeds.rows()[k * d..(k + 1) * d]);
    }
    m.reset_decode_cache();
    let last = if rotary {
        m.prefill_embeds_rotary(&rows, &PromptRope::new(embeds.rotary_positions().expect("rotary positions").to_vec()), 5)
    } else {
        m.prefill_embeds(&rows, 5)
    };
    head(&m.read_weight(rig.cfg.head_weight()), &last)
}

fn trained_logits(m: &Qwen35, ids: &[u32], example: Option<usize>) -> Vec<f32> {
    let y = targets_of(ids);
    Model::set_batch(m, Batch::LmExamples { split: Split::Train, tokens: ids, targets: &y, examples: &[example] });
    let v = m.config().vocab as usize;
    let all = m.logits_all(ids);
    all[(ids.len() - 1) * v..].to_vec()
}

#[test]
fn the_training_forward_has_the_logits_serving_produces_for_the_same_prompt() {
    let rig = Rig::new();
    let image = picture(8, 8, 5);
    let ids = prompt();
    let m = rig.trainer(1, ids.len() as u32, &image, "parity");

    let trained = trained_logits(&m, &ids, Some(0));
    let served = served_logits(&rig, &ids, &image, true);
    let err = maxabs(&trained, &served);
    eprintln!("train vs serve, last-position logits: max abs difference {err:e}");
    assert!(err < TOLERANCE, "the training forward differs from serving by {err}");

    // The comparison can fail: the rotary positions and the image rows both move the logits well past the tolerance.
    let unrotated = maxabs(&served, &served_logits(&rig, &ids, &image, false));
    let other = maxabs(&served, &served_logits(&rig, &ids, &picture(8, 8, 6), true));
    eprintln!("sensitivity: plain positions move the logits by {unrotated:e}, another picture by {other:e}");
    assert!(unrotated > 10.0 * TOLERANCE, "plain row positions would pass the comparison ({unrotated})");
    assert!(other > 10.0 * TOLERANCE, "another picture would pass the comparison ({other})");
}

#[test]
fn a_text_batch_after_an_image_batch_carries_nothing_over() {
    let rig = Rig::new();
    let image = picture(8, 8, 5);
    let text: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let ids = prompt();
    assert_eq!(ids.len(), text.len());
    let m = rig.trainer(1, ids.len() as u32, &image, "carry");

    let alone = trained_logits(&m, &text, None);
    let with_image = trained_logits(&m, &ids, Some(0));
    assert!(maxabs(&alone, &with_image) > 1e-3);
    let after = trained_logits(&m, &text, None);
    assert!(maxabs(&alone, &after) < 1e-6, "the splice or the rotary tables outlived the image batch");

    // A plain `Batch::Lm` clears them as well.
    trained_logits(&m, &ids, Some(0));
    let y = targets_of(&text);
    Model::set_batch(&m, Batch::Lm { tokens: &text, targets: &y });
    let v = m.config().vocab as usize;
    let all = m.logits_all(&text);
    assert!(maxabs(&alone, &all[(text.len() - 1) * v..]) < 1e-6);
}

#[test]
fn each_row_of_a_wider_batch_trains_as_it_would_alone() {
    let rig = Rig::new();
    let image = picture(8, 8, 5);
    let (image_row, text_row) = (prompt(), vec![9u32, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4]);
    let len = image_row.len();
    let (y_image, mut y_text) = (targets_of(&image_row), targets_of(&text_row));
    // The text row stops being supervised early, so the batch computes only the image row's full length.
    for t in y_text.iter_mut().skip(6) {
        *t = model::IGNORE;
    }

    let loss_alone = |m: &Qwen35, ids: &[u32], y: &[u32], example: Option<usize>| {
        Model::set_batch(m, Batch::LmExamples { split: Split::Train, tokens: ids, targets: y, examples: &[example] });
        (m.forward(), y.iter().filter(|&&t| t != model::IGNORE).count() as f32)
    };
    let one = rig.trainer(1, len as u32, &image, "wide-1");
    let (l_image, c_image) = loss_alone(&one, &image_row, &y_image, Some(0));
    let (l_text, c_text) = loss_alone(&one, &text_row, &y_text, None);

    let two = rig.trainer(2, len as u32, &image, "wide-2");
    let (tokens, targets) = ([image_row.clone(), text_row.clone()].concat(), [y_image, y_text].concat());
    Model::set_batch(&two, Batch::LmExamples { split: Split::Train, tokens: &tokens, targets: &targets, examples: &[Some(0), None] });
    let together = two.forward();
    let want = (l_image * c_image + l_text * c_text) / (c_image + c_text);
    assert!((together - want).abs() < 1e-4 * want.abs().max(1.0), "batch loss {together}, the rows alone give {want}");
}

/// An agent resends the same picture every turn: the tower reads it once, the next turn's prompt gets the rows it
/// would have computed (the same key for the decoder's prefix cache), and another picture is read by the tower.
#[test]
fn a_picture_seen_before_does_not_go_through_the_tower_again() {
    let rig = Rig::new();
    let tower = rig.tower();
    let seen = EncodedImages::new(2);
    let ids = prompt();
    let (cat, dog) = (picture(8, 8, 21), picture(8, 8, 22));
    let embeds = |image: &imaging::pixels::Rgb8| prompt_embeds_remembering(&tower, &seen, &ids, std::slice::from_ref(image), &|| false).unwrap();

    let first = embeds(&cat);
    assert_eq!(seen.encodes(), 1);
    let again = embeds(&cat);
    assert_eq!(seen.encodes(), 1, "the same picture is not encoded again");
    assert_eq!(first, again);
    assert_eq!(first, prompt_embeds(&tower, &ids, std::slice::from_ref(&cat), &|| false).unwrap(), "remembered rows are the tower's rows");

    let other = embeds(&dog);
    assert_eq!(seen.encodes(), 2, "another picture goes through the tower");
    assert_ne!(first.content_key().digest_at(ids.len()), other.content_key().digest_at(ids.len()));

    // Capacity 2: the cat was used before the dog, a third picture pushes out the least recently used (the cat).
    embeds(&picture(8, 8, 23));
    assert_eq!(seen.encodes(), 3);
    embeds(&dog);
    assert_eq!(seen.encodes(), 3, "the dog is still remembered");
    embeds(&cat);
    assert_eq!(seen.encodes(), 4, "the cat was pushed out");
}

#[test]
fn the_tower_reads_the_image_serving_reads_from_the_same_bytes() {
    // Serving hands the tower the image through the request's float blob; training decodes the same file directly.
    let image = picture(8, 8, 9);
    let file = ppm(&image);
    let from_file = imaging::codec::decode_bounded(&file, capability::blob::MAX_IMAGE_PIXELS).unwrap();
    let blob = capability::blob::image_blob(&from_file.to_hwc_unit(), from_file.w, from_file.h, 3);
    let served = qwen35::vision_frontend::request_images(&capability::Invocation::default().blob("image", blob)).unwrap();
    assert_eq!(served.len(), 1);
    assert_eq!(served[0], from_file, "serving and training decode one file to different pixels");
    assert_eq!(from_file, image);
}

#[test]
fn a_record_the_tower_cannot_read_is_refused_naming_why() {
    let reader = VisionReader::new(&tiny_spec());
    let url = |bytes: &[u8]| format!("data:image/x-ppm;base64,{}", data::base64::encode(bytes));

    let (bytes, text) = reader.image(&url(&ppm(&picture(8, 8, 1)))).expect("a readable picture");
    assert_eq!(&bytes[..2], b"P6");
    assert_eq!(text, qwen35::vision_frontend::placeholder_run(ROWS as u32), "one image is one run of the geometry's rows");

    for (what, input, why) in [
        ("an external URL", "https://example.com/a.png".to_string(), "inline 'data:"),
        ("bytes that are not an image", url(b"not an image"), "cannot be decoded"),
        ("a header declaring more pixels than are read", url(b"P6\n40000 40000\n255\n"), "cannot be decoded"),
    ] {
        let err = reader.image(&input).map(|_| ()).expect_err(what);
        assert!(err.contains(why), "{what}: {err}");
    }
    assert!(reader.check_text("look at <|image_pad|>").unwrap_err().contains("placeholder"));
    assert!(reader.check_record(&vec![bytes.clone(); 9]).unwrap_err().contains("at most"));
}

#[test]
fn a_dataset_with_images_needs_a_base_with_a_vision_tower() {
    let rig = Rig::new();
    let dir = scratch("no-tower-data");
    chat_media::write(&dir, "train", &[vec![ppm(&picture(8, 8, 1)).into()]]).unwrap();
    let m = Qwen35::new_train_on(rig.gpu.share(), rig.cfg.clone(), 1, 12, &rig.init);

    let text_base = scratch("no-tower-base");
    std::fs::write(text_base.join("config.json"), "{}").unwrap();
    let err = attach_for_dataset(&m, &text_base, &dir).unwrap_err().to_string();
    assert!(err.contains("holds images") && err.contains("no vision tower"), "{err}");

    // A text dataset needs nothing of its base.
    let text_only = scratch("no-tower-text");
    attach_for_dataset(&m, &text_base, &text_only).expect("no images, nothing to attach");
}

#[test]
fn a_prepared_record_whose_text_and_images_disagree_is_refused_naming_the_example() {
    let spec = tiny_spec();
    let dir = scratch("disagree");
    let ids = prompt();
    data::chat::write_split(&dir, "train", &{
        let mut split = data::chat::EncodedSplit::default();
        split.push_with_media(&ids, &vec![true; ids.len()], &[ppm(&picture(8, 8, 1)).into()]);
        // The second example places a run of two rows for a picture that needs four.
        let short: Vec<u32> = vec![1, TOKENS.start, TOKENS.image, TOKENS.image, TOKENS.end, 2];
        split.push_with_media(&short, &vec![true; short.len()], &[ppm(&picture(8, 8, 2)).into()]);
        split
    })
    .unwrap();
    data::chat::write_split(&dir, "val", &data::chat::EncodedSplit::default()).unwrap();
    let err = check_prepared(&dir, &spec).unwrap_err();
    assert!(err.contains("train example 1") && err.contains("[2]") && err.contains("[4]"), "{err}");
}

#[test]
fn a_vision_release_is_read_and_trains_through_the_dataset_entry_point() {
    let rig = Rig::new();
    let release = scratch("release");
    add_vision_release(&release, 3);
    let image = picture(8, 8, 5);
    let ids = prompt();
    let dir = scratch("e2e");
    let mut split = data::chat::EncodedSplit::default();
    split.push_with_media(&ids, &targets_of(&ids).iter().map(|&t| t != model::IGNORE).collect::<Vec<_>>(), &[ppm(&image).into()]);
    for name in ["train", "val"] {
        data::chat::write_split(&dir, name, &split).unwrap();
    }
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(rig.cfg.vocab as usize)).unwrap();

    let opts = model::FitOpts { batch_size: 1, block_size: ids.len() as u32, steps: 10, lr: 5e-2, min_lr: 5e-2, warmup: 0, grad_clip: 1.0, eval_interval: 0, ..model::FitOpts::default() };
    let lora = Qwen35Config { lora: Some(qwen35::config::lora_cfg(2, 4.0)), ..rig.cfg.clone() };
    let init = qwen35::init::init_weights(&lora, 11);
    let m = Qwen35::new_train_on(rig.gpu.share(), lora, 1, ids.len() as u32, &init);
    attach_for_dataset(&m, &release, &dir).expect("the tower of the release reads the dataset");
    let (train, val, cfg, _, itos) = model::load_dataset_with_images(&dir, &opts).unwrap();
    let obj = model::causal_lm_examples::<Qwen35>(train, val, cfg, itos);
    let (report, _) = model::fit_controlled(m, obj, &opts, None, model::FitControl::default()).unwrap();
    let last = report.final_loss.expect("a final loss");
    assert!(report.initial_loss.is_finite() && last.is_finite());
    assert!(last < 0.9 * report.initial_loss, "ten steps on one record should lower its loss by a tenth: {} -> {last}", report.initial_loss);

    // The text trainers refuse the same directory instead of dropping its images.
    let err = model::load_dataset_with_itos(&dir, &opts).map(|_| ()).unwrap_err().to_string();
    assert!(err.contains("holds images") && err.contains("text only"), "{err}");
}
