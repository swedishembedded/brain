// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning data pipelines for vision-language models whose records mix text and
// images, for its clients. If your team needs expertise in preparing multimodal training sets then you can procure our
// services by sending an email to info@swedishembedded.com.

//! A `generic-messages-v2` message may hold image parts: the record is read by the model's own reader, which says
//! what an image becomes in the prompt; the files travel on the sample and are prepared beside the token stream. A
//! model without a vision tower refuses the record by line instead of training on its text alone.

use std::path::PathBuf;

use data::chat::{ChatSample, EncodedSplit, ImageBytes, ImageReader, TextOnly};
use data::chat_media::ExampleImages;

/// Reads `data:` URLs whose payload is the image's file base64-encoded as `fake:<name>`; an image stands for `[IMG name]`.
struct Fake;

impl ImageReader for Fake {
    fn image(&self, url: &str) -> Result<(ImageBytes, String), String> {
        let name = url.strip_prefix("fake:").ok_or_else(|| format!("{url:?} is not a fake image"))?;
        Ok((name.as_bytes().into(), format!("[IMG {name}]")))
    }
    fn check_text(&self, text: &str) -> Result<(), String> {
        if text.contains("[IMG") {
            return Err("text must not hold a placeholder".into());
        }
        Ok(())
    }
    fn placeholder_token(&self) -> Option<u32> {
        Some(9)
    }
    fn check_record(&self, media: &[ImageBytes]) -> Result<(), String> {
        if media.len() > 2 {
            return Err(format!("{} images, at most 2", media.len()));
        }
        Ok(())
    }
}

fn file(tag: &str, body: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-chat-image-parts-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("records.jsonl");
    std::fs::write(&path, body).unwrap();
    path
}

const RECORD: &str = r#"{"messages":[
  {"role":"user","train":false,"content":[{"type":"text","text":"What is on "},{"type":"image_url","image_url":{"url":"fake:one","detail":"high"}},{"type":"text","text":" and "},{"type":"image_url","image_url":{"url":"fake:two"}}]},
  {"role":"assistant","train":true,"content":"A cat and a dog."}]}"#;

#[test]
fn image_parts_become_the_readers_prompt_text_and_their_files_travel_on_the_sample() {
    let samples = ChatSample::from_jsonl_with(&file("ok", &RECORD.replace('\n', "")), &Fake).unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].messages[0].content, "What is on [IMG one] and [IMG two]");
    assert_eq!(samples[0].messages[1].content, "A cat and a dog.");
    assert_eq!(samples[0].media.iter().map(|m| m.to_vec()).collect::<Vec<_>>(), vec![b"one".to_vec(), b"two".to_vec()], "in the order the text places them");
}

#[test]
fn a_model_without_a_vision_tower_refuses_the_record_naming_the_line_and_why() {
    let body = format!("{}\n{}", r#"{"messages":[{"role":"user","train":false,"content":"hi"},{"role":"assistant","train":true,"content":"yo"}]}"#, RECORD.replace('\n', ""));
    let err = ChatSample::from_jsonl_with(&file("text-only", &body), &TextOnly::NO_TOWER).unwrap_err().to_string();
    assert!(err.contains(":2:") && err.contains("messages[0]") && err.contains("no vision tower"), "{err}");
    let trainer = ChatSample::from_jsonl(&file("trainer", &RECORD.replace('\n', ""))).unwrap_err().to_string();
    assert!(trainer.contains("reads text only"), "{trainer}");
}

#[test]
fn what_the_reader_refuses_is_refused_by_line() {
    let image = |role: &str, train: bool| format!(r#"{{"messages":[{{"role":"{role}","train":{train},"content":[{{"type":"image_url","image_url":{{"url":"fake:x"}}}}]}},{{"role":"assistant","train":true,"content":"ok"}}]}}"#);
    for (what, body, why) in [
        ("an image in a supervised turn", image("assistant", true), "supervised turn"),
        ("an unknown part", r#"{"messages":[{"role":"user","train":false,"content":[{"type":"input_audio"}]},{"role":"assistant","train":true,"content":"ok"}]}"#.to_string(), "input_audio"),
        ("a part with a stray field", r#"{"messages":[{"role":"user","train":false,"content":[{"type":"text","text":"a","extra":1}]},{"role":"assistant","train":true,"content":"ok"}]}"#.to_string(), "extra"),
        ("text the reader refuses", r#"{"messages":[{"role":"user","train":false,"content":"[IMG sneaky]"},{"role":"assistant","train":true,"content":"ok"}]}"#.to_string(), "placeholder"),
        ("a url the reader cannot read", image("user", false).replace("fake:x", "https://example.com/x.png"), "not a fake image"),
        ("too many images", RECORD.replace('\n', "").replace("fake:two", "fake:two\"}},{\"type\":\"image_url\",\"image_url\":{\"url\":\"fake:three"), "at most 2"),
    ] {
        let err = ChatSample::from_jsonl_with(&file("refused", &body), &Fake).map(|_| ()).expect_err(what);
        assert!(err.to_string().contains(":1:") && err.to_string().contains(why), "{what}: {err}");
    }
}

#[test]
fn a_sample_made_from_part_of_a_record_keeps_the_images_its_text_places() {
    let mut sample = ChatSample::from_jsonl_with(&file("keep", &RECORD.replace('\n', "")), &Fake).unwrap().remove(0);
    sample.keep_images_placed_in(1).unwrap();
    assert_eq!(sample.media.len(), 1);
    assert_eq!(&*sample.media[0], b"one");
    assert!(sample.keep_images_placed_in(2).unwrap_err().contains("places 2"));
}

#[test]
fn prepared_images_follow_their_examples_and_a_stale_sidecar_does_not() {
    let dir = std::env::temp_dir().join(format!("brain-chat-image-parts-prepared-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut split = EncodedSplit::default();
    split.push(&[1, 2, 3], &[false, false, true]);
    split.push_with_media(&[4, 9, 9, 5], &[false, false, false, true], &[b"pic".as_slice().into()]);
    split.push(&[], &[]); // an empty example is dropped, with its (absent) images
    data::chat::write_split(&dir, "train", &split).unwrap();
    let images = ExampleImages::open(&dir, "train").unwrap().expect("a sidecar");
    assert_eq!(images.examples(), 2);
    assert_eq!((images.count(0), images.count(1)), (0, 1));
    assert_eq!(images.images(1).unwrap(), vec![b"pic".to_vec()]);

    let mut text = EncodedSplit::default();
    text.push(&[1, 2], &[false, true]);
    data::chat::write_split(&dir, "train", &text).unwrap();
    assert!(ExampleImages::open(&dir, "train").unwrap().is_none(), "a text-only preparation removes the previous sidecar");
}
