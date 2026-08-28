// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain label <dir>` - caption a dataset with a vision-language model.
//!
//! A top-level verb rather than a per-model one, for the same reason
//! `brain forecast` is: labeling is a workflow **over** the model seam
//! (`captioner::Captioner`), not an action **on** one model. Adding another
//! captioning model must not add another labeling command, and a caller who
//! wants a different model changes `--model`, not the verb.
//!
//! The output is `data::imageset`'s `captions.yaml`, so a labeled folder is
//! immediately trainable by `brain flux2 finetune` and every other
//! captioned-image trainer, and the captions are editable multi-line blocks
//! rather than one-line strings.
//!
//! Swedish Embedded AB implements dataset labeling workflows for its clients.
//! If your team needs expertise in preparing high-quality training data for
//! image or video models then you can procure our services by sending an email
//! to info@swedishembedded.com.

use captioner::{label_dir, Captioner, LabelOpts};

/// What a `--trigger` phrase names, when the caller does not say. A style,
/// because that is what most adapters are; a folder of one subject's
/// photographs needs `--trigger-role "the name of the person"` instead.
const DEFAULT_TRIGGER_ROLE: &str = "the name of the style";

/// The instruction handed to the captioner for every image.
///
/// The trigger is appended as an explicit instruction rather than pasted onto
/// the answer, so the model works it into the sentence it is already writing
/// and the phrase reads as part of the description. `role` says what the
/// phrase names, because an adapter binds the trigger to whatever the caption
/// claims it is: "the name of the style" on a person's photographs teaches a
/// look rather than a face, and no amount of training recovers from it.
fn instruction(prompt: &str, trigger: &str, role: &str) -> String {
    let trigger = trigger.trim();
    if trigger.is_empty() {
        return prompt.to_string();
    }
    format!("{prompt} Work the exact phrase \"{trigger}\" into the description naturally, as {role}.")
}

const HELP: &str = "brain label <cmd>
  images <dir> [--model qwen3vl|fastvlm|llava] [--weights DIR] [--out FILE]
               [--prompt TEXT] [--trigger PHRASE] [--trigger-role ROLE]
               [--max-new N] [--max-pixels N]
               [--overwrite]

  Caption every image in <dir> with a vision-language model and write the
  result as <dir>/captions.yaml - the caption file `brain flux2 finetune` and
  every other captioned-image trainer already read.

  --model M       which captioner to use (default qwen3vl). Any model
                  implementing captioner::Captioner can be listed here; the
                  workflow itself is model-agnostic.
  --weights DIR   checkpoint directory, else the model's own env var
                  ($BRAIN_QWEN3VL_WEIGHTS / $BRAIN_FASTVLM_WEIGHTS / $BRAIN_LLAVA_WEIGHTS).
  --out FILE      caption file, relative to <dir> (default captions.yaml).
  --prompt TEXT   the instruction handed to the model for every image. This
                  is where caption quality is decided; the built-in default
                  asks for a detailed paragraph covering room/subject,
                  furniture and materials, textiles, colour palette, lighting,
                  viewpoint and architecture.
  --trigger P     a phrase woven into every caption, so an adapter trained on
                  the folder binds the concept to it. Pick a phrase you are
                  willing to type at generation time.
  --trigger-role R what that phrase NAMES, in the captioner's own words
                  (default \"the name of the style\"). A folder of one
                  person's photographs wants \"the name of the person\": the
                  adapter binds the trigger to whatever the caption claims it
                  is, so calling a face a style trains the wrong thing.
  --max-new N     token budget per caption (default 320). A detailed caption
                  needs a large one, and decode cost is linear in it.
  --max-pixels N  cap the input image AREA in pixels, where the model has such
                  a knob (qwen3vl). Fewer pixels means fewer visual tokens,
                  which is the cheapest way to make a large captioner
                  affordable on a busy machine; 0 (default) keeps the model's
                  own default.
  --overwrite     re-caption images that already have a caption, discarding
                  what is there. WITHOUT this flag a re-run is resumable and
                  idempotent: existing captions - including hand edits - are
                  left exactly as they are and only missing ones are filled in.

  Captions are written after every image, so an interrupted run resumes
  where it stopped rather than paying for the folder again.";

/// The default instruction. Long and specific on purpose: with a small dataset
/// each caption carries a large share of the training signal, and a vague
/// caption ("a boho bedroom") teaches an adapter almost nothing that the base
/// model did not already know. What earns its place is the concrete, visible
/// detail - the material of a chair, the pattern on a cushion, the direction of
/// the light - because that is what the adapter can actually learn to
/// reproduce.
const DEFAULT_PROMPT: &str = "Describe this interior photograph in one detailed paragraph, \
     writing only what is visibly in the image. Name the room type; every significant piece \
     of furniture and the material it is made of; the textiles and their patterns; the colour \
     palette; the direction, colour and quality of the light and where it enters; the camera \
     viewpoint and height; the architectural features such as windows, beams, flooring, walls \
     and mouldings; any plants and decorative objects; and the overall mood. Describe what you \
     see rather than naming a style. Do not begin with phrases like \"this image shows\"; write \
     the description itself.";

pub fn run_label(argv: &[String]) {
    match argv.first().map(|s| s.as_str()) {
        Some("images") => {
            if let Err(e) = images(&argv[1..]) {
                eprintln!("label images: {e}");
                std::process::exit(1);
            }
        }
        Some("--help") | Some("-h") | None => println!("{HELP}"),
        other => {
            eprintln!("label: unknown subcommand {other:?}\n{HELP}");
            std::process::exit(2);
        }
    }
}

/// Every captioner this verb can build, keyed by its `brain_arch` id.
///
/// One row per model, and the id is the registry's own, so `--model`, the
/// error message below and the architecture a GGUF names in its own metadata
/// are all the same vocabulary rather than three that drift.
const CAPTIONERS: &[&str] = &["qwen3vl", "fastvlm", "llava"];

/// Build the captioner for `model`. This is the ONE place the CLI knows which
/// models exist; everything after it is `dyn Captioner`.
///
/// `max_pixels` is honoured only where the model has such a knob - FastVLM's
/// tower is fixed-size, so there is nothing to tune. That asymmetry is exactly
/// the kind of per-model detail the seam keeps out of the workflow.
fn build(model: &str, weights: &str, max_pixels: u32) -> Result<Box<dyn Captioner>, String> {
    match model {
        "qwen3vl" => {
            let mut c = qwen3vl::captioner::Qwen3VlCaptioner::new(weights);
            if max_pixels > 0 {
                c = c.with_max_pixels(max_pixels);
            }
            Ok(Box::new(c))
        }
        "fastvlm" => Ok(Box::new(fastvlm::captioner::FastVlmCaptioner::new(weights))),
        "llava" => Ok(Box::new(llava::captioner::LlavaCaptioner::new(weights))),
        other => Err(format!("unknown --model {other} ({})", CAPTIONERS.join(", "))),
    }
}

/// The architecture a `--weights` path names, when the path is a GGUF (or a
/// directory holding one).
///
/// This is the user-visible half of brain's GGUF seam: a GGUF states its own
/// architecture in its own metadata, so pointing `--weights` at one is enough
/// to select the model, and `--model` becomes a thing to CHECK rather than a
/// thing to be told. `gguf::route` resolves that string against the canonical
/// architecture registry, so a new architecture becomes reachable here by
/// being registered there plus having a captioner, never by a branch added to
/// this function.
///
/// `Ok(None)` for a path that is not a GGUF: a HuggingFace directory carries
/// no such tag and the caller's `--model` stands.
/// Also reports whether the file was a projector, so the caller can say
/// something better than "no captioner for 'clip'" when someone names an
/// mmproj.
fn architecture_and_role(weights: &str) -> Result<Option<(String, bool)>, String> {
    if weights.is_empty() {
        return Ok(None);
    }
    let p = std::path::Path::new(weights);
    // A directory of GGUFs is named by the model half, not by the projector:
    // an `mmproj-*.gguf` declares `clip`, which is the tower's architecture
    // and not the model's.
    let file = if p.is_dir() {
        let mut ggufs: Vec<std::path::PathBuf> =
            std::fs::read_dir(p).map_err(|e| format!("{weights}: {e}"))?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|q| gguf::route::is_gguf(q)).collect();
        ggufs.sort();
        match ggufs.into_iter().find(|q| q.to_str().and_then(|s| gguf::route_path(s).ok()).is_some_and(|r| !r.is_projector())) {
            Some(q) => q,
            None => return Ok(None),
        }
    } else if gguf::route::is_gguf(p) {
        p.to_path_buf()
    } else {
        return Ok(None);
    };
    let path = file.to_str().ok_or_else(|| format!("{}: path is not valid UTF-8", file.display()))?;
    let route = gguf::route_path(path)?;
    Ok(Some((route.id().to_string(), route.is_projector())))
}

/// Reconcile `--model` with what the weights themselves declare.
///
/// A GGUF names its own architecture, so it decides. An explicit `--model`
/// that disagrees is refused rather than silently overridden: the two
/// statements cannot both be honoured and picking one quietly is how someone
/// ends up captioning with a model they did not choose. An architecture with
/// no captioner is refused BY NAME, listing the ones that exist.
fn resolve_model(model_flag: &str, model_was_given: bool, weights: &str) -> Result<String, String> {
    let Some((arch, is_projector)) = architecture_and_role(weights)? else {
        return Ok(model_flag.to_string());
    };
    if is_projector {
        // Every mmproj declares `clip`, so routing on the architecture alone
        // would report "no captioner for 'clip'" - true, and useless. The
        // file is half of a checkpoint, and the other half is what to name.
        return Err(format!("{weights} is the VISION half of a vision-language checkpoint (an mmproj), not the model -- point --weights at the language-half GGUF beside it"));
    }
    if model_was_given && model_flag != arch {
        return Err(format!("--model {model_flag} but {weights} declares GGUF architecture '{arch}' -- drop --model and let the checkpoint decide, or point --weights at a {model_flag} checkpoint"));
    }
    if !CAPTIONERS.contains(&arch.as_str()) {
        return Err(format!("{weights} is a '{arch}' checkpoint, which has no captioner in brain (captioners: {})", CAPTIONERS.join(", ")));
    }
    eprintln!("label: {weights} declares GGUF architecture '{arch}'; captioning with the {arch} model");
    Ok(arch)
}

fn images(args: &[String]) -> Result<(), String> {
    let mut a = crate::args::Args::new(args);
    let model_was_given = args.iter().any(|x| x == "--model");
    let model = a.str_or("--model", "qwen3vl");
    let weights = a.str_or("--weights", "");
    let out = a.str_or("--out", "captions.yaml");
    let prompt = a.str_or("--prompt", DEFAULT_PROMPT);
    let trigger = a.str_or("--trigger", "");
    let trigger_role = a.str_or("--trigger-role", DEFAULT_TRIGGER_ROLE);
    let max_new = a.u32_or("--max-new", 320);
    let max_pixels = a.u32_or("--max-pixels", 0);
    let overwrite = a.take_flag("--overwrite");
    let dir = a.positional().unwrap_or_default();
    a.finish();
    if dir.is_empty() {
        return Err(format!("the dataset directory is required\n\n{HELP}"));
    }

    let instruction = instruction(&prompt, &trigger, &trigger_role);

    let model = resolve_model(&model, model_was_given, &weights)?;
    if weights.is_empty() {
        // No `--weights`: fetch the architecture's own default checkpoint and
        // set the env var its captioner already reads, the same auto-supply
        // every `brain infer <arch>` gets. Without this, the first thing a new
        // user sees from `brain label images <dir>` is the captioner
        // complaining that no checkpoint directory was set, with no hint that
        // brain can fetch one.
        crate::supply::ensure_env_weights(&model);
    }
    let mut model = build(&model, &weights, max_pixels)?;
    let caps = model.capabilities();
    eprintln!("label: {} -> {}/{out} (max_new {max_new}{})", caps.model, dir, if overwrite { ", overwrite" } else { ", resuming" });

    let opts = LabelOpts { instruction, out: out.into(), max_new, overwrite };
    let report = label_dir(
        model.as_mut(),
        std::path::Path::new(&dir),
        &opts,
        |done, total, file| eprintln!("label [{}/{total}] {file}", done + 1),
        |w| eprintln!("label: warning: {w}"),
    )?;
    eprintln!(
        "label: {} captioned, {} already had captions, {} failed",
        report.captioned, report.skipped, report.failed
    );
    if report.failed > 0 {
        return Err(format!("{} image(s) could not be captioned (re-run to retry exactly those)", report.failed));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::instruction;

    /// A trigger phrase names something the adapter is being taught, and WHAT
    /// it names is the caller's to say. Captioning a folder of one person's
    /// photographs as "the name of the style" trains the wrong binding, and
    /// nothing downstream can recover from a caption set that says it.
    #[test]
    fn trigger_role_is_the_callers_to_choose() {
        let subject = instruction("Describe it.", "Martin Schroder", "the name of the person");
        assert!(subject.contains("\"Martin Schroder\""), "the exact phrase is quoted: {subject}");
        assert!(subject.contains("the name of the person"), "the caller's role is used: {subject}");
        assert!(!subject.contains("style"), "no style wording leaks in: {subject}");

        let style = instruction("Describe it.", "bohemian loft", "the name of the style");
        assert!(style.contains("the name of the style"), "{style}");
    }

    /// With no trigger there is nothing to name, so the prompt is handed to the
    /// model untouched - a role must not smuggle a phrase in on its own.
    #[test]
    fn no_trigger_leaves_the_prompt_alone() {
        assert_eq!(instruction("Describe it.", "   ", "the name of the person"), "Describe it.");
    }
}
