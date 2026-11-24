// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a text-to-speech action takes, declared by the model that owns it, so
//! a generic speech request (the words, a reference voice and its transcript,
//! a style instruction, a preset voice, a seed) maps onto the action's own
//! params and inputs without guessing from their names.
//!
//! Speech models share no vocabulary: one clones from a `ref` WAV with an
//! optional `ref_text`, another from a `ref_audio` clip whose transcript it
//! requires, a third takes a free-text `instruct`. A [`SpeechContract`] states
//! those facts once; [`SpeechContract::fits`] says whether a request can be
//! served by the action, [`SpeechContract::unused`] ranks the actions that
//! can, and [`SpeechContract::listing`] is what a model card shows. A request
//! the contract cannot honour is refused with the field named, never served by
//! dropping the field.
//!
//! Swedish Embedded AB implements solutions for serving speech synthesis and
//! voice cloning behind one stable API for its clients. If your team needs
//! expertise in text-to-speech serving then you can procure our services by
//! sending an email to info@swedishembedded.com.

use serde_json::{json, Value};

/// How a reference recording's transcript enters the action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transcript {
    /// The action reads no transcript.
    None,
    /// A param the transcript may fill; without one the action clones from the
    /// recording's timbre alone.
    Optional(String),
    /// A param the transcript must fill.
    Required(String),
}

impl Transcript {
    /// The param a transcript fills, if the action reads one.
    pub fn param(&self) -> Option<&str> {
        match self {
            Transcript::None => None,
            Transcript::Optional(p) | Transcript::Required(p) => Some(p),
        }
    }
}

/// The voice an action can be given as a recording to imitate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    /// The input blob the recording fills, as a WAV container at its own rate.
    pub input: String,
    /// Where the recording's transcript goes.
    pub transcript: Transcript,
}

/// The facts a speech request is mapped by. Every name is one of the action's
/// own params or inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechContract {
    /// The param the words to speak fill.
    pub text: String,
    /// A recording the action imitates; `None` for an action that speaks in a
    /// voice of its own.
    pub reference: Option<Reference>,
    /// A param taking a free-text description of the voice and delivery
    /// (character, emotion, pace).
    pub instructions: Option<String>,
    /// A param naming one of the model's preset voices.
    pub voice: Option<String>,
    /// The param bounding the clip's length, in the model's own units.
    pub length: Option<String>,
    /// The param the request's seed fills.
    pub seed: Option<String>,
    /// The param naming the language the words are spoken in, as the model
    /// names its languages.
    pub language: Option<String>,
}

/// What a speech request asks for, as far as the choice of action goes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeechRequest {
    pub reference: bool,
    pub transcript: bool,
    pub instructions: bool,
    pub voice: bool,
}

impl SpeechContract {
    /// An action that speaks `text` in a voice of its own.
    pub fn new(text: &str) -> SpeechContract {
        SpeechContract { text: text.into(), reference: None, instructions: None, voice: None, length: None, seed: None, language: None }
    }

    /// The action imitates the recording in `input`, its transcript going as
    /// `transcript` says.
    pub fn reference(mut self, input: &str, transcript: Transcript) -> SpeechContract {
        self.reference = Some(Reference { input: input.into(), transcript });
        self
    }

    pub fn instructions(mut self, param: &str) -> SpeechContract {
        self.instructions = Some(param.into());
        self
    }

    pub fn voice(mut self, param: &str) -> SpeechContract {
        self.voice = Some(param.into());
        self
    }

    pub fn length(mut self, param: &str) -> SpeechContract {
        self.length = Some(param.into());
        self
    }

    pub fn seed(mut self, param: &str) -> SpeechContract {
        self.seed = Some(param.into());
        self
    }
    pub fn language(mut self, param: &str) -> SpeechContract {
        self.language = Some(param.into());
        self
    }

    /// Whether this action can serve `request`: everything it carries has a
    /// place here, and nothing the action requires is missing. A request with
    /// a reference goes only to an action that imitates one, and one without
    /// only to an action that does not.
    pub fn fits(&self, request: SpeechRequest) -> bool {
        let reference_fits = match (&self.reference, request.reference) {
            (None, false) => !request.transcript,
            (None, true) => false,
            (Some(_), false) => false,
            (Some(r), true) => match &r.transcript {
                Transcript::None => !request.transcript,
                Transcript::Optional(_) => true,
                Transcript::Required(_) => request.transcript,
            },
        };
        reference_fits && (!request.instructions || self.instructions.is_some()) && (!request.voice || self.voice.is_some())
    }

    /// How many of this action's ways of steering the voice `request` leaves
    /// unused: of the actions that fit, the one with the fewest is the
    /// narrowest that serves it (a plain request goes to plain synthesis, not
    /// to the action that also takes an instruction).
    pub fn unused(&self, request: SpeechRequest) -> usize {
        [self.instructions.is_some() && !request.instructions, self.voice.is_some() && !request.voice].into_iter().filter(|&u| u).count()
    }

    /// What a model card says the action takes, in the request's own terms.
    pub fn listing(&self) -> Value {
        let transcript = match self.reference.as_ref().map(|r| &r.transcript) {
            None | Some(Transcript::None) => "none",
            Some(Transcript::Optional(_)) => "optional",
            Some(Transcript::Required(_)) => "required",
        };
        json!({
            "reference_voice": self.reference.is_some(),
            "reference_transcript": transcript,
            "instructions": self.instructions.is_some(),
            "preset_voices": self.voice.is_some(),
            "seed": self.seed.is_some(),
            "language": self.language.is_some(),
        })
    }

    /// The names this contract gives that `spec` does not declare: a param it
    /// fills that is not among the action's params, or a recording input that is
    /// not among its inputs. Empty for a contract that only names what exists.
    pub fn undeclared(&self, spec: &crate::ActionSpec) -> Vec<String> {
        let param = |name: &str| spec.params.iter().any(|p| p.name == name);
        let mut missing: Vec<String> = [Some(self.text.as_str()), self.instructions.as_deref(), self.voice.as_deref(), self.length.as_deref(), self.seed.as_deref(), self.language.as_deref(), self.reference.as_ref().and_then(|r| r.transcript.param())]
            .into_iter()
            .flatten()
            .filter(|name| !param(name))
            .map(|name| format!("param '{name}'"))
            .collect();
        if let Some(r) = &self.reference {
            if !spec.inputs.iter().any(|b| b.name == r.input) {
                missing.push(format!("input '{}'", r.input));
            }
        }
        missing
    }

    /// The contract as the manifest's JSON carries it.
    pub fn to_json(&self) -> Value {
        json!({
            "text": self.text,
            "reference": self.reference.as_ref().map(|r| json!({"input": r.input, "transcript": r.transcript.param(), "transcript_required": matches!(r.transcript, Transcript::Required(_))})),
            "instructions": self.instructions,
            "voice": self.voice,
            "length": self.length,
            "seed": self.seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain() -> SpeechContract {
        SpeechContract::new("text").seed("seed")
    }
    fn clone(t: Transcript) -> SpeechContract {
        SpeechContract::new("text").reference("ref", t)
    }
    fn design() -> SpeechContract {
        SpeechContract::new("text").instructions("instruct").voice("speaker")
    }
    fn req(reference: bool, transcript: bool, instructions: bool, voice: bool) -> SpeechRequest {
        SpeechRequest { reference, transcript, instructions, voice }
    }

    #[test]
    fn a_request_fits_only_an_action_with_a_place_for_everything_it_carries() {
        assert!(plain().fits(req(false, false, false, false)));
        assert!(!plain().fits(req(true, false, false, false)), "a reference must not be dropped");
        assert!(!plain().fits(req(false, false, true, false)), "an instruction must not be dropped");
        assert!(!plain().fits(req(false, false, false, true)), "a voice name must not be dropped");
        assert!(!plain().fits(req(false, true, false, false)), "a transcript without a recording has no meaning");
        assert!(!clone(Transcript::Optional("ref_text".into())).fits(req(false, false, false, false)), "a cloning action needs its recording");
        assert!(clone(Transcript::Optional("ref_text".into())).fits(req(true, false, false, false)));
        assert!(clone(Transcript::Optional("ref_text".into())).fits(req(true, true, false, false)));
        assert!(!clone(Transcript::Required("ref_text".into())).fits(req(true, false, false, false)), "a required transcript is missing");
        assert!(!clone(Transcript::None).fits(req(true, true, false, false)), "a transcript the action cannot read");
        assert!(design().fits(req(false, false, true, false)));
        assert!(design().fits(req(false, false, false, true)));
    }

    #[test]
    fn the_narrowest_fitting_action_ranks_first() {
        let r = req(false, false, false, false);
        assert!(plain().unused(r) < design().unused(r));
        assert_eq!(design().unused(req(false, false, true, true)), 0);
    }

    #[test]
    fn a_contract_naming_what_its_action_lacks_is_caught() {
        use crate::{ActionSpec, BlobSpec, Media, ParamSpec, ParamType};
        let spec = ActionSpec::new("clone", "c").param(ParamSpec::new("text", ParamType::Str, "t")).input(BlobSpec::new("ref", Media::Bytes, "r"));
        assert!(clone(Transcript::None).undeclared(&spec).is_empty());
        assert_eq!(clone(Transcript::Optional("ref_text".into())).undeclared(&spec), vec!["param 'ref_text'".to_string()]);
        assert_eq!(SpeechContract::new("text").reference("voice", Transcript::None).undeclared(&spec), vec!["input 'voice'".to_string()]);
    }

    #[test]
    fn the_listing_says_what_a_caller_may_send() {
        let l = clone(Transcript::Required("ref_text".into())).listing();
        assert_eq!(l["reference_voice"], true);
        assert_eq!(l["reference_transcript"], "required");
        assert_eq!(design().listing()["instructions"], true);
        assert_eq!(plain().listing()["reference_transcript"], "none");
    }
}
