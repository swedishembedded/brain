// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Multimodal content (ATIF v1.6+): the `message` field on a step and the
//! `content` field on an observation result can each be either plain text or
//! an array of mixed text/image segments.

use serde::{Deserialize, Serialize};

/// Either a plain string or a sequence of [`ContentSegment`]s.
///
/// Used for `StepObject.message` (required, may be an empty string) and for
/// `ObservationResultSchema.content` (optional). Untagged so a bare JSON
/// string and a JSON array both deserialize without an explicit `type` tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageBody {
    /// Plain text (the common case; can be `""`).
    Text(String),
    /// Multimodal content: an ordered mix of text and image segments.
    Segments(Vec<ContentSegment>),
}

impl MessageBody {
    /// Construct a plain-text body.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Construct a multimodal body from segments.
    pub fn segments(parts: impl Into<Vec<ContentSegment>>) -> Self {
        Self::Segments(parts.into())
    }

    /// Borrow the plain-text form, if this body is [`MessageBody::Text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            Self::Segments(_) => None,
        }
    }

    /// Borrow the segment list, if this body is [`MessageBody::Segments`].
    pub fn as_segments(&self) -> Option<&[ContentSegment]> {
        match self {
            Self::Text(_) => None,
            Self::Segments(parts) => Some(parts.as_slice()),
        }
    }
}

impl From<&str> for MessageBody {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for MessageBody {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

/// One element of a multimodal [`MessageBody::Segments`] array.
///
/// Tagged on `type` so the spec's conditional-field rule ("`text` required
/// iff type is `text`, forbidden iff `image`; `source` required iff `image`,
/// forbidden iff `text`") is enforced structurally: there is no Rust value
/// of this type that can hold both `text` and `source`, or neither.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentSegment {
    /// A text run.
    Text {
        /// The text content.
        text: String,
    },
    /// A reference to an image stored alongside the trajectory file.
    Image {
        /// Where the image lives and what kind of image it is.
        source: ImageRef,
    },
}

impl ContentSegment {
    /// Construct a text segment.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Construct an image segment.
    pub fn image(media_type: ImageMediaType, path: impl Into<String>) -> Self {
        Self::Image {
            source: ImageRef::new(media_type, path),
        }
    }
}

/// `ImageSourceSchema` (ATIF v1.6+): where an image lives and what kind it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRef {
    /// MIME type of the image. Closed to the four values the spec allows -
    /// modeling this as an enum means an out-of-spec media type can never be
    /// constructed or successfully deserialized in the first place, which is
    /// this crate's chosen enforcement point for that rule (see the crate's
    /// validation tests for a deserialization-rejection test).
    pub media_type: ImageMediaType,
    /// Relative/absolute file path or URL to the image.
    pub path: String,
}

impl ImageRef {
    /// Construct a new image reference.
    pub fn new(media_type: ImageMediaType, path: impl Into<String>) -> Self {
        Self {
            media_type,
            path: path.into(),
        }
    }
}

/// The four MIME types ATIF v1.7 allows for `ImageSourceSchema.media_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageMediaType {
    /// `image/jpeg`
    #[serde(rename = "image/jpeg")]
    Jpeg,
    /// `image/png`
    #[serde(rename = "image/png")]
    Png,
    /// `image/gif`
    #[serde(rename = "image/gif")]
    Gif,
    /// `image/webp`
    #[serde(rename = "image/webp")]
    Webp,
}
