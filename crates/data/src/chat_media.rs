// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language models on a client's own annotated images. If your
// team needs expertise in training pipelines that carry images beside the token stream then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The images of a prepared chat dataset: a sidecar of [`crate::chat::write_split`]'s token files.
//!
//! A conversation whose user turns hold images is prepared like any other (the images stand in the token stream as
//! the placeholder rows their model reads), and its files are kept beside the stream, per split:
//!
//! * `<split>.img.bin` - the image files as supplied, one after another;
//! * `<split>.img.off.bin` - `u64` byte offset of every image in it, and the total length last;
//! * `<split>.img.ex.bin` - `u64` index of the first image of every example, and the image count last.
//!
//! Example `i` of the split (the `i`th entry of `<split>.ex.bin`) holds images `ex[i]..ex[i + 1]`, in the order its
//! text places them. A split with no images has no sidecar, and preparing one removes a stale sidecar.

use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::binio;
use crate::chat::ImageBytes;

fn names(dir: &Path, split: &str) -> (PathBuf, PathBuf, PathBuf) {
    (dir.join(format!("{split}.img.bin")), dir.join(format!("{split}.img.off.bin")), dir.join(format!("{split}.img.ex.bin")))
}

/// Write the images of a split, `per_example[i]` being example `i`'s. Nothing is written (and an old sidecar is
/// removed) when no example holds an image.
pub fn write(dir: &Path, split: &str, per_example: &[Vec<ImageBytes>]) -> io::Result<()> {
    let (bin, off, ex) = names(dir, split);
    for stale in [&bin, &off, &ex] {
        match std::fs::remove_file(stale) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    if per_example.iter().all(Vec::is_empty) {
        return Ok(());
    }
    let (mut bytes, mut offsets, mut firsts) = (Vec::new(), Vec::new(), Vec::new());
    let mut count = 0u64;
    for images in per_example {
        firsts.push(count);
        for image in images {
            offsets.push(bytes.len() as u64);
            bytes.extend_from_slice(image);
            count += 1;
        }
    }
    offsets.push(bytes.len() as u64);
    firsts.push(count);
    std::fs::write(&bin, &bytes)?;
    binio::write_u64_bin(&off, &offsets)?;
    binio::write_u64_bin(&ex, &firsts)
}

/// The images of one split of a prepared dataset, read from disk when asked for.
#[derive(Debug)]
pub struct ExampleImages {
    file: PathBuf,
    offsets: Vec<u64>,
    firsts: Vec<u64>,
}

impl ExampleImages {
    /// The sidecar of `split` in `dir`; `None` when the split has no images.
    pub fn open(dir: &Path, split: &str) -> io::Result<Option<ExampleImages>> {
        let (bin, off, ex) = names(dir, split);
        if !bin.exists() {
            return Ok(None);
        }
        let (offsets, firsts) = (binio::read_u64_bin(&off)?, binio::read_u64_bin(&ex)?);
        let len = std::fs::metadata(&bin)?.len();
        let well_formed = offsets.last() == Some(&len)
            && offsets.windows(2).all(|w| w[0] <= w[1])
            && firsts.last().is_some_and(|&n| n as usize + 1 == offsets.len())
            && firsts.windows(2).all(|w| w[0] <= w[1]);
        if !well_formed {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: the image index does not describe the image file", bin.display())));
        }
        Ok(Some(ExampleImages { file: bin, offsets, firsts }))
    }

    /// Examples the index covers.
    pub fn examples(&self) -> usize {
        self.firsts.len() - 1
    }

    /// Images in all.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many images example `example` holds (`0` past the last example).
    pub fn count(&self, example: usize) -> usize {
        match (self.firsts.get(example), self.firsts.get(example + 1)) {
            (Some(a), Some(b)) => (b - a) as usize,
            _ => 0,
        }
    }

    /// The files of example `example`, in the order its text places them.
    pub fn images(&self, example: usize) -> io::Result<Vec<Vec<u8>>> {
        let n = self.count(example);
        if n == 0 {
            return Ok(Vec::new());
        }
        let first = self.firsts[example] as usize;
        let mut file = std::fs::File::open(&self.file)?;
        (first..first + n)
            .map(|i| {
                let mut bytes = vec![0u8; (self.offsets[i + 1] - self.offsets[i]) as usize];
                file.seek(SeekFrom::Start(self.offsets[i]))?;
                file.read_exact(&mut bytes)?;
                Ok(bytes)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(bytes: &[u8]) -> ImageBytes {
        bytes.into()
    }

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-chat-media-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn each_example_gets_back_its_own_images_in_order() {
        let d = dir("roundtrip");
        write(&d, "train", &[vec![img(b"aa"), img(b"b")], vec![], vec![img(b"cccc")]]).unwrap();
        let images = ExampleImages::open(&d, "train").unwrap().expect("a sidecar");
        assert_eq!((images.examples(), images.len()), (3, 3));
        assert_eq!(images.images(0).unwrap(), vec![b"aa".to_vec(), b"b".to_vec()]);
        assert!(images.images(1).unwrap().is_empty());
        assert_eq!(images.images(2).unwrap(), vec![b"cccc".to_vec()]);
        assert!(images.images(7).unwrap().is_empty(), "past the last example there is nothing");
    }

    #[test]
    fn a_split_without_images_has_no_sidecar_and_loses_a_stale_one() {
        let d = dir("stale");
        write(&d, "train", &[vec![img(b"x")]]).unwrap();
        assert!(ExampleImages::open(&d, "train").unwrap().is_some());
        write(&d, "train", &[vec![], vec![]]).unwrap();
        assert!(ExampleImages::open(&d, "train").unwrap().is_none());
    }

    #[test]
    fn an_index_that_does_not_describe_the_file_is_refused() {
        let d = dir("torn");
        write(&d, "val", &[vec![img(b"abc")]]).unwrap();
        std::fs::write(d.join("val.img.bin"), b"ab").unwrap();
        let err = ExampleImages::open(&d, "val").unwrap_err();
        assert!(err.to_string().contains("image index"), "{err}");
    }
}
