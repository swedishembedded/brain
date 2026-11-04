// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for portable, high-throughput
// model storage for its clients. If your team needs expertise in model I/O
// across Linux, macOS and Windows then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Positioned file I/O and page-cache hints that work on every supported
//! system: what checkpoint writers, readers and optimizer state use to read
//! and write a file at an offset, and to let the kernel drop pages of a
//! mapping that are no longer needed.
//!
//! On Unix these are `pread`/`pwrite` and `madvise`. On Windows a read or
//! write at an offset is `seek_read`/`seek_write`, which may transfer fewer
//! bytes than asked and so loop to the end; dropping pages is a hint that
//! Windows has no equivalent of for a mapped file, so it does nothing.

use std::fs::File;
use std::io;

/// Reading and writing a file at an offset, without moving a shared cursor
/// that concurrent callers would race on.
pub trait PositionedIo {
    /// Fills `buf` from `offset`, or fails (`UnexpectedEof` past the end).
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
    /// Writes all of `buf` at `offset`.
    fn write_all_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
}

#[cfg(unix)]
impl PositionedIo for File {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset)
    }

    fn write_all_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        std::os::unix::fs::FileExt::write_all_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl PositionedIo for File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match self.seek_read(buf, offset) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "failed to fill whole buffer")),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match self.seek_write(buf, offset) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer")),
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Lets the kernel drop the pages of `len` bytes of a read-only file mapping
/// from `offset`: best-effort, and nothing at all where the system has no
/// such hint. Dropped pages re-fault from the file if they are read again.
pub fn drop_mapped_pages(mmap: &memmap2::Mmap, offset: usize, len: usize) {
    #[cfg(unix)]
    {
        // SAFETY: a read-only file mapping loses no data to MADV_DONTNEED.
        let _ = unsafe { mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, offset, len) };
    }
    #[cfg(not(unix))]
    let _ = (mmap, offset, len);
}

#[cfg(test)]
mod tests {
    use super::PositionedIo;

    #[test]
    fn a_file_is_written_and_read_back_at_offsets() {
        let dir = std::env::temp_dir().join(format!("brain-pio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.bin");
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
        file.write_all_at(b"world", 6).unwrap();
        file.write_all_at(b"hello ", 0).unwrap();
        let mut back = [0u8; 11];
        file.read_exact_at(&mut back, 0).unwrap();
        assert_eq!(&back, b"hello world");
        let mut past = [0u8; 4];
        assert_eq!(file.read_exact_at(&mut past, 9).unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
