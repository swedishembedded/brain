// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for moving bulk data between
// isolated processes for its clients. If your team needs expertise in Unix
// descriptor passing then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Descriptors beside frames on a real socket pair: each frame gets the
//! descriptors it declared, a blob larger than any line travels in a sealed
//! memfd, and a descriptor the receiver cannot trust is refused.

use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;

use gpu_supervisor::wire::{read_sealed, seal, send, Reader, MAX_BLOB_BYTES};

fn read_line(reader: &mut Reader, sock: &UnixStream) -> String {
    loop {
        if let Some(line) = reader.line().expect("a well-formed line") {
            return line;
        }
        assert!(reader.fill(sock).expect("the peer is alive") > 0, "end of stream before a line");
    }
}

#[test]
fn each_frame_gets_its_own_descriptors_and_a_blob_bigger_than_a_line_travels_in_a_memfd() {
    let (a, b) = UnixStream::pair().unwrap();
    // Eight MiB: eight times the longest line either end accepts.
    let big: Vec<u8> = (0..8u32 << 20).map(|i| (i % 251) as u8).collect();
    let first = seal("first", &big).unwrap();
    let second = seal("second", b"two").unwrap();
    let third = seal("third", b"three").unwrap();
    send(&a, b"1 ok+1 {}\n", &[first.as_fd()]).unwrap();
    send(&a, b"2 progress {}\n", &[]).unwrap();
    send(&a, b"3 ok+2 {}\n", &[second.as_fd(), third.as_fd()]).unwrap();
    drop((first, second, third));

    let mut reader = Reader::new(1 << 20);
    assert_eq!(read_line(&mut reader, &b), "1 ok+1 {}");
    let got: Vec<OwnedFd> = reader.take_fds(1).unwrap();
    assert_eq!(read_sealed(&got[0], MAX_BLOB_BYTES).unwrap(), big);
    assert_eq!(read_line(&mut reader, &b), "2 progress {}");
    assert_eq!(read_line(&mut reader, &b), "3 ok+2 {}");
    let got = reader.take_fds(2).unwrap();
    assert_eq!(read_sealed(&got[0], MAX_BLOB_BYTES).unwrap(), b"two");
    assert_eq!(read_sealed(&got[1], MAX_BLOB_BYTES).unwrap(), b"three");
    assert_eq!(reader.unclaimed_fds(), 0);
    assert!(reader.take_fds(1).is_err(), "a descriptor that never arrived cannot be taken");
}

#[test]
fn a_descriptor_that_is_not_sealed_or_is_too_large_is_refused() {
    let (a, b) = UnixStream::pair().unwrap();
    let unsealed: OwnedFd = std::fs::File::open("/proc/self/stat").unwrap().into();
    send(&a, b"1 ok+1 {}\n", &[unsealed.as_fd()]).unwrap();
    let mut reader = Reader::new(1 << 20);
    read_line(&mut reader, &b);
    let fd = reader.take_fds(1).unwrap().pop().unwrap();
    let err = read_sealed(&fd, MAX_BLOB_BYTES).expect_err("its sender could change an unsealed file while it is read");
    assert!(err.to_string().contains("sealed"), "{err}");
    let blob = seal("blob", &[7u8; 4096]).unwrap();
    let err = read_sealed(&blob, 4095).expect_err("over the bound");
    assert!(err.to_string().contains("4096"), "{err}");
}

#[test]
fn a_line_longer_than_the_bound_is_refused_without_waiting_for_its_end() {
    let (a, b) = UnixStream::pair().unwrap();
    send(&a, &[b'x'; 2048], &[]).unwrap();
    let mut reader = Reader::new(1024);
    reader.fill(&b).unwrap();
    assert!(reader.line().is_err(), "a line past the bound is a protocol error");
}
