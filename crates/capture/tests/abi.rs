// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pin the V4L2 ABI constants against the system `<linux/videodev2.h>`.
//!
//! When a C compiler is present, this compiles and runs a tiny program that prints
//! every ioctl number, struct size and field offset the FFI hard-codes, and asserts
//! they match. A kernel/arch where they differ fails HERE, in a fast test, rather
//! than as a corrupt frame or an ENOTTY at runtime. Skips cleanly with no compiler.
#![cfg(target_os = "linux")]

use std::process::Command;

fn cc() -> Option<&'static str> {
    for c in ["cc", "gcc"] {
        if Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(c);
        }
    }
    None
}

#[test]
fn ffi_constants_match_the_system_header() {
    let Some(cc) = cc() else {
        eprintln!("SKIP: no C compiler to verify the V4L2 ABI");
        return;
    };
    let dir = std::env::temp_dir().join("brain_v4l2_abi");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("abi.c");
    let bin = dir.join("abi");
    std::fs::write(&src, r#"
#include <linux/videodev2.h>
#include <stdio.h>
#include <stddef.h>
int main(void){
  printf("QUERYCAP %lu\nS_FMT %lu\nREQBUFS %lu\nQUERYBUF %lu\nQBUF %lu\nDQBUF %lu\nSTREAMON %lu\nSTREAMOFF %lu\n",
    (unsigned long)VIDIOC_QUERYCAP,(unsigned long)VIDIOC_S_FMT,(unsigned long)VIDIOC_REQBUFS,
    (unsigned long)VIDIOC_QUERYBUF,(unsigned long)VIDIOC_QBUF,(unsigned long)VIDIOC_DQBUF,
    (unsigned long)VIDIOC_STREAMON,(unsigned long)VIDIOC_STREAMOFF);
  printf("YUYV %u\n", V4L2_PIX_FMT_YUYV);
  printf("SZCAP %zu\nSZFMT %zu\nSZREQ %zu\nSZBUF %zu\n",
    sizeof(struct v4l2_capability),sizeof(struct v4l2_format),
    sizeof(struct v4l2_requestbuffers),sizeof(struct v4l2_buffer));
  printf("OFF_M_OFFSET %zu\nOFF_LENGTH %zu\nOFF_BYTESUSED %zu\n",
    offsetof(struct v4l2_buffer,m.offset),offsetof(struct v4l2_buffer,length),
    offsetof(struct v4l2_buffer,bytesused));
  printf("OFF_PIX %zu\n", offsetof(struct v4l2_format,fmt.pix));
  printf("OFF_DEVCAPS %zu\n", offsetof(struct v4l2_capability,device_caps));
  return 0;
}
"#).unwrap();
    let o = Command::new(cc).arg("-o").arg(&bin).arg(&src).output().unwrap();
    assert!(o.status.success(), "cc failed: {}", String::from_utf8_lossy(&o.stderr));
    let out = String::from_utf8(Command::new(&bin).output().unwrap().stdout).unwrap();
    let mut m = std::collections::HashMap::new();
    for line in out.lines() {
        let mut it = line.split_whitespace();
        if let (Some(k), Some(v)) = (it.next(), it.next()) {
            m.insert(k.to_string(), v.parse::<u64>().unwrap());
        }
    }
    use capture::v4l2::*;
    let eq = |k: &str, got: u64| assert_eq!(m[k], got, "V4L2 `{k}` drifted: header {} vs FFI {got}", m[k]);
    eq("QUERYCAP", VIDIOC_QUERYCAP as u64);
    eq("S_FMT", VIDIOC_S_FMT as u64);
    eq("REQBUFS", VIDIOC_REQBUFS as u64);
    eq("QUERYBUF", VIDIOC_QUERYBUF as u64);
    eq("QBUF", VIDIOC_QBUF as u64);
    eq("DQBUF", VIDIOC_DQBUF as u64);
    eq("STREAMON", VIDIOC_STREAMON as u64);
    eq("STREAMOFF", VIDIOC_STREAMOFF as u64);
    eq("YUYV", V4L2_PIX_FMT_YUYV as u64);
    eq("SZCAP", SZ_CAPABILITY as u64);
    eq("SZFMT", SZ_FORMAT as u64);
    eq("SZREQ", SZ_REQUESTBUFFERS as u64);
    eq("SZBUF", SZ_BUFFER as u64);
}
