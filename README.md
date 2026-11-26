# rd1-kaede-rs

**Zero-dependency, pure Rust parser, software synthesizer, and byte-exact golden-tested emulator for the Nintendo R&D1 "Kaede" GBA Sound Engine (Game Boy Advance).**

[![Crates.io](https://img.shields.io/crates/v/rd1-kaede.svg)](https://crates.io/crates/rd1-kaede)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

The **Nintendo R&D1 "Kaede"** Standard MIDI (`MThd`) DirectSound engine powered **12 commercial Game Boy Advance releases** across two hardware mixer revisions — including *WarioWare, Inc.: Mega Microgame$!* (*Made in Wario* / *Waliou Zhizao*) and *Rhythm Tengoku*.

`rd1-kaede` provides a clean-room, reverse-engineered implementation designed for audio preservation, DAWs, PC source ports, and emulators.

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, extract WAV samples, and compile SoundFont 2.04 (.sf2) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

---

## Features

- **Zero External Dependencies**: Pure Rust with no dependencies in `[dependencies]`, `#![deny(unsafe_code)]`.
- **`#![no_std]` Compatible**: Ready for embedded targets and WASM runtimes (`alloc` optional).
- **Zero-Allocation Binary Parsers**: Parsers for 24-byte `Rd1SampleDesc` PCM8 sample descriptors (`KaedeSampleDesc`), 20-byte `Rd1SongEntry` song table rows (`KaedeSongEntry`), and verbatim Standard MIDI (`MThd`/`MTrk`) validation (`KaedeMidiHeader`).
- **Dual Synthesis Engines**:
  - **`AccuracySynth`**: 100% bit-exact emulation of all **10** Nintendo R&D1 Kaede ARM7TDMI IWRAM mixer kernels across both hardware revisions (`V1Lut512` & `V2Lut1024`) — including 18.14 fixed-point linear interpolation (`fp & 0x3FC0`), 1:1 direct-copy, nearest-neighbor (including the authentic ARM fast-path `mla r1/r3, r4, sb, r2` quirk), stereo/mono band-pass feedback reverb, stereo 1-pole LPF/HPF post-mix filtering, and revision-specific 512-byte vs 1,024-byte DMA8 saturation LUTs (including the `lut[mask] = 0` quirk).
  - **`StudioSynth`**: Renders at 48,000 Hz, 32-bit floating-point precision with a compile-time 1,024-phase Lanczos-3 polyphase LUT (plus general Lanczos-`a` fallback) and true equal-power stereo panning.
- **Byte-Exact `libmgba` ARM Golden Tested**: **14 ARM7TDMI golden vectors (2,368 / 2,368 output bytes)** verified strictly byte-for-byte (`assert_eq!`, zero tolerance) against real ARM7TDMI execution in headless `libmgba.so.0.10`.

---

## Supported Nintendo R&D1 Kaede Engine Revisions

Auditing all 3,236 unique GBA ROMs in the corpus identifies **12 commercial Nintendo R&D1 releases** across **2 distinct ARM7TDMI mixer revisions** (both sharing the exact 3,476-byte `0xD94` IWRAM ARM mixer block, differing in the DMA8 clamping LUT mask at `+0x7F4`), all supported and byte-exact golden-tested via `KaedeVersion`:

| Revision | Corpus Share | Representative Titles | Embedded MIDIs / Samples | DMA8 Clamping LUT | Wraparound Headroom (`acc >> 7`) |
| --- | ---: | --- | --- | --- | --- |
| `V1Lut512` | 9 ROMs | *WarioWare, Inc.* / *Made in Wario* / *Waliou Zhizao* (`AZWC`, `AZWE` ×3, `AZWJ` ×3, `AZWP` ×2) | 1,342 `.mid` / 613–690 PCM8 | 512-byte LUT (`mask = 0x1FF`, `lut[0x1FF] = 0`) | `[-256, +255]` |
| `V2Lut1024` | 3 ROMs | *Rhythm Tengoku* Rev 0, Rev 1, Kiosk Demo (`BRIJ` r0/r1, `BZLJ`) | 722 `.mid` / 959–960 PCM8 | 1,024-byte LUT (`mask = 0x3FF`, `lut[0x3FF] = 0`) | `[-512, +511]` |

---

## Quickstart

### Zero-Allocation Mixing (`#![no_std]`)

```rust
use rd1_kaede::{AccuracySynth, KaedeFilterState, KaedeVersion, KaedeVoice};

// Raw signed 8-bit PCM waveform (0 = silence)
let pcm: &[u8] = &[0, 32, 64, 96, 127, 96, 64, 32, 0, 224, 192, 160, 128, 160, 192, 224];

// Stereo linear-interpolated looped voice, 1.25x pitch (0x5000 in 18.14 fixed point)
let mut voice = KaedeVoice::new(0, pcm.len(), Some(0), 0x5000, 110, 85);
let mut acc = [0i32; 64]; // 32 interleaved stereo frames
let mut flt = KaedeFilterState::new(0, 0, 0x30); // 1-pole LPF
let mut dma_l = [0i8; 32];
let mut dma_r = [0i8; 32];

AccuracySynth::mix_stereo_linear(pcm, &mut voice, &mut acc);
AccuracySynth::apply_stereo_filter(&mut flt, &mut acc);
AccuracySynth::pack_stereo_to_dma8(KaedeVersion::V2Lut1024, &acc, &mut dma_l, &mut dma_r);
```

---

## Golden Test Methodology

Every single ARM7TDMI kernel in `AccuracySynth` is verified strictly byte-for-byte (`assert_eq!`, zero tolerance) against real ARM7TDMI execution traces captured in headless `libmgba.so.0.10` using clean-room synthetic deterministic waveforms:

- **14 ARM7TDMI golden vectors (2,368 output bytes)**:
  1. `stereo_linear_oneshot` (Kernel 1, 256 bytes + end phase & inactive flag)
  2. `stereo_linear_looped` (Kernel 1, 256 bytes + end phase)
  3. `mono_linear_looped` (Kernel 2, 128 bytes + end phase)
  4. `stereo_reverb_biquad` (Kernel 3, 256 bytes, cascaded HP+LP with ring wrap)
  5. `mono_reverb_biquad` (Kernel 4, 128 bytes, cascaded HP+LP with ring wrap)
  6. `v1_lut512_dma_stereo` (Kernel 5, 64 bytes — WarioWare `0x1FF` LUT + `-1 -> 0` quirk + modulo-512 wrap)
  7. `v2_lut1024_dma_stereo` (Kernel 5, 64 bytes — Rhythm Tengoku `0x3FF` LUT on identical input, 28/64 bytes diverge from V1)
  8. `v1_lut512_dma_mono` (Kernel 6, 32 bytes — WarioWare `0x1FF` mono LUT)
  9. `v2_lut1024_dma_mono` (Kernel 6, 32 bytes — Rhythm Tengoku `0x3FF` mono LUT)
  10. `stereo_direct_copy` (Kernel 7, 256 bytes — unaligned start, 4-byte bulk load, loop wrap)
  11. `stereo_nearest_looped` (Kernel 8, 256 bytes + end phase)
  12. `mono_nearest_quirk` (Kernel 9, 128 bytes — verifies authentic ARM fast-path `mla r1/r3, r4, sb, r2`)
  13. `stereo_filter_lpf` (Kernel 10, 256 bytes + filter state, `cutoff = 0x40`)
  14. `stereo_filter_hpf` (Kernel 10, 256 bytes + filter state, `cutoff = 0xC0`)

---

## Benchmarks

```sh
cargo bench --bench bench_audio
```

| Benchmark | Throughput |
| --- | ---: |
| `AccuracySynth::mix_stereo_linear + pack_stereo_to_dma8` (`V2Lut1024`, 128f) | ~821.2 MSamples/s |
| `AccuracySynth::mix_stereo_direct_copy` (128f) | ~2.35 GSamples/s |
| `AccuracySynth::apply_stereo_reverb + apply_stereo_filter` (128f) | ~387.2 MSamples/s |
| `StudioSynth::render_voice_f32` (48 kHz Lanczos-3, 1,024f) | ~408.2 MSamples/s |

---

## Community & Support

Come discuss the wonders of GBA RE, modding, etc. — also the best way to report issues or bugs, or ask for help on your own project: https://discord.gg/khwxFSVNrS

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
