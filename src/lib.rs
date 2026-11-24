//! # rd1-kaede-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact golden-tested
//! emulator for the **Nintendo R&D1 "Kaede" GBA Sound Engine** on the Game Boy Advance.
//!
//! Covers **12 commercial Nintendo R&D1 GBA releases** across **2 distinct ARM7TDMI mixer
//! revisions**:
//! - [`KaedeVersion::V1Lut512`] (**9 ROMs** — *WarioWare, Inc.: Mega Microgame$!* /
//!   *Made in Wario* / *Waliou Zhizao* across USA, Europe, Japan, China, Virtual Console,
//!   Beta, and Kiosk Demo): 3,476-byte IWRAM ARM7TDMI mixer with **512-byte** DirectSound
//!   saturation LUT (`mask = 0x1FF`, headroom `[-256, +255]`, `lut[0x1FF] = 0`).
//! - [`KaedeVersion::V2Lut1024`] (**3 ROMs** — *Rhythm Tengoku* Japan Rev 0, Rev 1, and
//!   Kiosk Demo): identical 10-kernel ARM7TDMI mixer upgraded to a **1,024-byte** DirectSound
//!   saturation LUT (`mask = 0x3FF`, headroom `[-512, +511]`, `lut[0x3FF] = 0`).
//!
//! ## Features
//! - **Zero dependencies** in `[dependencies]`, `#![no_std]` compatible, `#![deny(unsafe_code)]`.
//! - **Zero-allocation binary parsers**: 24-byte `Rd1SampleDesc` PCM8 descriptors
//!   ([`KaedeSampleDesc`]), 20-byte song table entries ([`KaedeSongEntry`]), and verbatim
//!   Standard MIDI (`MThd` / `MTrk`) validator ([`KaedeMidiHeader`]).
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: 100% bit-exact emulation of all 10 Kaede ARM7TDMI mixer kernels
//!     (stereo/mono 14-bit linear with `0x3FC0` fractional mask, stereo 1:1 direct-copy,
//!     stereo/mono nearest-neighbor including the authentic ARM fast-path `mla r1/r3, r4, sb, r2`
//!     quirk, stereo/mono band-pass feedback reverb, stereo 1-pole LPF/HPF, and V1/V2
//!     LUT-clamped DirectSound DMA8 output).
//!   - [`StudioSynth`]: 48,000 Hz `f32` compile-time Lanczos-3 polyphase LUT renderer (with
//!     general Lanczos-`a` fallback) and equal-power stereo panning.
//! - **Byte-exact `libmgba` ARM golden tested**: all 14 ARM7TDMI golden vectors across both
//!   revisions verified strictly byte-for-byte (`assert_eq!`, zero tolerance).
//!
//! ## Quickstart
//!
//! ```rust
//! use rd1_kaede::{AccuracySynth, KaedeFilterState, KaedeVersion, KaedeVoice};
//!
//! // Raw signed 8-bit PCM waveform (0 = silence)
//! let pcm: &[u8] = &[0, 32, 64, 96, 127, 96, 64, 32, 0, 224, 192, 160, 128, 160, 192, 224];
//!
//! // Stereo linear-interpolated looped voice, 1.25x pitch (0x5000 in 18.14 fixed point)
//! let mut voice = KaedeVoice::new(0, pcm.len(), Some(0), 0x5000, 110, 85);
//! let mut acc = [0i32; 64]; // 32 interleaved stereo frames
//! let mut flt = KaedeFilterState::new(0, 0, 0x30); // 1-pole LPF
//! let mut dma_l = [0i8; 32];
//! let mut dma_r = [0i8; 32];
//!
//! AccuracySynth::mix_stereo_linear(pcm, &mut voice, &mut acc);
//! AccuracySynth::apply_stereo_filter(&mut flt, &mut acc);
//! AccuracySynth::pack_stereo_to_dma8(KaedeVersion::V2Lut1024, &acc, &mut dma_l, &mut dma_r);
//! assert!(dma_l.iter().any(|&s| s != 0));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod freq;
pub mod sample;
pub mod synth;
pub mod version;
pub mod voice;

pub use error::{Error, Result};
pub use freq::{
    freq_to_step_q14, midi_note_step_q14, rate_from_timer_reload, step_q14_to_freq,
    volume_pan_gains, GBA_CLOCK_HZ, KAEDE_DEFAULT_MIX_RATE_HZ, Q14_INTERP_MASK, Q14_ONE,
};
#[cfg(feature = "alloc")]
pub use sample::KaedeSample;
pub use sample::{
    KaedeMidiHeader, KaedeSampleDesc, KaedeSampleView, KaedeSongEntry, GBA_ROM_BUS_BASE,
    KAEDE_SAMPLE_DESC_SIZE, KAEDE_SONG_ENTRY_SIZE, MTHD_HEADER_SIZE, MTHD_MAGIC, MTRK_MAGIC,
};
pub use synth::{AccuracySynth, StudioSynth, KAEDE_CLAMP_LUT_1024, KAEDE_CLAMP_LUT_512};
pub use version::KaedeVersion;
pub use voice::{KaedeFilterState, KaedeMixMode, KaedeReverbState, KaedeVoice};
