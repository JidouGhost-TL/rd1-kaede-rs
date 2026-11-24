//! Dual synthesis engines for Nintendo R&D1 "Kaede":
//! - [`AccuracySynth`]: 100% bit-exact emulation of all 10 ARM7TDMI mixer kernels across both
//!   Kaede hardware revisions ([`KaedeVersion::V1Lut512`] and [`KaedeVersion::V2Lut1024`]).
//! - [`StudioSynth`]: 48,000 Hz `f32` compile-time 1,024-phase Lanczos-3 windowed-sinc
//!   resampler (with general Lanczos-`a` fallback) and true equal-power stereo panning.

use crate::freq::Q14_INTERP_MASK;
use crate::sample::KaedeSampleView;
use crate::version::KaedeVersion;
use crate::voice::{KaedeFilterState, KaedeMixMode, KaedeReverbState, KaedeVoice};

// ============================================================================
// Compile-Time Hardware Clamping LUTs (V1 512-byte & V2 1024-byte)
// ============================================================================

const fn build_kaede_clamping_lut<const N: usize>() -> [i8; N] {
    let mut lut = [0i8; N];
    let half = N / 2;
    let mut i = 0usize;
    while i < N {
        if i < half {
            lut[i] = if i <= 127 { i as i8 } else { 127 };
        } else {
            let signed_val = (i as i32) - (N as i32);
            lut[i] = if signed_val <= -129 {
                -128
            } else {
                (i as u8) as i8
            };
        }
        i += 1;
    }
    // Authentic Nintendo R&D1 Kaede Thumb LUT init (`strb r0, [lut_base + mask]`):
    // maps shifted accumulator value -1 (`acc in -128..-1`) to 0 instead of -1.
    lut[N - 1] = 0;
    lut
}

/// Exact 512-byte DirectSound output clamping LUT used by [`KaedeVersion::V1Lut512`]
/// (*WarioWare, Inc.* / *Made in Wario* / *Waliou Zhizao*).
pub const KAEDE_CLAMP_LUT_512: [i8; 512] = build_kaede_clamping_lut::<512>();

/// Exact 1,024-byte DirectSound output clamping LUT used by [`KaedeVersion::V2Lut1024`]
/// (*Rhythm Tengoku*).
pub const KAEDE_CLAMP_LUT_1024: [i8; 1024] = build_kaede_clamping_lut::<1024>();

// ============================================================================
// AccuracySynth: Bit-Exact ARM7TDMI Software Mixer Emulation
// ============================================================================

/// Bit-exact software synthesizer implementing all 10 ARM7TDMI mixer kernels of the
/// Nintendo R&D1 "Kaede" DirectSound engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct AccuracySynth;

impl AccuracySynth {
    /// Unified voice mixer dispatcher for any [`KaedeMixMode`].
    ///
    /// - For stereo modes (`StereoLinear`, `StereoDirectCopy`, `StereoNearest`), `acc` is an
    ///   interleaved `[L0, R0, L1, R1, ...]` 32-bit accumulator slice (length = `frames * 2`).
    /// - For mono modes (`MonoLinear`, `MonoNearest`), `acc` is a mono `[M0, M1, ...]` 32-bit
    ///   accumulator slice (length = `frames`).
    pub fn mix_voice(
        mode: KaedeMixMode,
        pcm_bytes: &[u8],
        voice: &mut KaedeVoice,
        acc: &mut [i32],
    ) {
        match mode {
            KaedeMixMode::StereoLinear => Self::mix_stereo_linear(pcm_bytes, voice, acc),
            KaedeMixMode::MonoLinear => Self::mix_mono_linear(pcm_bytes, voice, acc),
            KaedeMixMode::StereoDirectCopy => Self::mix_stereo_direct_copy(pcm_bytes, voice, acc),
            KaedeMixMode::StereoNearest => Self::mix_stereo_nearest(pcm_bytes, voice, acc),
            KaedeMixMode::MonoNearest => Self::mix_mono_nearest(pcm_bytes, voice, acc),
        }
    }

    /// **ARM Kernel 1 (`+0x094`)**: Stereo 18.14 fixed-point linear-interpolated PCM8 mixer.
    ///
    /// Uses Kaede's exact 8-bit quantized fractional linear interpolation:
    /// `s = s0 + (((s1 - s0) * (phase_q14 & 0x3FC0)) >> 14)`, accumulating
    /// `s * vol_l` and `s * vol_r` into interleaved `[L, R]` 32-bit accumulators `acc`.
    /// Processes blocks of 4 stereo frames (`acc.len() / 2` rounded down to a multiple of 4,
    /// matching ARM `ip & 0xFF`).
    pub fn mix_stereo_linear(pcm_bytes: &[u8], voice: &mut KaedeVoice, acc: &mut [i32]) {
        if !voice.active || pcm_bytes.is_empty() {
            return;
        }
        let blocks = (acc.len() / 2) / 4;
        if blocks == 0 {
            return;
        }

        let pcm_len = pcm_bytes.len();
        let step = voice.step_q14;
        let step4 = step.wrapping_shl(2);
        let end_fp = voice.end_q14;
        let loop_delta = voice.loop_delta_q14();
        let vol_l = voice.vol_l;
        let vol_r = voice.vol_r;
        let mut fp = voice.phase_q14;
        let mut out_idx = 0usize;

        for _ in 0..blocks {
            if fp.wrapping_add(step4) < end_fp {
                for _ in 0..4 {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                    acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                    out_idx += 2;
                    fp = fp.wrapping_add(step);
                }
            } else if loop_delta != 0 {
                for _ in 0..4 {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                    acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                    out_idx += 2;
                    fp = fp.wrapping_add(step);
                    while fp >= end_fp {
                        fp = fp.wrapping_add(loop_delta as u32);
                    }
                }
            } else {
                loop {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                    acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                    out_idx += 2;
                    fp = fp.wrapping_add(step);
                    if fp >= end_fp || out_idx + 1 >= acc.len() {
                        voice.phase_q14 = fp;
                        voice.active = false;
                        return;
                    }
                }
            }
        }

        voice.phase_q14 = fp;
    }

    /// **ARM Kernel 2 (`+0x228`)**: Mono 18.14 fixed-point linear-interpolated PCM8 mixer.
    ///
    /// Uses `voice.vol_l` as the mono gain and accumulates into `acc` in blocks of 4 frames.
    pub fn mix_mono_linear(pcm_bytes: &[u8], voice: &mut KaedeVoice, acc: &mut [i32]) {
        if !voice.active || pcm_bytes.is_empty() {
            return;
        }
        let blocks = acc.len() / 4;
        if blocks == 0 {
            return;
        }

        let pcm_len = pcm_bytes.len();
        let step = voice.step_q14;
        let step4 = step.wrapping_shl(2);
        let end_fp = voice.end_q14;
        let loop_delta = voice.loop_delta_q14();
        let vol = voice.vol_l;
        let mut fp = voice.phase_q14;
        let mut out_idx = 0usize;

        for _ in 0..blocks {
            if fp.wrapping_add(step4) < end_fp {
                for _ in 0..4 {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol));
                    out_idx += 1;
                    fp = fp.wrapping_add(step);
                }
            } else if loop_delta != 0 {
                for _ in 0..4 {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol));
                    out_idx += 1;
                    fp = fp.wrapping_add(step);
                    while fp >= end_fp {
                        fp = fp.wrapping_add(loop_delta as u32);
                    }
                }
            } else {
                loop {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol));
                    out_idx += 1;
                    fp = fp.wrapping_add(step);
                    if fp >= end_fp || out_idx >= acc.len() {
                        voice.phase_q14 = fp;
                        voice.active = false;
                        return;
                    }
                }
            }
        }

        voice.phase_q14 = fp;
    }

    /// **ARM Kernel 3 (`+0x438`)**: Stereo band-pass feedback reverb / echo delay tap.
    ///
    /// Reads signed 8-bit samples from `delay_l` and `delay_r` ring buffers, applies Kaede's
    /// cascaded 1-pole high-pass (`hp_shift`) + 1-pole low-pass (`lp_shift`) filter, scales by
    /// `reverb_vol`, and overwrites interleaved `[L, R]` 32-bit accumulators `out_acc`.
    pub fn apply_stereo_reverb(
        delay_l: &[u8],
        delay_r: &[u8],
        state: &mut KaedeReverbState,
        out_acc: &mut [i32],
    ) {
        if delay_l.is_empty() || delay_r.is_empty() {
            return;
        }
        let blocks = (out_acc.len() / 2) / 4;
        if blocks == 0 {
            return;
        }

        let len_l = delay_l.len();
        let len_r = delay_r.len();
        let hp_sh = state.hp_shift & 31;
        let lp_sh = state.lp_shift & 31;
        let vol = state.reverb_vol;
        let mut hp_l = state.hp_l;
        let mut lp_l = state.lp_l;
        let mut hp_r = state.hp_r;
        let mut lp_r = state.lp_r;
        let mut pos_l = state.pos_l % len_l;
        let mut pos_r = state.pos_r % len_r;
        let mut out_idx = 0usize;

        for _ in 0..blocks {
            for _ in 0..4 {
                let sl = delay_l[pos_l] as i8 as i32;
                pos_l += 1;
                hp_l = hp_l.wrapping_sub(hp_l >> hp_sh).wrapping_add(sl);
                let hpo_l = sl.wrapping_sub(hp_l >> hp_sh);
                lp_l = lp_l.wrapping_sub(lp_l >> lp_sh).wrapping_add(hpo_l);
                out_acc[out_idx] = lp_l.wrapping_mul(vol);

                let sr = delay_r[pos_r] as i8 as i32;
                pos_r += 1;
                hp_r = hp_r.wrapping_sub(hp_r >> hp_sh).wrapping_add(sr);
                let hpo_r = sr.wrapping_sub(hp_r >> hp_sh);
                lp_r = lp_r.wrapping_sub(lp_r >> lp_sh).wrapping_add(hpo_r);
                out_acc[out_idx + 1] = lp_r.wrapping_mul(vol);

                out_idx += 2;
            }
            if pos_l >= len_l {
                pos_l = 0;
            }
            if pos_r >= len_r {
                pos_r = 0;
            }
        }

        state.hp_l = hp_l;
        state.lp_l = lp_l;
        state.hp_r = hp_r;
        state.lp_r = lp_r;
        state.pos_l = pos_l;
        state.pos_r = pos_r;
    }

    /// **ARM Kernel 4 (`+0x5AC`)**: Mono band-pass feedback reverb / echo delay tap.
    ///
    /// Uses `state.hp_l`, `state.lp_l`, and `state.pos_l` against `delay_buf`, writing 32-bit
    /// mono outputs into `out_acc`.
    pub fn apply_mono_reverb(
        delay_buf: &[u8],
        state: &mut KaedeReverbState,
        out_acc: &mut [i32],
    ) {
        if delay_buf.is_empty() {
            return;
        }
        let blocks = out_acc.len() / 4;
        if blocks == 0 {
            return;
        }

        let len = delay_buf.len();
        let hp_sh = state.hp_shift & 31;
        let lp_sh = state.lp_shift & 31;
        let vol = state.reverb_vol;
        let mut hp = state.hp_l;
        let mut lp = state.lp_l;
        let mut pos = state.pos_l % len;
        let mut out_idx = 0usize;

        for _ in 0..blocks {
            for _ in 0..4 {
                let s = delay_buf[pos] as i8 as i32;
                pos += 1;
                hp = hp.wrapping_sub(hp >> hp_sh).wrapping_add(s);
                let hpo = s.wrapping_sub(hp >> hp_sh);
                lp = lp.wrapping_sub(lp >> lp_sh).wrapping_add(hpo);
                out_acc[out_idx] = lp.wrapping_mul(vol);
                out_idx += 1;
            }
            if pos >= len {
                pos -= len;
            }
        }

        state.hp_l = hp;
        state.lp_l = lp;
        state.pos_l = pos;
    }

    /// **ARM Kernel 5 (`+0x6C8`)**: Stereo 32-bit accumulator to 8-bit DirectSound DMA output
    /// using the revision-specific saturation LUT ([`KaedeVersion::V1Lut512`] 512-byte LUT vs
    /// [`KaedeVersion::V2Lut1024`] 1,024-byte LUT).
    ///
    /// Computes `lut[((acc as u32) >> 7) & mask]` for each interleaved `(L, R)` sample pair.
    pub fn pack_stereo_to_dma8(
        version: KaedeVersion,
        acc_interleaved: &[i32],
        dma_l: &mut [i8],
        dma_r: &mut [i8],
    ) {
        let frames = (acc_interleaved.len() / 2)
            .min(dma_l.len())
            .min(dma_r.len());
        match version {
            KaedeVersion::V1Lut512 => {
                for i in 0..frames {
                    let l_idx = ((acc_interleaved[i * 2] as u32) >> 7) as usize & 0x1FF;
                    let r_idx = ((acc_interleaved[i * 2 + 1] as u32) >> 7) as usize & 0x1FF;
                    dma_l[i] = KAEDE_CLAMP_LUT_512[l_idx];
                    dma_r[i] = KAEDE_CLAMP_LUT_512[r_idx];
                }
            }
            KaedeVersion::V2Lut1024 => {
                for i in 0..frames {
                    let l_idx = ((acc_interleaved[i * 2] as u32) >> 7) as usize & 0x3FF;
                    let r_idx = ((acc_interleaved[i * 2 + 1] as u32) >> 7) as usize & 0x3FF;
                    dma_l[i] = KAEDE_CLAMP_LUT_1024[l_idx];
                    dma_r[i] = KAEDE_CLAMP_LUT_1024[r_idx];
                }
            }
        }
    }

    /// **ARM Kernel 6 (`+0x758`)**: Mono 32-bit accumulator to 8-bit DirectSound DMA output
    /// using the revision-specific saturation LUT ([`KaedeVersion::V1Lut512`] vs
    /// [`KaedeVersion::V2Lut1024`]).
    pub fn pack_mono_to_dma8(version: KaedeVersion, acc_mono: &[i32], dma_out: &mut [i8]) {
        let frames = acc_mono.len().min(dma_out.len());
        match version {
            KaedeVersion::V1Lut512 => {
                for i in 0..frames {
                    let idx = ((acc_mono[i] as u32) >> 7) as usize & 0x1FF;
                    dma_out[i] = KAEDE_CLAMP_LUT_512[idx];
                }
            }
            KaedeVersion::V2Lut1024 => {
                for i in 0..frames {
                    let idx = ((acc_mono[i] as u32) >> 7) as usize & 0x3FF;
                    dma_out[i] = KAEDE_CLAMP_LUT_1024[idx];
                }
            }
        }
    }

    /// **ARM Kernel 7 (`+0x880`)**: Stereo 1:1 native-rate direct-copy fast-path PCM8 mixer.
    ///
    /// Reads raw signed 8-bit samples sequentially starting at `voice.sample_pos()`,
    /// accumulating `s * vol_l` and `s * vol_r` into interleaved `[L, R]` 32-bit accumulators.
    pub fn mix_stereo_direct_copy(pcm_bytes: &[u8], voice: &mut KaedeVoice, acc: &mut [i32]) {
        if !voice.active || pcm_bytes.is_empty() {
            return;
        }
        let frames = acc.len() / 2;
        if frames == 0 {
            return;
        }

        let end_pos = ((voice.end_q14 >> 14) as usize).min(pcm_bytes.len());
        let loop_start = voice.loop_start_q14.map(|ls| (ls >> 14) as usize);
        let vol_l = voice.vol_l;
        let vol_r = voice.vol_r;
        let mut pos = voice.sample_pos();
        let mut rem = frames;
        let mut out_idx = 0usize;

        while rem > 0 {
            if pos >= end_pos {
                if let Some(ls) = loop_start {
                    if ls < end_pos {
                        pos = ls + (pos - end_pos);
                        if pos >= end_pos {
                            pos = ls;
                        }
                    } else {
                        voice.active = false;
                        break;
                    }
                } else {
                    voice.active = false;
                    break;
                }
            }

            let run = (end_pos - pos).min(rem);
            for &b in &pcm_bytes[pos..pos + run] {
                let s = b as i8 as i32;
                acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                out_idx += 2;
            }
            pos += run;
            rem -= run;

            if pos >= end_pos {
                if let Some(ls) = loop_start {
                    if ls < end_pos {
                        pos = ls;
                    } else {
                        voice.active = false;
                        break;
                    }
                } else {
                    voice.active = false;
                    break;
                }
            }
        }

        voice.phase_q14 = (pos as u32) << 14;
    }

    /// **ARM Kernel 8 (`+0xA34`)**: Stereo 18.14 fixed-point nearest-neighbor PCM8 mixer.
    pub fn mix_stereo_nearest(pcm_bytes: &[u8], voice: &mut KaedeVoice, acc: &mut [i32]) {
        if !voice.active || pcm_bytes.is_empty() {
            return;
        }
        let blocks = (acc.len() / 2) / 4;
        if blocks == 0 {
            return;
        }

        let pcm_len = pcm_bytes.len();
        let step = voice.step_q14;
        let step4 = step.wrapping_shl(2);
        let end_fp = voice.end_q14;
        let loop_delta = voice.loop_delta_q14();
        let vol_l = voice.vol_l;
        let vol_r = voice.vol_r;
        let mut fp = voice.phase_q14;
        let mut out_idx = 0usize;

        for _ in 0..blocks {
            if fp.wrapping_add(step4) < end_fp {
                for _ in 0..4 {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                    acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                    out_idx += 2;
                    fp = fp.wrapping_add(step);
                }
            } else if loop_delta != 0 {
                for _ in 0..4 {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                    acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                    out_idx += 2;
                    fp = fp.wrapping_add(step);
                    while fp >= end_fp {
                        fp = fp.wrapping_add(loop_delta as u32);
                    }
                }
            } else {
                loop {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol_l));
                    acc[out_idx + 1] = acc[out_idx + 1].wrapping_add(s.wrapping_mul(vol_r));
                    out_idx += 2;
                    fp = fp.wrapping_add(step);
                    if fp >= end_fp || out_idx + 1 >= acc.len() {
                        voice.phase_q14 = fp;
                        voice.active = false;
                        return;
                    }
                }
            }
        }

        voice.phase_q14 = fp;
    }

    /// **ARM Kernel 9 (`+0xB30`)**: Mono 18.14 fixed-point nearest-neighbor PCM8 mixer.
    ///
    /// Faithfully reproduces the authentic Nintendo R&D1 ARM7TDMI fast-path register typo
    /// (`mla r1, r4, sb, r2` at `+0xB58` and `mla r3, r4, sb, r2` at `+0xB78`), where samples
    /// 1 and 3 of each 4-sample fast-path block accumulate onto `r2` instead of `r1`/`r3`.
    pub fn mix_mono_nearest(pcm_bytes: &[u8], voice: &mut KaedeVoice, acc: &mut [i32]) {
        if !voice.active || pcm_bytes.is_empty() {
            return;
        }
        let blocks = acc.len() / 4;
        if blocks == 0 {
            return;
        }

        let pcm_len = pcm_bytes.len();
        let step = voice.step_q14;
        let step4 = step.wrapping_shl(2);
        let end_fp = voice.end_q14;
        let loop_delta = voice.loop_delta_q14();
        let vol = voice.vol_l;
        let mut fp = voice.phase_q14;
        let mut out_idx = 0usize;

        for _ in 0..blocks {
            if fp.wrapping_add(step4) < end_fp {
                let mut r0 = acc[out_idx];
                let mut r2 = acc[out_idx + 2];

                let s0 = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                r0 = r0.wrapping_add(s0.wrapping_mul(vol));
                fp = fp.wrapping_add(step);

                let s1 = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                // Authentic ARM7TDMI instruction at +0xB58: `mla r1, r4, sb, r2`
                let r1 = r2.wrapping_add(s1.wrapping_mul(vol));
                fp = fp.wrapping_add(step);

                let s2 = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                r2 = r2.wrapping_add(s2.wrapping_mul(vol));
                fp = fp.wrapping_add(step);

                let s3 = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                // Authentic ARM7TDMI instruction at +0xB78: `mla r3, r4, sb, r2`
                let r3 = r2.wrapping_add(s3.wrapping_mul(vol));
                fp = fp.wrapping_add(step);

                acc[out_idx] = r0;
                acc[out_idx + 1] = r1;
                acc[out_idx + 2] = r2;
                acc[out_idx + 3] = r3;
                out_idx += 4;
            } else if loop_delta != 0 {
                for _ in 0..4 {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol));
                    out_idx += 1;
                    fp = fp.wrapping_add(step);
                    while fp >= end_fp {
                        fp = fp.wrapping_add(loop_delta as u32);
                    }
                }
            } else {
                loop {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    acc[out_idx] = acc[out_idx].wrapping_add(s.wrapping_mul(vol));
                    out_idx += 1;
                    fp = fp.wrapping_add(step);
                    if fp >= end_fp || out_idx >= acc.len() {
                        voice.phase_q14 = fp;
                        voice.active = false;
                        return;
                    }
                }
            }
        }

        voice.phase_q14 = fp;
    }

    /// **ARM Kernel 10 (`+0xC44`)**: In-place stereo 1-pole Low-Pass / High-Pass post-mix filter.
    ///
    /// - `state.cutoff < 0x80`: 1-pole LPF (`alpha = cutoff << 1`, `beta = 256 - alpha`,
    ///   `y = (beta * x + alpha * y_prev) >> 8`).
    /// - `state.cutoff >= 0x80`: 1-pole HPF (`alpha = (cutoff - 0x80) << 1`, `beta = 256 - alpha`,
    ///   `lp = (beta * x + alpha * lp_prev) >> 8`, `out = x - lp`).
    pub fn apply_stereo_filter(state: &mut KaedeFilterState, acc_interleaved: &mut [i32]) {
        let blocks = (acc_interleaved.len() / 2) / 4;
        if blocks == 0 {
            return;
        }
        let frames = blocks * 4;
        let mut st_l = state.state_l;
        let mut st_r = state.state_r;

        if state.cutoff < 0x80 {
            let alpha = (state.cutoff as i32) << 1;
            let beta = 256 - alpha;
            for i in 0..frames {
                let xl = acc_interleaved[i * 2];
                let xr = acc_interleaved[i * 2 + 1];
                st_l = (beta.wrapping_mul(xl).wrapping_add(alpha.wrapping_mul(st_l))) >> 8;
                st_r = (beta.wrapping_mul(xr).wrapping_add(alpha.wrapping_mul(st_r))) >> 8;
                acc_interleaved[i * 2] = st_l;
                acc_interleaved[i * 2 + 1] = st_r;
            }
        } else {
            let alpha = ((state.cutoff - 0x80) as i32) << 1;
            let beta = 256 - alpha;
            for i in 0..frames {
                let xl = acc_interleaved[i * 2];
                let xr = acc_interleaved[i * 2 + 1];
                let lp_l = (beta.wrapping_mul(xl).wrapping_add(alpha.wrapping_mul(st_l))) >> 8;
                let lp_r = (beta.wrapping_mul(xr).wrapping_add(alpha.wrapping_mul(st_r))) >> 8;
                acc_interleaved[i * 2] = xl.wrapping_sub(lp_l);
                acc_interleaved[i * 2 + 1] = xr.wrapping_sub(lp_r);
                st_l = lp_l;
                st_r = lp_r;
            }
        }

        state.state_l = st_l;
        state.state_r = st_r;
    }
}

#[inline(always)]
fn interp_q14_sample(pcm_bytes: &[u8], pcm_len: usize, fp: u32) -> i32 {
    let idx = (fp >> 14) as usize;
    if idx >= pcm_len {
        return 0;
    }
    let s0 = pcm_bytes[idx] as i8 as i32;
    let s1 = if idx + 1 < pcm_len {
        pcm_bytes[idx + 1] as i8 as i32
    } else {
        0
    };
    let frac = (fp & Q14_INTERP_MASK) as i32;
    s0 + (((s1 - s0) * frac) >> 14)
}

#[inline(always)]
fn nearest_q14_sample(pcm_bytes: &[u8], pcm_len: usize, fp: u32) -> i32 {
    let idx = (fp >> 14) as usize;
    if idx < pcm_len {
        pcm_bytes[idx] as i8 as i32
    } else {
        0
    }
}

// ============================================================================
// StudioSynth: 48,000 Hz f32 Band-Limited Lanczos-3 + Equal-Power Panning
// ============================================================================

/// Number of fractional polyphase steps in [`StudioSynth`]'s compile-time Lanczos-3 LUT.
const LANCZOS_PHASES: usize = 1024;

const fn sin_const(mut x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    const TWO_PI: f64 = core::f64::consts::TAU;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
    let k = (x / TWO_PI) as i64;
    x -= (k as f64) * TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }
    if x > HALF_PI {
        x = PI - x;
    } else if x < -HALF_PI {
        x = -PI - x;
    }
    let r2 = x * x;
    x * (1.0
        - r2 * (1.0 / 6.0
            - r2 * (1.0 / 120.0
                - r2 * (1.0 / 5_040.0
                    - r2 * (1.0 / 362_880.0
                        - r2 * (1.0 / 39_916_800.0 - r2 / 6_227_020_800.0))))))
}

const fn cos_const(x: f64) -> f64 {
    sin_const(x + core::f64::consts::FRAC_PI_2)
}

const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (sin_const(pi_x) / pi_x) * (sin_const(pi_x_a) / pi_x_a)
}

const fn build_lanczos3_lut() -> [[f32; 6]; LANCZOS_PHASES] {
    let mut table = [[0.0f32; 6]; LANCZOS_PHASES];
    let mut p = 0usize;
    while p < LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut sum = 0.0f64;
        let mut w = [0.0f64; 6];
        let mut k = 0usize;
        while k < 6 {
            let offset = (k as i32) - 2; // -2, -1, 0, +1, +2, +3
            let val = lanczos_kernel(frac - (offset as f64), 3.0);
            w[k] = val;
            sum += val;
            k += 1;
        }
        let inv_sum = if sum != 0.0 { 1.0 / sum } else { 1.0 };
        let mut j = 0usize;
        while j < 6 {
            table[p][j] = (w[j] * inv_sum) as f32;
            j += 1;
        }
        p += 1;
    }
    table
}

static LANCZOS3_LUT: [[f32; 6]; LANCZOS_PHASES] = build_lanczos3_lut();

/// High-precision 48,000 Hz `f32` studio synthesizer with band-limited Lanczos windowed-sinc
/// resampling and true equal-power stereo panning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudioSynth {
    /// Target output sample rate in Hz (default: `48_000.0`).
    pub output_rate_hz: f64,
    /// Lanczos kernel half-width `a` (default: `3`, which uses the compile-time 1,024-phase LUT).
    pub lanczos_a: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self {
            output_rate_hz: 48_000.0,
            lanczos_a: 3,
        }
    }
}

impl StudioSynth {
    /// Create a new [`StudioSynth`] configured for `output_rate_hz` with Lanczos-3 resampling.
    #[inline]
    #[must_use]
    pub const fn new(output_rate_hz: f64) -> Self {
        Self {
            output_rate_hz,
            lanczos_a: 3,
        }
    }

    /// Compute true equal-power stereo panning gains `(gain_l, gain_r)` for `pan in -1.0..=1.0`.
    #[inline]
    #[must_use]
    pub fn equal_power_pan(pan: f32) -> (f32, f32) {
        let clamped = pan.clamp(-1.0, 1.0) as f64;
        let angle = (clamped + 1.0) * core::f64::consts::FRAC_PI_4;
        (cos_const(angle) as f32, sin_const(angle) as f32)
    }

    /// Render a single [`KaedeSampleView`] voice additively into `out_l` and `out_r` at
    /// `self.output_rate_hz` using band-limited Lanczos interpolation and equal-power panning.
    ///
    /// Returns `true` if the voice is still active after rendering, or `false` if a one-shot
    /// sample reached its end.
    #[allow(clippy::too_many_arguments)]
    pub fn render_voice_f32(
        &self,
        sample: &KaedeSampleView<'_>,
        pos: &mut f64,
        playback_rate_hz: f64,
        gain: f32,
        pan: f32,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> bool {
        let frames = out_l.len().min(out_r.len());
        if frames == 0 || sample.is_empty() || self.output_rate_hz <= 0.0 || playback_rate_hz <= 0.0
        {
            return false;
        }

        let (pan_l, pan_r) = Self::equal_power_pan(pan);
        let scale_l = (gain * pan_l) * (1.0 / 128.0);
        let scale_r = (gain * pan_r) * (1.0 / 128.0);
        let step = playback_rate_hz / self.output_rate_hz;
        let pcm = sample.pcm_bytes;
        let pcm_len = pcm.len();
        let (loop_start, loop_end) = match sample.loop_range {
            Some((ls, le)) if (ls as usize) < (le as usize) && (le as usize) <= pcm_len => {
                (Some(ls as usize), le as usize)
            }
            _ => (None, pcm_len),
        };
        let loop_end_f = loop_end as f64;
        let loop_len_f = loop_start.map(|ls| (loop_end - ls) as f64);

        let mut p = *pos;
        let use_lut = self.lanczos_a == 3;
        let a_i32 = self.lanczos_a.max(1) as i32;

        for i in 0..frames {
            if p >= loop_end_f {
                if let Some(ll) = loop_len_f {
                    while p >= loop_end_f {
                        p -= ll;
                    }
                } else {
                    *pos = p;
                    return false;
                }
            }

            let idx = p as isize;
            let frac = p - (idx as f64);

            let val = if use_lut {
                let phase = ((frac * (LANCZOS_PHASES as f64)) as usize).min(LANCZOS_PHASES - 1);
                let weights = &LANCZOS3_LUT[phase];
                let mut acc = 0.0f32;
                for (k, &w) in weights.iter().enumerate() {
                    let s_idx = idx + (k as isize) - 2;
                    let s = fetch_sample_wrapped(pcm, s_idx, loop_start, loop_end);
                    acc += s * w;
                }
                acc
            } else {
                let mut acc = 0.0f64;
                let mut wsum = 0.0f64;
                let mut k = -a_i32 + 1;
                while k <= a_i32 {
                    let w = lanczos_kernel(frac - (k as f64), a_i32 as f64);
                    let s = fetch_sample_wrapped(pcm, idx + (k as isize), loop_start, loop_end);
                    acc += (s as f64) * w;
                    wsum += w;
                    k += 1;
                }
                if wsum != 0.0 {
                    (acc / wsum) as f32
                } else {
                    0.0
                }
            };

            out_l[i] += val * scale_l;
            out_r[i] += val * scale_r;
            p += step;
        }

        *pos = p;
        true
    }
}

#[inline(always)]
fn fetch_sample_wrapped(
    pcm: &[u8],
    mut idx: isize,
    loop_start: Option<usize>,
    loop_end: usize,
) -> f32 {
    if idx < 0 {
        return 0.0;
    }
    if (idx as usize) >= loop_end {
        if let Some(ls) = loop_start {
            let ll = (loop_end - ls) as isize;
            idx = (ls as isize) + ((idx - (ls as isize)) % ll);
        } else {
            return 0.0;
        }
    }
    (pcm[idx as usize] as i8) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_synth_renders_finite_stereo_audio() {
        let pcm: [u8; 64] = core::array::from_fn(|i| (((i * 19) % 256) as i32 - 128) as i8 as u8);
        let view = KaedeSampleView::from_pcm_bytes(0, 13_379, 60, Some((8, 56)), &pcm);
        let studio = StudioSynth::default();
        let mut l = [0.0f32; 128];
        let mut r = [0.0f32; 128];
        let mut pos = 0.0;
        assert!(studio.render_voice_f32(&view, &mut pos, 13_379.0, 0.9, -0.3, &mut l, &mut r));
        assert!(l.iter().all(|v| v.is_finite()));
        assert!(r.iter().all(|v| v.is_finite()));
        assert!(l.iter().any(|&v| v.abs() > 1e-4));

        // Verify general Lanczos-a fallback (a = 2)
        let studio_a2 = StudioSynth {
            output_rate_hz: 48_000.0,
            lanczos_a: 2,
        };
        let mut l2 = [0.0f32; 64];
        let mut r2 = [0.0f32; 64];
        let mut pos2 = 0.0;
        assert!(studio_a2.render_voice_f32(&view, &mut pos2, 13_379.0, 0.9, 0.2, &mut l2, &mut r2));
        assert!(l2.iter().all(|v| v.is_finite()));
    }
}
