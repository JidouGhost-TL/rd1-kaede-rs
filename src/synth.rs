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

/// Branchless, auto-vectorizable equivalent of `KAEDE_CLAMP_LUT_512[((acc as u32) >> 7) & 0x1FF]`.
///
/// Sign-extends bits `7..=15` of `acc` into `[-256, +255]`, saturates to `[-128, +127]`,
/// and reproduces the authentic `lut[0x1FF] = 0` (`s == -1 -> 0`) quirk.
#[inline(always)]
const fn clamp_dma8_v1(acc: i32) -> i8 {
    let s = (acc << 16) >> 23;
    let clamped = if s > 127 {
        127
    } else if s < -128 {
        -128
    } else {
        s
    };
    if s == -1 {
        0
    } else {
        clamped as i8
    }
}

/// Branchless, auto-vectorizable equivalent of `KAEDE_CLAMP_LUT_1024[((acc as u32) >> 7) & 0x3FF]`.
///
/// Sign-extends bits `7..=16` of `acc` into `[-512, +511]`, saturates to `[-128, +127]`,
/// and reproduces the authentic `lut[0x3FF] = 0` (`s == -1 -> 0`) quirk.
#[inline(always)]
const fn clamp_dma8_v2(acc: i32) -> i8 {
    let s = (acc << 15) >> 22;
    let clamped = if s > 127 {
        127
    } else if s < -128 {
        -128
    } else {
        s
    };
    if s == -1 {
        0
    } else {
        clamped as i8
    }
}

// ============================================================================
// AccuracySynth: Bit-Exact ARM7TDMI Software Mixer Emulation
// ============================================================================

/// Bit-exact software synthesizer implementing all 10 ARM7TDMI mixer kernels of the
/// Nintendo R&D1 "Kaede" DirectSound engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
        let max_safe_fp = if pcm_len > 1 {
            ((pcm_len - 1).min(0x3_FFFF) as u32) << 14
        } else {
            0
        };
        let can_interior = step < (1 << 30) && end_fp <= max_safe_fp;

        let mut fp = voice.phase_q14;
        let mut block_idx = 0usize;

        while block_idx < blocks {
            if can_interior && fp < end_fp {
                let run_blocks = if step4 == 0 {
                    blocks - block_idx
                } else {
                    (((end_fp - 1 - fp) as u64) / (step4 as u64)) as usize
                }
                .min(blocks - block_idx);

                if run_blocks > 0 {
                    let slice = &mut acc[block_idx * 8..(block_idx + run_blocks) * 8];
                    for chunk in slice.as_chunks_mut::<8>().0 {
                        let fp0 = fp;
                        let fp1 = fp0.wrapping_add(step);
                        let fp2 = fp1.wrapping_add(step);
                        let fp3 = fp2.wrapping_add(step);
                        fp = fp0.wrapping_add(step4);

                        let i0 = (fp0 >> 14) as usize;
                        let s0b = pcm_bytes[i0 + 1] as i8 as i32;
                        let s0a = pcm_bytes[i0] as i8 as i32;
                        let v0 = s0a + (((s0b - s0a) * ((fp0 & Q14_INTERP_MASK) as i32)) >> 14);

                        let i1 = (fp1 >> 14) as usize;
                        let s1b = pcm_bytes[i1 + 1] as i8 as i32;
                        let s1a = pcm_bytes[i1] as i8 as i32;
                        let v1 = s1a + (((s1b - s1a) * ((fp1 & Q14_INTERP_MASK) as i32)) >> 14);

                        let i2 = (fp2 >> 14) as usize;
                        let s2b = pcm_bytes[i2 + 1] as i8 as i32;
                        let s2a = pcm_bytes[i2] as i8 as i32;
                        let v2 = s2a + (((s2b - s2a) * ((fp2 & Q14_INTERP_MASK) as i32)) >> 14);

                        let i3 = (fp3 >> 14) as usize;
                        let s3b = pcm_bytes[i3 + 1] as i8 as i32;
                        let s3a = pcm_bytes[i3] as i8 as i32;
                        let v3 = s3a + (((s3b - s3a) * ((fp3 & Q14_INTERP_MASK) as i32)) >> 14);

                        chunk[0] = chunk[0].wrapping_add(v0.wrapping_mul(vol_l));
                        chunk[1] = chunk[1].wrapping_add(v0.wrapping_mul(vol_r));
                        chunk[2] = chunk[2].wrapping_add(v1.wrapping_mul(vol_l));
                        chunk[3] = chunk[3].wrapping_add(v1.wrapping_mul(vol_r));
                        chunk[4] = chunk[4].wrapping_add(v2.wrapping_mul(vol_l));
                        chunk[5] = chunk[5].wrapping_add(v2.wrapping_mul(vol_r));
                        chunk[6] = chunk[6].wrapping_add(v3.wrapping_mul(vol_l));
                        chunk[7] = chunk[7].wrapping_add(v3.wrapping_mul(vol_r));
                    }
                    block_idx += run_blocks;
                    continue;
                }
            }

            if fp.wrapping_add(step4) < end_fp {
                let block = &mut acc[block_idx * 8..(block_idx + 1) * 8];
                for pair in block.as_chunks_mut::<2>().0 {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    pair[0] = pair[0].wrapping_add(s.wrapping_mul(vol_l));
                    pair[1] = pair[1].wrapping_add(s.wrapping_mul(vol_r));
                    fp = fp.wrapping_add(step);
                }
                block_idx += 1;
            } else if loop_delta != 0 {
                let block = &mut acc[block_idx * 8..(block_idx + 1) * 8];
                for pair in block.as_chunks_mut::<2>().0 {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    pair[0] = pair[0].wrapping_add(s.wrapping_mul(vol_l));
                    pair[1] = pair[1].wrapping_add(s.wrapping_mul(vol_r));
                    fp = fp.wrapping_add(step);
                    fp = wrap_loop_q14(fp, end_fp, loop_delta);
                }
                block_idx += 1;
            } else {
                let mut out_idx = block_idx * 8;
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
        let max_safe_fp = if pcm_len > 1 {
            ((pcm_len - 1).min(0x3_FFFF) as u32) << 14
        } else {
            0
        };
        let can_interior = step < (1 << 30) && end_fp <= max_safe_fp;

        let mut fp = voice.phase_q14;
        let mut block_idx = 0usize;

        while block_idx < blocks {
            if can_interior && fp < end_fp {
                let run_blocks = if step4 == 0 {
                    blocks - block_idx
                } else {
                    (((end_fp - 1 - fp) as u64) / (step4 as u64)) as usize
                }
                .min(blocks - block_idx);

                if run_blocks > 0 {
                    let slice = &mut acc[block_idx * 4..(block_idx + run_blocks) * 4];
                    for chunk in slice.as_chunks_mut::<4>().0 {
                        let fp0 = fp;
                        let fp1 = fp0.wrapping_add(step);
                        let fp2 = fp1.wrapping_add(step);
                        let fp3 = fp2.wrapping_add(step);
                        fp = fp0.wrapping_add(step4);

                        let i0 = (fp0 >> 14) as usize;
                        let s0b = pcm_bytes[i0 + 1] as i8 as i32;
                        let s0a = pcm_bytes[i0] as i8 as i32;
                        let v0 = s0a + (((s0b - s0a) * ((fp0 & Q14_INTERP_MASK) as i32)) >> 14);

                        let i1 = (fp1 >> 14) as usize;
                        let s1b = pcm_bytes[i1 + 1] as i8 as i32;
                        let s1a = pcm_bytes[i1] as i8 as i32;
                        let v1 = s1a + (((s1b - s1a) * ((fp1 & Q14_INTERP_MASK) as i32)) >> 14);

                        let i2 = (fp2 >> 14) as usize;
                        let s2b = pcm_bytes[i2 + 1] as i8 as i32;
                        let s2a = pcm_bytes[i2] as i8 as i32;
                        let v2 = s2a + (((s2b - s2a) * ((fp2 & Q14_INTERP_MASK) as i32)) >> 14);

                        let i3 = (fp3 >> 14) as usize;
                        let s3b = pcm_bytes[i3 + 1] as i8 as i32;
                        let s3a = pcm_bytes[i3] as i8 as i32;
                        let v3 = s3a + (((s3b - s3a) * ((fp3 & Q14_INTERP_MASK) as i32)) >> 14);

                        chunk[0] = chunk[0].wrapping_add(v0.wrapping_mul(vol));
                        chunk[1] = chunk[1].wrapping_add(v1.wrapping_mul(vol));
                        chunk[2] = chunk[2].wrapping_add(v2.wrapping_mul(vol));
                        chunk[3] = chunk[3].wrapping_add(v3.wrapping_mul(vol));
                    }
                    block_idx += run_blocks;
                    continue;
                }
            }

            if fp.wrapping_add(step4) < end_fp {
                for out in &mut acc[block_idx * 4..(block_idx + 1) * 4] {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    *out = out.wrapping_add(s.wrapping_mul(vol));
                    fp = fp.wrapping_add(step);
                }
                block_idx += 1;
            } else if loop_delta != 0 {
                for out in &mut acc[block_idx * 4..(block_idx + 1) * 4] {
                    let s = interp_q14_sample(pcm_bytes, pcm_len, fp);
                    *out = out.wrapping_add(s.wrapping_mul(vol));
                    fp = fp.wrapping_add(step);
                    fp = wrap_loop_q14(fp, end_fp, loop_delta);
                }
                block_idx += 1;
            } else {
                let mut out_idx = block_idx * 4;
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
        let mut rem_blocks = blocks;

        while rem_blocks > 0 {
            let run_blocks = ((len_l - pos_l) / 4)
                .min((len_r - pos_r) / 4)
                .min(rem_blocks);
            if run_blocks > 0 {
                let run_frames = run_blocks * 4;
                let dl = &delay_l[pos_l..pos_l + run_frames];
                let dr = &delay_r[pos_r..pos_r + run_frames];
                let out = &mut out_acc[out_idx..out_idx + run_frames * 2];
                for ((&bl, &br), pair) in dl.iter().zip(dr).zip(out.as_chunks_mut::<2>().0) {
                    let sl = bl as i8 as i32;
                    let sr = br as i8 as i32;
                    // Group L+R HP shifts together to halve `cl` shift-count reloads on x86_64
                    hp_l = hp_l.wrapping_sub(hp_l >> hp_sh).wrapping_add(sl);
                    let hpo_l = sl.wrapping_sub(hp_l >> hp_sh);
                    hp_r = hp_r.wrapping_sub(hp_r >> hp_sh).wrapping_add(sr);
                    let hpo_r = sr.wrapping_sub(hp_r >> hp_sh);

                    lp_l = lp_l.wrapping_sub(lp_l >> lp_sh).wrapping_add(hpo_l);
                    lp_r = lp_r.wrapping_sub(lp_r >> lp_sh).wrapping_add(hpo_r);
                    pair[0] = lp_l.wrapping_mul(vol);
                    pair[1] = lp_r.wrapping_mul(vol);
                }
                pos_l += run_frames;
                pos_r += run_frames;
                if pos_l >= len_l {
                    pos_l = 0;
                }
                if pos_r >= len_r {
                    pos_r = 0;
                }
                out_idx += run_frames * 2;
                rem_blocks -= run_blocks;
            } else {
                let block = &mut out_acc[out_idx..out_idx + 8];
                for pair in block.as_chunks_mut::<2>().0 {
                    let sl = delay_l[pos_l % len_l] as i8 as i32;
                    pos_l += 1;
                    hp_l = hp_l.wrapping_sub(hp_l >> hp_sh).wrapping_add(sl);
                    let hpo_l = sl.wrapping_sub(hp_l >> hp_sh);
                    lp_l = lp_l.wrapping_sub(lp_l >> lp_sh).wrapping_add(hpo_l);
                    pair[0] = lp_l.wrapping_mul(vol);

                    let sr = delay_r[pos_r % len_r] as i8 as i32;
                    pos_r += 1;
                    hp_r = hp_r.wrapping_sub(hp_r >> hp_sh).wrapping_add(sr);
                    let hpo_r = sr.wrapping_sub(hp_r >> hp_sh);
                    lp_r = lp_r.wrapping_sub(lp_r >> lp_sh).wrapping_add(hpo_r);
                    pair[1] = lp_r.wrapping_mul(vol);
                }
                if pos_l >= len_l {
                    pos_l = 0;
                }
                if pos_r >= len_r {
                    pos_r = 0;
                }
                out_idx += 8;
                rem_blocks -= 1;
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
        let mut rem_blocks = blocks;

        while rem_blocks > 0 {
            let run_blocks = ((len - pos) / 4).min(rem_blocks);
            if run_blocks > 0 {
                let run_frames = run_blocks * 4;
                for (&b, out) in delay_buf[pos..pos + run_frames]
                    .iter()
                    .zip(&mut out_acc[out_idx..out_idx + run_frames])
                {
                    let s = b as i8 as i32;
                    hp = hp.wrapping_sub(hp >> hp_sh).wrapping_add(s);
                    let hpo = s.wrapping_sub(hp >> hp_sh);
                    lp = lp.wrapping_sub(lp >> lp_sh).wrapping_add(hpo);
                    *out = lp.wrapping_mul(vol);
                }
                pos += run_frames;
                if pos >= len {
                    pos %= len;
                }
                out_idx += run_frames;
                rem_blocks -= run_blocks;
            } else {
                for out in &mut out_acc[out_idx..out_idx + 4] {
                    let s = delay_buf[pos % len] as i8 as i32;
                    pos += 1;
                    hp = hp.wrapping_sub(hp >> hp_sh).wrapping_add(s);
                    let hpo = s.wrapping_sub(hp >> hp_sh);
                    lp = lp.wrapping_sub(lp >> lp_sh).wrapping_add(hpo);
                    *out = lp.wrapping_mul(vol);
                }
                if pos >= len {
                    pos %= len;
                }
                out_idx += 4;
                rem_blocks -= 1;
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
        let acc = &acc_interleaved[..frames * 2];
        let dl = &mut dma_l[..frames];
        let dr = &mut dma_r[..frames];
        match version {
            KaedeVersion::V1Lut512 => {
                for ((pair, out_l), out_r) in acc.as_chunks::<2>().0.iter().zip(dl).zip(dr) {
                    *out_l = clamp_dma8_v1(pair[0]);
                    *out_r = clamp_dma8_v1(pair[1]);
                }
            }
            KaedeVersion::V2Lut1024 => {
                for ((pair, out_l), out_r) in acc.as_chunks::<2>().0.iter().zip(dl).zip(dr) {
                    *out_l = clamp_dma8_v2(pair[0]);
                    *out_r = clamp_dma8_v2(pair[1]);
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
                for (&a, out) in acc_mono[..frames].iter().zip(&mut dma_out[..frames]) {
                    *out = clamp_dma8_v1(a);
                }
            }
            KaedeVersion::V2Lut1024 => {
                for (&a, out) in acc_mono[..frames].iter().zip(&mut dma_out[..frames]) {
                    *out = clamp_dma8_v2(a);
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
            let dst = &mut acc[out_idx..out_idx + run * 2];
            for (&b, pair) in pcm_bytes[pos..pos + run]
                .iter()
                .zip(dst.as_chunks_mut::<2>().0)
            {
                let s = b as i8 as i32;
                pair[0] = pair[0].wrapping_add(s.wrapping_mul(vol_l));
                pair[1] = pair[1].wrapping_add(s.wrapping_mul(vol_r));
            }
            out_idx += run * 2;
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
        let max_safe_fp = (pcm_len.min(0x3_FFFF) as u32) << 14;
        let can_interior = step < (1 << 30) && end_fp <= max_safe_fp;

        let mut fp = voice.phase_q14;
        let mut block_idx = 0usize;

        while block_idx < blocks {
            if can_interior && fp < end_fp {
                let run_blocks = if step4 == 0 {
                    blocks - block_idx
                } else {
                    (((end_fp - 1 - fp) as u64) / (step4 as u64)) as usize
                }
                .min(blocks - block_idx);

                if run_blocks > 0 {
                    let slice = &mut acc[block_idx * 8..(block_idx + run_blocks) * 8];
                    for pair in slice.as_chunks_mut::<2>().0 {
                        let s = pcm_bytes[(fp >> 14) as usize] as i8 as i32;
                        pair[0] = pair[0].wrapping_add(s.wrapping_mul(vol_l));
                        pair[1] = pair[1].wrapping_add(s.wrapping_mul(vol_r));
                        fp = fp.wrapping_add(step);
                    }
                    block_idx += run_blocks;
                    continue;
                }
            }

            if fp.wrapping_add(step4) < end_fp {
                let block = &mut acc[block_idx * 8..(block_idx + 1) * 8];
                for pair in block.as_chunks_mut::<2>().0 {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    pair[0] = pair[0].wrapping_add(s.wrapping_mul(vol_l));
                    pair[1] = pair[1].wrapping_add(s.wrapping_mul(vol_r));
                    fp = fp.wrapping_add(step);
                }
                block_idx += 1;
            } else if loop_delta != 0 {
                let block = &mut acc[block_idx * 8..(block_idx + 1) * 8];
                for pair in block.as_chunks_mut::<2>().0 {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    pair[0] = pair[0].wrapping_add(s.wrapping_mul(vol_l));
                    pair[1] = pair[1].wrapping_add(s.wrapping_mul(vol_r));
                    fp = fp.wrapping_add(step);
                    fp = wrap_loop_q14(fp, end_fp, loop_delta);
                }
                block_idx += 1;
            } else {
                let mut out_idx = block_idx * 8;
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
        let max_safe_fp = (pcm_len.min(0x3_FFFF) as u32) << 14;
        let can_interior = step < (1 << 30) && end_fp <= max_safe_fp;

        let mut fp = voice.phase_q14;
        let mut block_idx = 0usize;

        while block_idx < blocks {
            if can_interior && fp < end_fp {
                let run_blocks = if step4 == 0 {
                    blocks - block_idx
                } else {
                    (((end_fp - 1 - fp) as u64) / (step4 as u64)) as usize
                }
                .min(blocks - block_idx);

                if run_blocks > 0 {
                    let slice = &mut acc[block_idx * 4..(block_idx + run_blocks) * 4];
                    for chunk in slice.as_chunks_mut::<4>().0 {
                        let mut r0 = chunk[0];
                        let mut r2 = chunk[2];

                        let s0 = pcm_bytes[(fp >> 14) as usize] as i8 as i32;
                        r0 = r0.wrapping_add(s0.wrapping_mul(vol));
                        fp = fp.wrapping_add(step);

                        let s1 = pcm_bytes[(fp >> 14) as usize] as i8 as i32;
                        // Authentic ARM7TDMI instruction at +0xB58: `mla r1, r4, sb, r2`
                        let r1 = r2.wrapping_add(s1.wrapping_mul(vol));
                        fp = fp.wrapping_add(step);

                        let s2 = pcm_bytes[(fp >> 14) as usize] as i8 as i32;
                        r2 = r2.wrapping_add(s2.wrapping_mul(vol));
                        fp = fp.wrapping_add(step);

                        let s3 = pcm_bytes[(fp >> 14) as usize] as i8 as i32;
                        // Authentic ARM7TDMI instruction at +0xB78: `mla r3, r4, sb, r2`
                        let r3 = r2.wrapping_add(s3.wrapping_mul(vol));
                        fp = fp.wrapping_add(step);

                        chunk[0] = r0;
                        chunk[1] = r1;
                        chunk[2] = r2;
                        chunk[3] = r3;
                    }
                    block_idx += run_blocks;
                    continue;
                }
            }

            if fp.wrapping_add(step4) < end_fp {
                let chunk = &mut acc[block_idx * 4..(block_idx + 1) * 4];
                let mut r0 = chunk[0];
                let mut r2 = chunk[2];

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

                chunk[0] = r0;
                chunk[1] = r1;
                chunk[2] = r2;
                chunk[3] = r3;
                block_idx += 1;
            } else if loop_delta != 0 {
                for out in &mut acc[block_idx * 4..(block_idx + 1) * 4] {
                    let s = nearest_q14_sample(pcm_bytes, pcm_len, fp);
                    *out = out.wrapping_add(s.wrapping_mul(vol));
                    fp = fp.wrapping_add(step);
                    fp = wrap_loop_q14(fp, end_fp, loop_delta);
                }
                block_idx += 1;
            } else {
                let mut out_idx = block_idx * 4;
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
            for pair in acc_interleaved[..frames * 2].as_chunks_mut::<2>().0 {
                let xl = pair[0];
                let xr = pair[1];
                st_l = (beta.wrapping_mul(xl).wrapping_add(alpha.wrapping_mul(st_l))) >> 8;
                st_r = (beta.wrapping_mul(xr).wrapping_add(alpha.wrapping_mul(st_r))) >> 8;
                pair[0] = st_l;
                pair[1] = st_r;
            }
        } else {
            let alpha = ((state.cutoff - 0x80) as i32) << 1;
            let beta = 256 - alpha;
            for pair in acc_interleaved[..frames * 2].as_chunks_mut::<2>().0 {
                let xl = pair[0];
                let xr = pair[1];
                let lp_l = (beta.wrapping_mul(xl).wrapping_add(alpha.wrapping_mul(st_l))) >> 8;
                let lp_r = (beta.wrapping_mul(xr).wrapping_add(alpha.wrapping_mul(st_r))) >> 8;
                pair[0] = xl.wrapping_sub(lp_l);
                pair[1] = xr.wrapping_sub(lp_r);
                st_l = lp_l;
                st_r = lp_r;
            }
        }

        state.state_l = st_l;
        state.state_r = st_r;
    }
}

#[inline(always)]
fn wrap_loop_q14(mut fp: u32, end_fp: u32, loop_delta: i32) -> u32 {
    if fp >= end_fp {
        let loop_len = (-(loop_delta as i64)) as u32;
        fp = fp.wrapping_sub(loop_len);
        if fp >= end_fp && loop_len > 0 {
            let ls = end_fp.wrapping_sub(loop_len);
            fp = ls + ((fp - ls) % loop_len);
        }
    }
    fp
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

const fn build_lanczos3_lut() -> [[f32; 8]; LANCZOS_PHASES] {
    let mut table = [[0.0f32; 8]; LANCZOS_PHASES];
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

static LANCZOS3_LUT: [[f32; 8]; LANCZOS_PHASES] = build_lanczos3_lut();

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
        let clamped = if pan.is_finite() {
            pan.clamp(-1.0, 1.0) as f64
        } else {
            0.0
        };
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
        if frames == 0
            || sample.is_empty()
            || !self.output_rate_hz.is_finite()
            || self.output_rate_hz <= 0.0
            || !playback_rate_hz.is_finite()
            || playback_rate_hz <= 0.0
            || !pos.is_finite()
            || !gain.is_finite()
        {
            return false;
        }
        if *pos < 0.0 {
            *pos = 0.0;
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
        let loop_start_f = loop_start.unwrap_or(0) as f64;
        let loop_len_f = loop_start.map(|ls| (loop_end - ls) as f64);

        let mut p = *pos;

        if self.lanczos_a == 3 {
            const Q32_SCALE: f64 = 4_294_967_296.0;
            let safe_end_f = loop_end.saturating_sub(3) as f64;
            let step_fp = (step * Q32_SCALE) as u64;
            let mut i = 0usize;

            while i < frames {
                if p >= loop_end_f {
                    if let Some(ll) = loop_len_f {
                        p -= ll;
                        if p >= loop_end_f {
                            p = loop_start_f + ((p - loop_start_f) % ll);
                        }
                        if p < loop_start_f {
                            p = loop_start_f;
                        }
                    } else {
                        *pos = p;
                        return false;
                    }
                }

                // Interior fast path: 32.32 fixed-point phase + zero boundary/wrap checks.
                if loop_end >= 6 && p >= 2.0 && p < safe_end_f && step_fp > 0 {
                    let rem = frames - i;
                    let max_n = ((safe_end_f - p) / step) as usize;
                    let mut n = max_n.min(rem);
                    let fp = (p * Q32_SCALE) as u64;
                    while n > 0 {
                        let last_fp = fp.saturating_add(((n - 1) as u64).saturating_mul(step_fp));
                        if ((last_fp >> 32) as usize) + 3 < loop_end {
                            break;
                        }
                        n -= 1;
                    }
                    if n > 0 && (fp >> 32) >= 2 {
                        let last_idx = ((fp.wrapping_add(((n - 1) as u64).wrapping_mul(step_fp)))
                            >> 32) as usize;
                        let src = &pcm[..last_idx + 4];
                        let mut ph_shifted = fp - (2u64 << 32);
                        for (ol, or) in out_l[i..i + n].iter_mut().zip(&mut out_r[i..i + n]) {
                            let b0 = (ph_shifted >> 32) as usize;
                            let phase = ((ph_shifted >> 22) as usize) & (LANCZOS_PHASES - 1);
                            let w = &LANCZOS3_LUT[phase];
                            // Load highest index first so LLVM merges all 6 bounds checks into 1
                            let s5 = (src[b0 + 5] as i8) as f32;
                            let s0 = (src[b0] as i8) as f32;
                            let s1 = (src[b0 + 1] as i8) as f32;
                            let s2 = (src[b0 + 2] as i8) as f32;
                            let s3 = (src[b0 + 3] as i8) as f32;
                            let s4 = (src[b0 + 4] as i8) as f32;
                            let val = (s0 * w[0] + s1 * w[1])
                                + (s2 * w[2] + s3 * w[3])
                                + (s4 * w[4] + s5 * w[5]);
                            *ol += val * scale_l;
                            *or += val * scale_r;
                            ph_shifted = ph_shifted.wrapping_add(step_fp);
                        }
                        p += (n as f64) * step;
                        i += n;
                        continue;
                    }
                }

                // Boundary / wrap single-frame fallback
                let idx = p as isize;
                let frac = p - (idx as f64);
                let phase = ((frac * (LANCZOS_PHASES as f64)) as usize).min(LANCZOS_PHASES - 1);
                let w = &LANCZOS3_LUT[phase];
                let val = (fetch_sample_wrapped(pcm, idx - 2, loop_start, loop_end) * w[0]
                    + fetch_sample_wrapped(pcm, idx - 1, loop_start, loop_end) * w[1]
                    + fetch_sample_wrapped(pcm, idx, loop_start, loop_end) * w[2])
                    + (fetch_sample_wrapped(pcm, idx + 1, loop_start, loop_end) * w[3]
                        + fetch_sample_wrapped(pcm, idx + 2, loop_start, loop_end) * w[4]
                        + fetch_sample_wrapped(pcm, idx + 3, loop_start, loop_end) * w[5]);
                out_l[i] += val * scale_l;
                out_r[i] += val * scale_r;
                p += step;
                i += 1;
            }
        } else {
            let a_i32 = self.lanczos_a.max(1) as i32;
            let a_f64 = a_i32 as f64;
            for (ol, or) in out_l[..frames].iter_mut().zip(&mut out_r[..frames]) {
                if p >= loop_end_f {
                    if let Some(ll) = loop_len_f {
                        p -= ll;
                        if p >= loop_end_f {
                            p = loop_start_f + ((p - loop_start_f) % ll);
                        }
                        if p < loop_start_f {
                            p = loop_start_f;
                        }
                    } else {
                        *pos = p;
                        return false;
                    }
                }

                let idx = p as isize;
                let frac = p - (idx as f64);
                let mut acc = 0.0f64;
                let mut wsum = 0.0f64;
                let mut k = -a_i32 + 1;
                while k <= a_i32 {
                    let w = lanczos_kernel(frac - (k as f64), a_f64);
                    let s = fetch_sample_wrapped(pcm, idx + (k as isize), loop_start, loop_end);
                    acc += (s as f64) * w;
                    wsum += w;
                    k += 1;
                }
                let val = if wsum != 0.0 {
                    (acc / wsum) as f32
                } else {
                    0.0
                };
                *ol += val * scale_l;
                *or += val * scale_r;
                p += step;
            }
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
    fn branchless_dma8_clamps_match_luts_exhaustively() {
        // Test every single 9-bit / 10-bit LUT entry plus high-bit wraparound & low-bit noise
        for base in -2048i32..=2048 {
            for low in [0i32, 1, 63, 127] {
                let acc = (base << 7).wrapping_add(low);
                let idx_v1 = ((acc as u32) >> 7) as usize & 0x1FF;
                let idx_v2 = ((acc as u32) >> 7) as usize & 0x3FF;
                assert_eq!(clamp_dma8_v1(acc), KAEDE_CLAMP_LUT_512[idx_v1]);
                assert_eq!(clamp_dma8_v2(acc), KAEDE_CLAMP_LUT_1024[idx_v2]);
            }
        }
        // Extreme i32 boundary values
        for acc in [i32::MIN, i32::MAX, -1, -128, 0, 127, -129, 0x7FFF_FFFF] {
            let idx_v1 = ((acc as u32) >> 7) as usize & 0x1FF;
            let idx_v2 = ((acc as u32) >> 7) as usize & 0x3FF;
            assert_eq!(clamp_dma8_v1(acc), KAEDE_CLAMP_LUT_512[idx_v1]);
            assert_eq!(clamp_dma8_v2(acc), KAEDE_CLAMP_LUT_1024[idx_v2]);
        }
    }

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

    #[test]
    fn malformed_inputs_never_hang_or_panic() {
        let pcm = [32u8; 16];
        let view = KaedeSampleView::from_pcm_bytes(0, 13_379, 60, Some((0, 16)), &pcm);
        let studio = StudioSynth::default();
        let mut l = [0.0f32; 16];
        let mut r = [0.0f32; 16];

        // Non-finite inputs must bail out cleanly without hanging or poisoning buffers
        let mut pos_inf = f64::INFINITY;
        assert!(!studio.render_voice_f32(&view, &mut pos_inf, 13_379.0, 1.0, 0.0, &mut l, &mut r));
        let mut pos_nan = f64::NAN;
        assert!(!studio.render_voice_f32(&view, &mut pos_nan, 13_379.0, 1.0, 0.0, &mut l, &mut r));
        let mut pos_huge = 1e15;
        assert!(studio.render_voice_f32(&view, &mut pos_huge, 13_379.0, 1.0, f32::NAN, &mut l, &mut r));
        assert!(l.iter().all(|v| v.is_finite()));
        assert!(r.iter().all(|v| v.is_finite()));

        // Short (< 4 byte) mono reverb delay buffer must not underflow
        let mut rev = KaedeReverbState {
            hp_shift: 3,
            lp_shift: 2,
            reverb_vol: 16,
            ..Default::default()
        };
        let mut acc = [0i32; 16];
        AccuracySynth::apply_mono_reverb(&[10], &mut rev, &mut acc);

        // Huge starting phase in looped Q14 mixer must wrap in O(1)
        let mut voice = KaedeVoice::new(0xFFFF_0000, 16, Some(15), 0x4000, 64, 64);
        AccuracySynth::mix_stereo_linear(&pcm, &mut voice, &mut acc);
    }
}
