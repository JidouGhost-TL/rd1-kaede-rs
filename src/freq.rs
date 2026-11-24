//! Fixed-point pitch, phase, and volume/panning helpers for Nintendo R&D1 Kaede.

/// GBA ARM7TDMI master clock frequency in Hz (`2^24` = 16,777,216 Hz).
pub const GBA_CLOCK_HZ: u32 = 16_777_216;

/// Standard default DirectSound mixing rate used across Nintendo R&D1 Kaede titles (Hz).
pub const KAEDE_DEFAULT_MIX_RATE_HZ: u32 = 13_379;

/// Unit step (`1.0x` playback speed) in Kaede's 18.14 fixed-point phase format (`1 << 14`).
pub const Q14_ONE: u32 = 1 << 14;

/// Fractional bitmask applied by Kaede's linear-interpolating ARM kernels (`fp & 0x3FC0`),
/// quantizing the 14-bit fractional phase to its top 8 bits before multiplying by `(s1 - s0)`.
pub const Q14_INTERP_MASK: u32 = 0x3FC0;

/// Compute the 18.14 fixed-point phase increment (`step_q14`) for playing a sample recorded at
/// `sample_rate_hz` through a mixer running at `mix_rate_hz`.
#[inline]
#[must_use]
pub const fn freq_to_step_q14(sample_rate_hz: u32, mix_rate_hz: u32) -> u32 {
    if mix_rate_hz == 0 {
        return 0;
    }
    (((sample_rate_hz as u64) << 14) / (mix_rate_hz as u64)) as u32
}

/// Convert a Kaede 18.14 fixed-point phase increment (`step_q14`) back to an effective sample
/// playback rate in Hz for a given `mix_rate_hz`.
#[inline]
#[must_use]
pub fn step_q14_to_freq(step_q14: u32, mix_rate_hz: u32) -> f64 {
    (step_q14 as f64) * (mix_rate_hz as f64) / (Q14_ONE as f64)
}

/// 12-TET semitone frequency ratios for `0..=11` semitones above an octave, scaled by `2^16`.
const SEMITONE_RATIOS_Q16: [u32; 12] = [
    65_536, 69_433, 73_562, 77_936, 82_570, 87_480, 92_682, 98_193, 104_032, 110_218, 116_772,
    123_715,
];

/// Compute the 18.14 fixed-point phase increment (`step_q14`) for playing a sample at a target
/// MIDI `note` relative to its `root_note`, native `sample_rate_hz`, and `mix_rate_hz`.
#[must_use]
pub const fn midi_note_step_q14(
    note: u8,
    root_note: u8,
    sample_rate_hz: u32,
    mix_rate_hz: u32,
) -> u32 {
    if mix_rate_hz == 0 {
        return 0;
    }
    let diff = (note as i32) - (root_note as i32);
    let octave = diff.div_euclid(12);
    let semi = diff.rem_euclid(12) as usize;

    let base = (((sample_rate_hz as u64) << 14) * (SEMITONE_RATIOS_Q16[semi] as u64))
        / ((mix_rate_hz as u64) << 16);

    if octave >= 0 {
        let shift = octave as u32;
        if shift >= 32 {
            u32::MAX
        } else {
            (base << shift) as u32
        }
    } else {
        let shift = (-octave) as u32;
        if shift >= 64 {
            0
        } else {
            (base >> shift) as u32
        }
    }
}

/// Compute the exact left and right integer mixer gains `(vol_l, vol_r)` matching the Kaede
/// Thumb voice wrapper (`adds r0, r0, r4; muls r2, r0, r2; asrs r2, #7`).
#[inline]
#[must_use]
pub const fn volume_pan_gains(
    voice_vol: u8,
    master_vol: u16,
    pan_l: i8,
    pan_r: i8,
) -> (i32, i32) {
    let sum = (voice_vol as i32) + (master_vol as i32);
    let vol_l = (sum * (pan_l as i32)) >> 7;
    let vol_r = (sum * (pan_r as i32)) >> 7;
    (vol_l, vol_r)
}

/// Compute the exact DirectSound hardware sample rate in Hz for a GBA Timer0 reload value.
#[inline]
#[must_use]
pub const fn rate_from_timer_reload(reload: u16) -> u32 {
    let ticks = 65_536u32 - (reload as u32);
    (GBA_CLOCK_HZ + (ticks / 2)) / ticks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_and_midi_pitch_helpers_work() {
        assert_eq!(freq_to_step_q14(13_379, 13_379), Q14_ONE);
        assert_eq!(freq_to_step_q14(26_758, 13_379), Q14_ONE * 2);
        assert_eq!(midi_note_step_q14(60, 60, 13_379, 13_379), Q14_ONE);
        assert_eq!(midi_note_step_q14(72, 60, 13_379, 13_379), Q14_ONE * 2);
        assert_eq!(midi_note_step_q14(48, 60, 13_379, 13_379), Q14_ONE / 2);
        assert_eq!(volume_pan_gains(128, 0, 127, 64), (127, 64));
        assert_eq!(rate_from_timer_reload(0xFB1A), 13_379);
    }
}
