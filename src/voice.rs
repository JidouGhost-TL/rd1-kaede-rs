//! Voice, reverb, and post-mix filter state structures for Nintendo R&D1 Kaede.

/// Software mixing kernel mode supported by the Nintendo R&D1 Kaede ARM7TDMI mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum KaedeMixMode {
    /// Stereo 18.14 fixed-point linear interpolation (`fp & 0x3FC0` 8-bit quantized fraction).
    #[default]
    StereoLinear,
    /// Mono 18.14 fixed-point linear interpolation (`fp & 0x3FC0` 8-bit quantized fraction).
    MonoLinear,
    /// Stereo 1:1 native-rate direct copy (4-byte aligned word-load fast path).
    StereoDirectCopy,
    /// Stereo 18.14 fixed-point nearest-neighbor resampling.
    StereoNearest,
    /// Mono 18.14 fixed-point nearest-neighbor resampling (including the authentic R&D1
    /// ARM fast-path `mla r1/r3, r4, sb, r2` accumulator quirk).
    MonoNearest,
}

/// Playback state for a single Nintendo R&D1 Kaede software mixer voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KaedeVoice {
    /// Current playback position in 18.14 fixed-point samples (`sample_index << 14 | frac`).
    pub phase_q14: u32,
    /// Playback phase increment per output frame in 18.14 fixed-point (`0x4000` = 1.0x).
    pub step_q14: u32,
    /// Sample end position in 18.14 fixed-point (`end_sample << 14`).
    pub end_q14: u32,
    /// Optional loop start position in 18.14 fixed-point (`loop_start << 14`).
    /// When `None`, the voice stops upon reaching `end_q14` (one-shot).
    pub loop_start_q14: Option<u32>,
    /// Left-channel signed integer gain (or mono gain for mono kernels).
    pub vol_l: i32,
    /// Right-channel signed integer gain (unused by mono kernels).
    pub vol_r: i32,
    /// Whether the voice is currently active.
    pub active: bool,
}

impl KaedeVoice {
    /// Construct a new active [`KaedeVoice`] using integer sample positions and a 18.14 step.
    #[inline]
    #[must_use]
    pub const fn new(
        start_phase_q14: u32,
        end_sample: usize,
        loop_start: Option<usize>,
        step_q14: u32,
        vol_l: i32,
        vol_r: i32,
    ) -> Self {
        Self {
            phase_q14: start_phase_q14,
            step_q14,
            end_q14: (end_sample as u32) << 14,
            loop_start_q14: match loop_start {
                Some(ls) => Some((ls as u32) << 14),
                None => None,
            },
            vol_l,
            vol_r,
            active: true,
        }
    }

    /// Construct a new active [`KaedeVoice`] configured for 1:1 direct-copy playback
    /// (`phase_q14 = start_pos << 14`, `step_q14 = 0x4000`).
    #[inline]
    #[must_use]
    pub const fn new_direct_copy(
        start_pos: usize,
        end_pos: usize,
        loop_start: Option<usize>,
        vol_l: i32,
        vol_r: i32,
    ) -> Self {
        Self::new(
            (start_pos as u32) << 14,
            end_pos,
            loop_start,
            1 << 14,
            vol_l,
            vol_r,
        )
    }

    /// Current integer sample position (`phase_q14 >> 14`).
    #[inline]
    #[must_use]
    pub const fn sample_pos(&self) -> usize {
        (self.phase_q14 >> 14) as usize
    }

    /// Signed negative loop delta in 18.14 fixed-point (`loop_start_q14 - end_q14`), or `0`
    /// if one-shot.
    #[inline]
    #[must_use]
    pub const fn loop_delta_q14(&self) -> i32 {
        match self.loop_start_q14 {
            Some(ls) if ls < self.end_q14 => (ls as i32).wrapping_sub(self.end_q14 as i32),
            _ => 0,
        }
    }
}

/// Filter and ring-buffer cursor state for Kaede's band-pass reverb / echo feedback delay tap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KaedeReverbState {
    /// Left (or mono) 1-pole high-pass / DC-blocker state accumulator.
    pub hp_l: i32,
    /// Left (or mono) 1-pole low-pass damping state accumulator.
    pub lp_l: i32,
    /// Right 1-pole high-pass / DC-blocker state accumulator.
    pub hp_r: i32,
    /// Right 1-pole low-pass damping state accumulator.
    pub lp_r: i32,
    /// High-pass shift exponent (`r2` in ARM Kernel 3/4, e.g., `3` or `4`).
    pub hp_shift: u32,
    /// Low-pass shift exponent (`r3` in ARM Kernel 3/4, e.g., `2`).
    pub lp_shift: u32,
    /// Reverb feedback gain multiplier (`r8` in ARM Kernel 3/4).
    pub reverb_vol: i32,
    /// Current read cursor within the left (or mono) delay ring buffer.
    pub pos_l: usize,
    /// Current read cursor within the right delay ring buffer.
    pub pos_r: usize,
}

/// State for Kaede's post-mix stereo 1-pole Low-Pass / High-Pass filter (ARM Kernel 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KaedeFilterState {
    /// Left channel filter memory (`r2` in ARM Kernel 10).
    pub state_l: i32,
    /// Right channel filter memory (`r3` in ARM Kernel 10).
    pub state_r: i32,
    /// Filter cutoff byte: `0x00..=0x7F` selects 1-pole LPF; `0x80..=0xFF` selects 1-pole HPF.
    pub cutoff: u8,
}

impl KaedeFilterState {
    /// Construct a new [`KaedeFilterState`] with the given initial memory and cutoff byte.
    #[inline]
    #[must_use]
    pub const fn new(state_l: i32, state_r: i32, cutoff: u8) -> Self {
        Self {
            state_l,
            state_r,
            cutoff,
        }
    }

    /// Returns `true` if `cutoff >= 0x80` (1-pole high-pass mode), or `false` (1-pole low-pass).
    #[inline]
    #[must_use]
    pub const fn is_high_pass(&self) -> bool {
        self.cutoff >= 0x80
    }
}
