//! Nintendo R&D1 "Kaede" engine revision detection and mixer variant metadata.
//!
//! A full-corpus audit of all 3,236 unique GBA ROMs identifies **12 commercial Nintendo R&D1
//! releases** powered by the Kaede ARM7TDMI software mixer across **2 distinct hardware
//! revisions**:
//!
//! 1. [`KaedeVersion::V1Lut512`] (**9 ROMs** — *WarioWare, Inc.: Mega Microgame$!* /
//!    *Made in Wario* / *Waliou Zhizao*: `AZWC`, `AZWE` ×3, `AZWJ` ×3, `AZWP` ×2):
//!    - 3,476-byte (`0xD94`) IWRAM ARM7TDMI mixer with 10 kernels (stereo/mono 14-bit linear
//!      with 8-bit quantized fraction `fp & 0x3FC0`, stereo 1:1 direct-copy, stereo/mono
//!      nearest-neighbor, stereo/mono band-pass feedback reverb, stereo 1-pole LPF/HPF, and
//!      stereo/mono LUT-clamped DMA8 output).
//!    - **512-byte output clamping LUT** (`mask = 0x1FF`, headroom `[-256, +255]` before
//!      modulo-512 wraparound, with `lut[0x1FF] = 0` mapping `-1` to `0`).
//! 2. [`KaedeVersion::V2Lut1024`] (**3 ROMs** — *Rhythm Tengoku* Japan Rev 0, Rev 1, and
//!    Kiosk Demo: `BRIJ` r0, `BRIJ` r1, `BZLJ`):
//!    - Identical 3,476-byte (`0xD94`) ARM7TDMI mixer kernels, upgraded to a **1,024-byte
//!      output clamping LUT** (`mask = 0x3FF`, headroom `[-512, +511]` — double the
//!      pre-wraparound headroom of V1 — with `lut[0x3FF] = 0`).

/// Distinct Nintendo R&D1 "Kaede" ARM7TDMI mixer revisions discovered across the GBA corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum KaedeVersion {
    /// Kaede Rev 1 (9 ROMs: *WarioWare, Inc.* / *Made in Wario* / *Waliou Zhizao*):
    /// 512-byte DirectSound output clamping LUT (`mask = 0x1FF`, headroom `[-256, +255]`).
    #[default]
    V1Lut512,
    /// Kaede Rev 2 (3 ROMs: *Rhythm Tengoku* Rev 0 / Rev 1 / Kiosk Demo):
    /// 1,024-byte DirectSound output clamping LUT (`mask = 0x3FF`, headroom `[-512, +511]`).
    V2Lut1024,
}

impl KaedeVersion {
    /// Size of the DirectSound DMA8 saturation lookup table in bytes (`512` for V1, `1024` for V2).
    #[inline]
    #[must_use]
    pub const fn lut_size(self) -> usize {
        match self {
            Self::V1Lut512 => 512,
            Self::V2Lut1024 => 1024,
        }
    }

    /// Bitmask applied to `(acc >> 7)` before indexing the DirectSound DMA8 saturation LUT
    /// (`0x1FF` for V1, `0x3FF` for V2).
    #[inline]
    #[must_use]
    pub const fn lut_mask(self) -> u32 {
        match self {
            Self::V1Lut512 => 0x01FF,
            Self::V2Lut1024 => 0x03FF,
        }
    }

    /// Inclusive `(min, max)` range of `(acc >> 7)` values that clamp cleanly to `[-128, +127]`
    /// before modulo wraparound occurs in the saturation LUT.
    #[inline]
    #[must_use]
    pub const fn clamp_headroom(self) -> (i32, i32) {
        match self {
            Self::V1Lut512 => (-256, 255),
            Self::V2Lut1024 => (-512, 511),
        }
    }

    /// Detect the exact Nintendo R&D1 Kaede ARM7TDMI mixer revision from a GBA ROM or IWRAM
    /// image.
    ///
    /// Matches the distinctive 16-byte ARM7TDMI post-mix 1-pole LPF/HPF entry signature
    /// (`push {r8, sb, sl, fp, lr}; mov lr, r5; mov ip, r6; cmp r4, #0x80`) at offset `+0xC44`
    /// of the 3,476-byte Kaede ARM mixer block, then inspects the DMA8 LUT mask constant at
    /// `-0x450` (`+0x7F4` of the ARM block: `0x000001FF` for V1, `0x000003FF` for V2).
    #[must_use]
    pub fn detect_from_bytes(data: &[u8]) -> Option<Self> {
        // Post-mix filter ARM signature at +0xC44 of the 0xD94 Kaede ARM mixer block:
        //   E92D4F00  push {r8, sb, sl, fp, lr}
        //   E1A0E005  mov  lr, r5
        //   E1A0C006  mov  ip, r6
        //   E3540080  cmp  r4, #0x80
        const FILTER_SIG: [u8; 16] = [
            0x00, 0x4F, 0x2D, 0xE9, 0x05, 0xE0, 0xA0, 0xE1, 0x06, 0xC0, 0xA0, 0xE1, 0x80, 0x00,
            0x54, 0xE3,
        ];
        // Distance from +0x7F4 (LUT mask literal) to +0xC44 (filter entry) is exactly 0x450 bytes.
        const MASK_DELTA: usize = 0x450;

        let mut start = 0usize;
        while let Some(rel) = find_subslice(&data[start..], &FILTER_SIG) {
            let filter_pos = start + rel;
            if filter_pos >= MASK_DELTA {
                let mask_off = filter_pos - MASK_DELTA;
                let mask = u32::from_le_bytes([
                    data[mask_off],
                    data[mask_off + 1],
                    data[mask_off + 2],
                    data[mask_off + 3],
                ]);
                match mask {
                    0x0000_01FF => return Some(Self::V1Lut512),
                    0x0000_03FF => return Some(Self::V2Lut1024),
                    _ => {}
                }
            }
            start = filter_pos + 4;
        }
        None
    }
}

/// Find the first occurrence of `needle` in `haystack`.
#[inline]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_v1_and_v2_from_synthetic_arm_block() {
        let mut buf = [0u8; 0x1000];
        let filter_off = 0xC44usize;
        let sig: [u8; 16] = [
            0x00, 0x4F, 0x2D, 0xE9, 0x05, 0xE0, 0xA0, 0xE1, 0x06, 0xC0, 0xA0, 0xE1, 0x80, 0x00,
            0x54, 0xE3,
        ];
        buf[filter_off..filter_off + 16].copy_from_slice(&sig);

        // V1 (0x1FF)
        buf[0x7F4..0x7F8].copy_from_slice(&0x0000_01FFu32.to_le_bytes());
        assert_eq!(
            KaedeVersion::detect_from_bytes(&buf),
            Some(KaedeVersion::V1Lut512)
        );
        assert_eq!(KaedeVersion::V1Lut512.lut_size(), 512);
        assert_eq!(KaedeVersion::V1Lut512.lut_mask(), 0x1FF);
        assert_eq!(KaedeVersion::V1Lut512.clamp_headroom(), (-256, 255));

        // V2 (0x3FF)
        buf[0x7F4..0x7F8].copy_from_slice(&0x0000_03FFu32.to_le_bytes());
        assert_eq!(
            KaedeVersion::detect_from_bytes(&buf),
            Some(KaedeVersion::V2Lut1024)
        );
        assert_eq!(KaedeVersion::V2Lut1024.lut_size(), 1024);
        assert_eq!(KaedeVersion::V2Lut1024.lut_mask(), 0x3FF);
        assert_eq!(KaedeVersion::V2Lut1024.clamp_headroom(), (-512, 511));

        // Unknown mask returns None
        buf[0x7F4..0x7F8].copy_from_slice(&0x0000_07FFu32.to_le_bytes());
        assert_eq!(KaedeVersion::detect_from_bytes(&buf), None);
    }
}
