//! Zero-allocation parsers for Nintendo R&D1 Kaede binary structures:
//! - 24-byte (`0x18`) DirectSound PCM8 sample descriptors ([`KaedeSampleDesc`])
//! - 20-byte (`0x14`) song table entries ([`KaedeSongEntry`])
//! - Verbatim Standard MIDI (`MThd` / `MTrk`) header validation ([`KaedeMidiHeader`])
//! - Borrowed ([`KaedeSampleView`]) and owned ([`KaedeSample`]) PCM8 sample views.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// GBA Cartridge ROM bus base address (`0x0800_0000`).
pub const GBA_ROM_BUS_BASE: u32 = 0x0800_0000;

/// Size of a Nintendo R&D1 Kaede PCM8 sample descriptor in bytes (`0x18` = 24 bytes).
pub const KAEDE_SAMPLE_DESC_SIZE: usize = 24;

/// Size of a Nintendo R&D1 Kaede song table entry in bytes (`0x14` = 20 bytes).
pub const KAEDE_SONG_ENTRY_SIZE: usize = 20;

/// Standard MIDI file header magic (`"MThd"` + 4-byte big-endian chunk length `6`).
pub const MTHD_MAGIC: &[u8; 8] = b"MThd\x00\x00\x00\x06";

/// Standard MIDI track chunk magic (`"MTrk"`).
pub const MTRK_MAGIC: &[u8; 4] = b"MTrk";

/// Size of a Standard MIDI `MThd` header chunk in bytes (`14`).
pub const MTHD_HEADER_SIZE: usize = 14;

/// Parsed 24-byte (`0x18`) Nintendo R&D1 Kaede DirectSound PCM8 sample descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KaedeSampleDesc {
    /// GBA ROM pointer (`0x08xxxxxx`) to raw signed 8-bit linear mono PCM (`i8`).
    pub sample_ptr: u32,
    /// Sample length in bytes (8-bit mono samples).
    pub sample_len: u32,
    /// Native sample playback rate in Hz (e.g., `10512`, `13379`, `16000`).
    pub sample_rate_hz: u32,
    /// MIDI root note (`0..=127`; `60` = Middle C / C4).
    pub root_note: u32,
    /// Loop start offset in samples (`0` when unlooped).
    pub loop_start: u32,
    /// Loop end offset in samples (`0` when unlooped; `loop_start < loop_end <= sample_len`).
    pub loop_end: u32,
}

impl KaedeSampleDesc {
    /// Parse and validate a 24-byte `KaedeSampleDesc` from `data[..24]`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < KAEDE_SAMPLE_DESC_SIZE {
            return Err(Error::UnexpectedEof);
        }
        let sample_ptr = read_u32_le(data, 0x00);
        let sample_len = read_u32_le(data, 0x04);
        let sample_rate_hz = read_u32_le(data, 0x08);
        let root_note = read_u32_le(data, 0x0C);
        let loop_start = read_u32_le(data, 0x10);
        let loop_end = read_u32_le(data, 0x14);

        if sample_ptr < GBA_ROM_BUS_BASE {
            return Err(Error::InvalidHeader("sample_ptr below GBA ROM bus base"));
        }
        if sample_len == 0 {
            return Err(Error::InvalidHeader("sample_len is zero"));
        }
        if sample_rate_hz == 0 {
            return Err(Error::InvalidHeader("sample_rate_hz is zero"));
        }
        if root_note > 127 {
            return Err(Error::InvalidHeader("root_note exceeds 127"));
        }
        let valid_loop =
            (loop_start == 0 && loop_end == 0) || (loop_start < loop_end && loop_end <= sample_len);
        if !valid_loop {
            return Err(Error::InvalidLoopRange {
                loop_start,
                loop_end,
                sample_len,
            });
        }

        Ok(Self {
            sample_ptr,
            sample_len,
            sample_rate_hz,
            root_note,
            loop_start,
            loop_end,
        })
    }

    /// Returns `true` if the sample has a valid active loop range.
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.loop_end > 0 && self.loop_start < self.loop_end && self.loop_end <= self.sample_len
    }

    /// Returns `Some((loop_start, loop_end))` if looped, or `None` for one-shot samples.
    #[inline]
    #[must_use]
    pub const fn loop_range(&self) -> Option<(u32, u32)> {
        if self.is_looped() {
            Some((self.loop_start, self.loop_end))
        } else {
            None
        }
    }

    /// Convert `sample_ptr` to a 0-based ROM file offset, if it resides in the GBA ROM window.
    #[inline]
    #[must_use]
    pub const fn pcm_rom_offset(&self) -> Option<usize> {
        if self.sample_ptr >= GBA_ROM_BUS_BASE {
            Some((self.sample_ptr - GBA_ROM_BUS_BASE) as usize)
        } else {
            None
        }
    }
}

/// Parsed 20-byte (`0x14`) Nintendo R&D1 Kaede song table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KaedeSongEntry {
    /// GBA ROM pointer (`0x08xxxxxx`) directly to `"MThd\x00\x00\x00\x06"`.
    pub mthd_ptr: u32,
    /// Packed tempo / soundbank flags.
    pub flags_tempo: u32,
    /// Volume / priority (typically `0x0000_00FF`).
    pub vol_priority: u32,
    /// GBA ROM pointer to NUL-terminated ASCII song name (empty `""` in *Rhythm Tengoku*).
    pub name_ptr: u32,
    /// Sequential song ID (`0, 1, 2...` in *WarioWare*; `1000, 1001...` in *Rhythm Tengoku*).
    pub song_id: u32,
}

impl KaedeSongEntry {
    /// Parse a 20-byte `KaedeSongEntry` from `data[..20]`.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < KAEDE_SONG_ENTRY_SIZE {
            return Err(Error::UnexpectedEof);
        }
        let mthd_ptr = read_u32_le(data, 0x00);
        if mthd_ptr < GBA_ROM_BUS_BASE {
            return Err(Error::InvalidHeader("mthd_ptr below GBA ROM bus base"));
        }
        Ok(Self {
            mthd_ptr,
            flags_tempo: read_u32_le(data, 0x04),
            vol_priority: read_u32_le(data, 0x08),
            name_ptr: read_u32_le(data, 0x0C),
            song_id: read_u32_le(data, 0x10),
        })
    }

    /// Convert `mthd_ptr` to a 0-based ROM file offset.
    #[inline]
    #[must_use]
    pub const fn mthd_rom_offset(&self) -> Option<usize> {
        if self.mthd_ptr >= GBA_ROM_BUS_BASE {
            Some((self.mthd_ptr - GBA_ROM_BUS_BASE) as usize)
        } else {
            None
        }
    }
}

/// Parsed Standard MIDI (`MThd`) header chunk embedded in a Nintendo R&D1 Kaede ROM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KaedeMidiHeader {
    /// Standard MIDI format (`0`, `1`, or `2`; Kaede sequences are typically Format `1`).
    pub format: u16,
    /// Number of contiguous `MTrk` chunks following the header.
    pub num_tracks: u16,
    /// Timing division (ticks per quarter note; typically `24` PPQN in Kaede titles).
    pub division: u16,
    /// Total byte length of the complete Standard MIDI file (`MThd` + all `MTrk` chunks).
    pub total_bytes: usize,
}

impl KaedeMidiHeader {
    /// Parse and structurally validate a complete Standard MIDI file starting at `data[0]`.
    ///
    /// Walks all `num_tracks` contiguous `"MTrk"` chunks to ensure none are truncated,
    /// returning the header and exact `total_bytes` slice length.
    pub fn parse_and_validate(data: &[u8]) -> Result<Self> {
        if data.len() < MTHD_HEADER_SIZE {
            return Err(Error::UnexpectedEof);
        }
        if &data[0..8] != MTHD_MAGIC {
            return Err(Error::InvalidMidi("missing MThd\\x00\\x00\\x00\\x06 magic"));
        }
        let format = u16::from_be_bytes([data[8], data[9]]);
        let num_tracks = u16::from_be_bytes([data[10], data[11]]);
        let division = u16::from_be_bytes([data[12], data[13]]);
        if format > 2 || num_tracks == 0 || num_tracks > 64 {
            return Err(Error::InvalidMidi("invalid MIDI format or track count"));
        }

        let mut cur = MTHD_HEADER_SIZE;
        for _ in 0..num_tracks {
            if cur + 8 > data.len() {
                return Err(Error::UnexpectedEof);
            }
            if &data[cur..cur + 4] != MTRK_MAGIC {
                return Err(Error::InvalidMidi("expected MTrk chunk header"));
            }
            let track_len = u32::from_be_bytes([
                data[cur + 4],
                data[cur + 5],
                data[cur + 6],
                data[cur + 7],
            ]) as usize;
            let next = cur
                .checked_add(8)
                .and_then(|v| v.checked_add(track_len))
                .ok_or(Error::InvalidMidi("MTrk length overflow"))?;
            if next > data.len() {
                return Err(Error::UnexpectedEof);
            }
            cur = next;
        }

        Ok(Self {
            format,
            num_tracks,
            division,
            total_bytes: cur,
        })
    }
}

/// Zero-copy borrowed view of a Nintendo R&D1 Kaede signed 8-bit PCM sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KaedeSampleView<'a> {
    /// Sample index or identifier.
    pub id: u32,
    /// Native sample playback rate in Hz.
    pub sample_rate_hz: u32,
    /// MIDI root note (`0..=127`).
    pub root_note: u8,
    /// Optional `(loop_start, loop_end)` in sample frames.
    pub loop_range: Option<(u32, u32)>,
    /// Raw signed 8-bit linear mono PCM bytes (`i8` stored as `u8`).
    pub pcm_bytes: &'a [u8],
}

impl<'a> KaedeSampleView<'a> {
    /// Create a borrowed [`KaedeSampleView`] from raw PCM8 bytes and metadata.
    #[inline]
    #[must_use]
    pub const fn from_pcm_bytes(
        id: u32,
        sample_rate_hz: u32,
        root_note: u8,
        loop_range: Option<(u32, u32)>,
        pcm_bytes: &'a [u8],
    ) -> Self {
        Self {
            id,
            sample_rate_hz,
            root_note,
            loop_range,
            pcm_bytes,
        }
    }

    /// Resolve a [`KaedeSampleDesc`] against a full GBA ROM slice into a zero-copy sample view.
    pub fn from_desc_and_rom(id: u32, desc: &KaedeSampleDesc, rom: &'a [u8]) -> Result<Self> {
        let off = desc
            .pcm_rom_offset()
            .ok_or(Error::InvalidHeader("sample_ptr out of ROM range"))?;
        let end = off
            .checked_add(desc.sample_len as usize)
            .ok_or(Error::UnexpectedEof)?;
        if end > rom.len() {
            return Err(Error::UnexpectedEof);
        }
        Ok(Self {
            id,
            sample_rate_hz: desc.sample_rate_hz,
            root_note: desc.root_note as u8,
            loop_range: desc.loop_range(),
            pcm_bytes: &rom[off..end],
        })
    }

    /// Number of 8-bit mono samples in the buffer.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.pcm_bytes.len()
    }

    /// Returns `true` if the sample buffer is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pcm_bytes.is_empty()
    }
}

/// Owned Nintendo R&D1 Kaede signed 8-bit PCM sample (requires `alloc` feature).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KaedeSample {
    /// Sample index or identifier.
    pub id: u32,
    /// Native sample playback rate in Hz.
    pub sample_rate_hz: u32,
    /// MIDI root note (`0..=127`).
    pub root_note: u8,
    /// Optional `(loop_start, loop_end)` in sample frames.
    pub loop_range: Option<(u32, u32)>,
    /// Owned raw signed 8-bit linear mono PCM bytes (`i8` stored as `u8`).
    pub pcm_bytes: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl KaedeSample {
    /// Borrow this owned sample as a [`KaedeSampleView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> KaedeSampleView<'_> {
        KaedeSampleView {
            id: self.id,
            sample_rate_hz: self.sample_rate_hz,
            root_note: self.root_note,
            loop_range: self.loop_range,
            pcm_bytes: &self.pcm_bytes,
        }
    }
}

#[inline]
fn read_u32_le(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sample_desc_song_entry_and_midi() {
        let mut raw = [0u8; 24];
        raw[0x00..0x04].copy_from_slice(&(GBA_ROM_BUS_BASE + 0x1000).to_le_bytes());
        raw[0x04..0x08].copy_from_slice(&256u32.to_le_bytes());
        raw[0x08..0x0C].copy_from_slice(&13_379u32.to_le_bytes());
        raw[0x0C..0x10].copy_from_slice(&60u32.to_le_bytes());
        raw[0x10..0x14].copy_from_slice(&32u32.to_le_bytes());
        raw[0x14..0x18].copy_from_slice(&240u32.to_le_bytes());

        let desc = KaedeSampleDesc::parse(&raw).expect("valid desc");
        assert_eq!(desc.pcm_rom_offset(), Some(0x1000));
        assert_eq!(desc.loop_range(), Some((32, 240)));
        assert!(desc.is_looped());

        let mut song = [0u8; 20];
        song[0x00..0x04].copy_from_slice(&(GBA_ROM_BUS_BASE + 0x2000).to_le_bytes());
        song[0x08..0x0C].copy_from_slice(&0xFFu32.to_le_bytes());
        song[0x10..0x14].copy_from_slice(&1000u32.to_le_bytes());
        let entry = KaedeSongEntry::parse(&song).expect("valid song entry");
        assert_eq!(entry.mthd_rom_offset(), Some(0x2000));
        assert_eq!(entry.song_id, 1000);

        let midi: &[u8] = &[
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 1, 0, 1, 0, 24, b'M', b'T', b'r', b'k', 0, 0,
            0, 4, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let hdr = KaedeMidiHeader::parse_and_validate(midi).expect("valid MIDI");
        assert_eq!(hdr.format, 1);
        assert_eq!(hdr.num_tracks, 1);
        assert_eq!(hdr.division, 24);
        assert_eq!(hdr.total_bytes, midi.len());
    }
}
