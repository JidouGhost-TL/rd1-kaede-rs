//! Error types for Nintendo R&D1 Kaede parsing and synthesis.

use core::fmt;

/// Errors returned when parsing Nintendo R&D1 Kaede sample descriptors, song tables,
/// Standard MIDI chunks, or synthesis parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before a complete header or payload could be read.
    UnexpectedEof,
    /// Descriptor, song table entry, or container header failed structural validation.
    InvalidHeader(&'static str),
    /// Loop range was invalid for the declared sample length.
    InvalidLoopRange {
        /// Loop start offset in samples.
        loop_start: u32,
        /// Loop end offset in samples.
        loop_end: u32,
        /// Total sample length in samples.
        sample_len: u32,
    },
    /// Embedded Standard MIDI (`MThd` / `MTrk`) chunk failed structural validation.
    InvalidMidi(&'static str),
    /// Sample index was out of range.
    InvalidSample(u32),
    /// Invalid synthesis parameter.
    InvalidParameter(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => {
                write!(f, "Unexpected end of data while parsing R&D1 Kaede stream")
            }
            Self::InvalidHeader(msg) => write!(f, "Invalid R&D1 Kaede header: {msg}"),
            Self::InvalidLoopRange {
                loop_start,
                loop_end,
                sample_len,
            } => write!(
                f,
                "Invalid R&D1 Kaede loop range {loop_start}..{loop_end} for {sample_len}-byte sample"
            ),
            Self::InvalidMidi(msg) => write!(f, "Invalid R&D1 Kaede Standard MIDI: {msg}"),
            Self::InvalidSample(id) => write!(f, "Invalid R&D1 Kaede sample index {id}"),
            Self::InvalidParameter(msg) => write!(f, "Invalid R&D1 Kaede parameter: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience result alias for `rd1-kaede`.
pub type Result<T> = core::result::Result<T, Error>;
