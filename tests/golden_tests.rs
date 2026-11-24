//! Byte-exact ARM7TDMI golden tests for all 10 Nintendo R&D1 Kaede mixer kernels across
//! both hardware revisions (`V1Lut512` and `V2Lut1024`).
//!
//! Every golden binary in `tests/golden/` was captured by executing the authentic ARM7TDMI
//! Kaede mixer instructions inside headless `libmgba.so.0.10` on clean-room synthetic
//! deterministic waveforms. Every single output byte is verified with strict `assert_eq!`
//! (zero tolerance).

use rd1_kaede::{
    AccuracySynth, KaedeFilterState, KaedeReverbState, KaedeVersion, KaedeVoice,
};

fn synth_pcm() -> [u8; 64] {
    core::array::from_fn(|i| ((((i as i32) * 29 + 13) % 256) - 128) as i8 as u8)
}

fn synth_delay_l() -> [u8; 64] {
    core::array::from_fn(|i| ((((i as i32) * 17 + 5) % 256) - 128) as i8 as u8)
}

fn synth_delay_r() -> [u8; 64] {
    core::array::from_fn(|i| ((((i as i32) * 23 + 91) % 256) - 128) as i8 as u8)
}

fn init_acc_stereo() -> [i32; 64] {
    core::array::from_fn(|i| {
        let idx = i as i32;
        if idx % 3 != 0 {
            idx * 131 - 4000
        } else {
            idx * 53
        }
    })
}

fn init_acc_mono() -> [i32; 32] {
    core::array::from_fn(|i| {
        let idx = i as i32;
        if idx % 2 == 0 {
            idx * 97 - 2000
        } else {
            idx * 61
        }
    })
}

fn extreme_acc_i32() -> [i32; 64] {
    const SHIFTS: [i32; 32] = [
        0, 1, 64, 127, 128, 200, 255, 256, 300, 383, 384, 511, 512, 600, -1, -2, -64, -127, -128,
        -129, -200, -255, -256, -257, -300, -384, -511, -512, -513, -600, 42, -42,
    ];
    let mut out = [0i32; 64];
    for i in 0..32 {
        let vl = (SHIFTS[i] << 7) + (((i as i32) * 19) & 0x7F);
        let vr = (SHIFTS[(i * 7 + 3) % 32] << 7) + (((i as i32) * 37) & 0x7F);
        out[i * 2] = vl;
        out[i * 2 + 1] = vr;
    }
    out
}

fn i32s_to_le_bytes(slice: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(slice.len() * 4);
    for &v in slice {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn read_u32_le(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_i32_le(bytes: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

#[test]
fn golden_01_stereo_linear_oneshot() {
    let golden = include_bytes!("golden/stereo_linear_oneshot.bin");
    let pcm = synth_pcm();
    let mut acc = init_acc_stereo();
    let mut voice = KaedeVoice::new((40 << 14) + 0x1234, 63, None, 0x5000, 95, 63);

    AccuracySynth::mix_stereo_linear(&pcm, &mut voice, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
    assert_eq!(u32::from(voice.active), read_u32_le(golden, 256));
    assert_eq!(voice.phase_q14, read_u32_le(golden, 256 + 12));
}

#[test]
fn golden_02_stereo_linear_looped() {
    let golden = include_bytes!("golden/stereo_linear_looped.bin");
    let pcm = synth_pcm();
    let mut acc = init_acc_stereo();
    let mut voice = KaedeVoice::new((30 << 14) + 0x2B70, 48, Some(16), 0x5234, 110, 45);

    AccuracySynth::mix_stereo_linear(&pcm, &mut voice, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
    assert_eq!(u32::from(voice.active), read_u32_le(golden, 256));
    assert_eq!(voice.phase_q14, read_u32_le(golden, 256 + 12));
}

#[test]
fn golden_03_mono_linear_looped() {
    let golden = include_bytes!("golden/mono_linear_looped.bin");
    let pcm = synth_pcm();
    let mut acc = init_acc_mono();
    let mut voice = KaedeVoice::new((20 << 14) + 0x19C0, 40, Some(8), 0x6789, 88, 0);

    AccuracySynth::mix_mono_linear(&pcm, &mut voice, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..128]);
    assert_eq!(u32::from(voice.active), read_u32_le(golden, 128));
    assert_eq!(voice.phase_q14, read_u32_le(golden, 128 + 12));
}

#[test]
fn golden_04_stereo_reverb_biquad() {
    let golden = include_bytes!("golden/stereo_reverb_biquad.bin");
    let dl = synth_delay_l();
    let dr = synth_delay_r();
    let mut acc = [0i32; 64];
    let mut state = KaedeReverbState {
        hp_l: 120,
        lp_l: -45,
        hp_r: -80,
        lp_r: 65,
        hp_shift: 4,
        lp_shift: 2,
        reverb_vol: 40,
        pos_l: 48,
        pos_r: 52,
    };

    AccuracySynth::apply_stereo_reverb(&dl, &dr, &mut state, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
}

#[test]
fn golden_05_mono_reverb_biquad() {
    let golden = include_bytes!("golden/mono_reverb_biquad.bin");
    let dl = synth_delay_l();
    let mut acc = [0i32; 32];
    let mut state = KaedeReverbState {
        hp_l: 90,
        lp_l: -30,
        hp_r: 0,
        lp_r: 0,
        hp_shift: 3,
        lp_shift: 2,
        reverb_vol: 35,
        pos_l: 48,
        pos_r: 0,
    };

    AccuracySynth::apply_mono_reverb(&dl, &mut state, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..128]);
}

#[test]
fn golden_06_v1_lut512_dma_stereo() {
    let golden = include_bytes!("golden/v1_lut512_dma_stereo.bin");
    let acc = extreme_acc_i32();
    let mut dma_l = [0i8; 32];
    let mut dma_r = [0i8; 32];

    AccuracySynth::pack_stereo_to_dma8(KaedeVersion::V1Lut512, &acc, &mut dma_l, &mut dma_r);

    let got_l: Vec<u8> = dma_l.iter().map(|&b| b as u8).collect();
    let got_r: Vec<u8> = dma_r.iter().map(|&b| b as u8).collect();
    assert_eq!(&got_l[..], &golden[0..32]);
    assert_eq!(&got_r[..], &golden[32..64]);
}

#[test]
fn golden_07_v2_lut1024_dma_stereo() {
    let golden_v1 = include_bytes!("golden/v1_lut512_dma_stereo.bin");
    let golden_v2 = include_bytes!("golden/v2_lut1024_dma_stereo.bin");
    let acc = extreme_acc_i32();
    let mut dma_l = [0i8; 32];
    let mut dma_r = [0i8; 32];

    AccuracySynth::pack_stereo_to_dma8(KaedeVersion::V2Lut1024, &acc, &mut dma_l, &mut dma_r);

    let got_l: Vec<u8> = dma_l.iter().map(|&b| b as u8).collect();
    let got_r: Vec<u8> = dma_r.iter().map(|&b| b as u8).collect();
    assert_eq!(&got_l[..], &golden_v2[0..32]);
    assert_eq!(&got_r[..], &golden_v2[32..64]);

    // Explicitly verify V1 vs V2 divergence (28/64 bytes differ on extreme accumulators)
    let diffs = golden_v1
        .iter()
        .zip(golden_v2.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(diffs, 28);
}

#[test]
fn golden_08_v1_lut512_dma_mono() {
    let golden = include_bytes!("golden/v1_lut512_dma_mono.bin");
    let acc = extreme_acc_i32();
    let mut dma = [0i8; 32];

    AccuracySynth::pack_mono_to_dma8(KaedeVersion::V1Lut512, &acc[..32], &mut dma);

    let got: Vec<u8> = dma.iter().map(|&b| b as u8).collect();
    assert_eq!(&got[..], &golden[0..32]);
}

#[test]
fn golden_09_v2_lut1024_dma_mono() {
    let golden = include_bytes!("golden/v2_lut1024_dma_mono.bin");
    let acc = extreme_acc_i32();
    let mut dma = [0i8; 32];

    AccuracySynth::pack_mono_to_dma8(KaedeVersion::V2Lut1024, &acc[..32], &mut dma);

    let got: Vec<u8> = dma.iter().map(|&b| b as u8).collect();
    assert_eq!(&got[..], &golden[0..32]);
}

#[test]
fn golden_10_stereo_direct_copy() {
    let golden = include_bytes!("golden/stereo_direct_copy.bin");
    let pcm = synth_pcm();
    let mut acc = init_acc_stereo();
    let mut voice = KaedeVoice::new_direct_copy(29, 48, Some(16), 77, 103);

    AccuracySynth::mix_stereo_direct_copy(&pcm, &mut voice, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
    assert_eq!(u32::from(voice.active), read_u32_le(golden, 256));
    assert_eq!(voice.sample_pos() as u32, read_u32_le(golden, 256 + 16) - 0x0200_0200);
}

#[test]
fn golden_11_stereo_nearest_looped() {
    let golden = include_bytes!("golden/stereo_nearest_looped.bin");
    let pcm = synth_pcm();
    let mut acc = init_acc_stereo();
    let mut voice = KaedeVoice::new((25 << 14) + 0x1F00, 44, Some(12), 0x5678, 66, 120);

    AccuracySynth::mix_stereo_nearest(&pcm, &mut voice, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
    assert_eq!(u32::from(voice.active), read_u32_le(golden, 256));
    assert_eq!(voice.phase_q14, read_u32_le(golden, 256 + 12));
}

#[test]
fn golden_12_mono_nearest_quirk() {
    let golden = include_bytes!("golden/mono_nearest_quirk.bin");
    let pcm = synth_pcm();
    let mut acc = init_acc_mono();
    let mut voice = KaedeVoice::new((10 << 14) + 0x0800, 56, Some(8), 0x4800, 75, 0);

    AccuracySynth::mix_mono_nearest(&pcm, &mut voice, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..128]);
    assert_eq!(u32::from(voice.active), read_u32_le(golden, 128));
    assert_eq!(voice.phase_q14, read_u32_le(golden, 128 + 12));
}

#[test]
fn golden_13_stereo_filter_lpf() {
    let golden = include_bytes!("golden/stereo_filter_lpf.bin");
    let mut acc = init_acc_stereo();
    let mut state = KaedeFilterState::new(1500, -900, 0x40);

    AccuracySynth::apply_stereo_filter(&mut state, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
    assert_eq!(state.state_l, read_i32_le(golden, 256 + 4));
    assert_eq!(state.state_r, read_i32_le(golden, 256 + 8));
}

#[test]
fn golden_14_stereo_filter_hpf() {
    let golden = include_bytes!("golden/stereo_filter_hpf.bin");
    let mut acc = init_acc_stereo();
    let mut state = KaedeFilterState::new(1500, -900, 0xC0);

    AccuracySynth::apply_stereo_filter(&mut state, &mut acc);

    assert_eq!(&i32s_to_le_bytes(&acc)[..], &golden[0..256]);
    assert_eq!(state.state_l, read_i32_le(golden, 256 + 4));
    assert_eq!(state.state_r, read_i32_le(golden, 256 + 8));
}
