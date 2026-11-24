use std::hint::black_box;
use std::time::Instant;

use rd1_kaede::{
    AccuracySynth, KaedeFilterState, KaedeReverbState, KaedeSampleView, KaedeVersion, KaedeVoice,
    StudioSynth,
};

fn main() {
    println!("=== rd1-kaede-rs Audio Performance Benchmarks ===");

    let pcm: Vec<u8> = (0i32..4096)
        .map(|i| (((i * 29 + 13) % 256) - 128) as i8 as u8)
        .collect();
    let sample = KaedeSampleView::from_pcm_bytes(0, 13_379, 60, Some((256, 3840)), &pcm);

    let mut acc_stereo = [0i32; 256]; // 128 stereo frames
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];

    let iters = 200_000usize;

    // 1. AccuracySynth Stereo Linear + V2 LUT1024 Clamp (128 frames)
    let start = Instant::now();
    for _ in 0..iters {
        acc_stereo.fill(0);
        let mut voice = KaedeVoice::new(0, 3840, Some(256), 0x5234, 110, 85);
        AccuracySynth::mix_stereo_linear(black_box(&pcm), &mut voice, &mut acc_stereo);
        AccuracySynth::pack_stereo_to_dma8(
            KaedeVersion::V2Lut1024,
            &acc_stereo,
            &mut dma_l,
            &mut dma_r,
        );
        black_box((&dma_l, &dma_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::stereo_linear + lut1024 (128f):     {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 128) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 2. AccuracySynth Stereo Direct Copy (128 frames)
    let start = Instant::now();
    for _ in 0..iters {
        acc_stereo.fill(0);
        let mut voice = KaedeVoice::new_direct_copy(0, 3840, Some(256), 95, 95);
        AccuracySynth::mix_stereo_direct_copy(black_box(&pcm), &mut voice, &mut acc_stereo);
        black_box(&acc_stereo);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::stereo_direct_copy (128f):          {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 128) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 3. AccuracySynth Stereo Reverb + LPF (128 frames)
    let start = Instant::now();
    for _ in 0..iters {
        let mut rev = KaedeReverbState {
            hp_l: 0,
            lp_l: 0,
            hp_r: 0,
            lp_r: 0,
            hp_shift: 4,
            lp_shift: 2,
            reverb_vol: 40,
            pos_l: 0,
            pos_r: 64,
        };
        let mut flt = KaedeFilterState::new(0, 0, 0x40);
        AccuracySynth::apply_stereo_reverb(
            black_box(&pcm[..512]),
            black_box(&pcm[512..1024]),
            &mut rev,
            &mut acc_stereo,
        );
        AccuracySynth::apply_stereo_filter(&mut flt, &mut acc_stereo);
        black_box(&acc_stereo);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::stereo_reverb + lpf (128f):         {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 128) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 4. StudioSynth 48 kHz Lanczos-3 (1024 frames)
    let studio = StudioSynth::default();
    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];
    let studio_iters = 50_000usize;
    let start = Instant::now();
    for _ in 0..studio_iters {
        let mut pos = 0.0f64;
        studio.render_voice_f32(
            black_box(&sample),
            &mut pos,
            13_379.0,
            0.85,
            0.25,
            &mut out_l,
            &mut out_r,
        );
        black_box((&out_l, &out_r));
    }
    let elapsed = start.elapsed();
    println!(
        "StudioSynth::render_voice_f32 (48kHz Lanczos3, 1k):{:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / studio_iters as f64,
        ((studio_iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
