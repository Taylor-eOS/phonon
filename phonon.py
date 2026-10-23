import os
import time
import wave
import numpy as np
import ptts

MODEL = "hf://gradium/phonon-40m-de"
LANG = "de"
QUANT = ""
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.wav")
GAP_SECONDS = 0.4
SENTENCES = [
    "Guten Tag, das ist ein kurzer Test des deutschen Phonon Modells.",
    "Schöne Grüße aus Köln, wo die Straßenbahn heute größtenteils pünktlich fuhr.",
    "Wir haben viel zu feiern.",
]

def load_model():
    kwargs = {"lang": LANG}
    if QUANT:
        kwargs["quant"] = QUANT
    start = time.perf_counter()
    try:
        tts = ptts.TTS(MODEL, **kwargs)
    except Exception as error:
        print(f"Model load failed: {error}")
        print("The 40m models are gated: set HF_TOKEN or run 'hf auth login'.")
        raise SystemExit(1)
    print(f"Model loaded in {time.perf_counter() - start:.2f} s")
    return tts

def to_mono(samples):
    return np.asarray(samples, dtype=np.float32).reshape(-1)

def resample(samples, from_rate, to_rate):
    if from_rate == to_rate or samples.size == 0:
        return samples
    n_out = int(round(samples.size * to_rate / from_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)

def write_wav(path, samples):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())

def synthesize(tts, text, voice_name="Anton"): 
    start = time.perf_counter()
    samples = to_mono(tts.synth(text, voice=voice_name))
    elapsed = time.perf_counter() - start
    duration = samples.size / SAMPLE_RATE
    speed = duration / elapsed if elapsed > 0 else float("inf")
    print(f"  {duration:.2f} s audio in {elapsed:.3f} s ({speed:.1f}x real time)")
    return samples

def main():
    tts = load_model()
    print("Warming up...")
    tts.synth("Hallo.")
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    all_samples = []
    for i, text in enumerate(SENTENCES):
        print(f"[{i+1}/{len(SENTENCES)}] {text}")
        samples = synthesize(tts, text)
        if samples.size == 0:
            print("  WARNING: no audio returned, skipping")
            continue
        all_samples.append(samples)
        all_samples.append(gap)
    if not all_samples:
        print("No audio produced.")
        raise SystemExit(1)
    full = np.concatenate(all_samples)
    write_wav(OUTPUT_FILE, full)
    print(f"Wrote {full.size / SAMPLE_RATE:.2f} s to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
