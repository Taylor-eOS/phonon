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
INPUT_FILE = os.path.join(OUTPUT_DIR, "input.txt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.wav")
WARMUP_SENTENCE = "Hi"
GAP_SECONDS = 0.4

def load_sentences(path):
    if not os.path.exists(path):
        print(f"Input file not found: {path}")
        raise SystemExit(1)
    sentences = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sentences.append(line)
    if not sentences:
        print(f"No sentences found in {path}")
        raise SystemExit(1)
    return sentences

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
    print(f"Model loaded in {time.perf_counter() - start:.2f}s")
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
    print(f"{duration:.2f} s audio in {elapsed:.3f} s {speed:.1f}x")
    return samples

def main():
    sentences = load_sentences(INPUT_FILE)
    print(f"Loaded {len(sentences)} sentence(s) from {INPUT_FILE}")
    tts = load_model(
    tts.synth(WARMUP_SENTENCE)
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    all_samples = []
    for i, text in enumerate(sentences):
        print(f"[{i+1}/{len(sentences)}] {text}")
        samples = synthesize(tts, text)
        if samples.size == 0:
            print("Warning: no audio returned, skipping")
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
