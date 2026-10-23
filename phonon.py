import os
import time
import wave
import numpy as np
import ptts

MODEL_TEMPLATE = "hf://gradium/phonon-40m-{lang}"
LANGUAGES = ("en", "de")
DEFAULT_LANG = "en"
VOICES = {
    "de": [
        ("Male", "Anton"), ("Male", "Mirko"), ("Male", "Mats"), ("Male", "Henrik"), ("Male", "Leon"), ("Male", "Mika"),
        ("Female", "Nele"), ("Female", "Jette"), ("Female", "Annika"), ("Female", "Jule"), ("Female", "Lotta"), ("Female", "Josefine"), ("Female", "Svea"),
    ],
    "en": [
        ("US Female", "Zoey"), ("US Female", "Juniper"), ("US Female", "Everly"), ("US Female", "Harper"), ("US Female", "Hadley"), ("US Female", "Delia"),
        ("US Male", "Easton"), ("US Male", "Micah"), ("US Male", "Trevor"), ("US Male", "Russell"), ("US Male", "Preston"), ("US Male", "Miles"),
        ("UK Female", "Maeve"), ("UK Female", "Freya"), ("UK Female", "Pippa"), ("UK Female", "Elodie-Rose"), ("UK Female", "Imogen"), ("UK Female", "Lottie"), ("UK Female", "Bryony"),
        ("UK Male", "Alfie"), ("UK Male", "Hugh"), ("UK Male", "Edwin"), ("UK Male", "Spencer"), ("UK Male", "Toby"), ("UK Male", "Teddy"),
    ],
}
MODEL = ""
LANG = DEFAULT_LANG
VOICE = ""
QUANT = ""
SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(OUTPUT_DIR, "input.txt")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.wav")
WARMUP_SENTENCE = "Hi"
GAP_SECONDS = 0.4

def choose_language():
    global LANG, MODEL
    while True:
        answer = input(f"Language {'/'.join(LANGUAGES)} ({DEFAULT_LANG}): ").strip().lower() or DEFAULT_LANG
        if answer in LANGUAGES:
            break
        print(f"Unsupported language: {answer}")
    LANG = answer
    MODEL = MODEL_TEMPLATE.format(lang=LANG)

def choose_voice():
    global VOICE
    options = VOICES[LANG]
    names = {name.lower(): name for _, name in options}
    for number, (group, name) in enumerate(options, 1):
        print(f"{number:2d}. {name} ({group})")
    while True:
        answer = input("Voice (number or name) [1]: ").strip().lower()
        if not answer:
            VOICE = options[0][1]
            return
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            VOICE = options[int(answer) - 1][1]
            return
        if answer in names:
            VOICE = names[answer]
            return
        print(f"Unsupported voice: {answer}")

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

def synthesize(tts, text, voice_name=None):
    voice_name = voice_name or VOICE
    start = time.perf_counter()
    samples = to_mono(tts.synth(text, voice=voice_name))
    elapsed = time.perf_counter() - start
    duration = samples.size / SAMPLE_RATE
    speed = duration / elapsed if elapsed > 0 else float("inf")
    print(f"{duration:.2f} s audio in {elapsed:.3f} s {speed:.1f}x")
    return samples

def main():
    choose_language()
    choose_voice()
    sentences = load_sentences(INPUT_FILE)
    tts = load_model()
    tts.synth(WARMUP_SENTENCE, voice=VOICE)
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
