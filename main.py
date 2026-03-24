import whisperx
import sounddevice as sd
import numpy as np
import queue
from jiwer import wer
from difflib import SequenceMatcher
import warnings
import logging

# ------------------ CLEAN LOGS ------------------
warnings.filterwarnings("ignore")
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)

# ------------------ SETTINGS ------------------
device = "cpu"
compute_type = "int8"
samplerate = 16000
chunk_duration = 6

# ------------------ LOAD MODEL ------------------
print("Loading model...")
model = whisperx.load_model(
    "kurianbenoy/vegam-whisper-medium-ml",
    device,
    compute_type=compute_type
)

print("🎤 Malayalam speech recognition started...")
print("⏹ Press Ctrl+C to stop\n")

# ------------------ AUDIO QUEUE ------------------
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# ------------------ MIC ------------------
stream = sd.InputStream(
    samplerate=samplerate,
    channels=1,
    blocksize=1024,
    callback=callback
)

# ------------------ REFERENCE ------------------
reference_text = """എല്ലാവർക്കും നമസ്കാരം, വായന മനുഷ്യന്റെ ജീവിതത്തിൽ അത്യന്തം പ്രധാനപ്പെട്ട ഒന്നാണ്. വായന നമ്മുക്ക് അറിവും ചിന്താശേഷിയും നൽകുന്നു."""

# ------------------ STORAGE ------------------
full_transcript = ""
confidence_list = []   # ⭐ store all chunk confidences

# ------------------ UTIL FUNCTIONS ------------------

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\u0D00-\u0D7F\s]', '', text)  # keep Malayalam only
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_transcript(text):
    words = text.split()
    unique = []
    for w in words:
        if w not in unique:
            unique.append(w)
    return " ".join(unique)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_same_text(ref, hyp, threshold=0.65):   # 🔥 change 0.85 → 0.65
    ref = clean_text(ref)
    hyp = clean_text(hyp)

    sim = similarity(ref, hyp)
    length_ratio = len(hyp) / (len(ref) + 1e-8)

    return sim > threshold and 0.7 < length_ratio < 1.3

def char_accuracy(ref, hyp):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, ref, hyp).ratio() * 100

def calculate_wer(reference, hypothesis):
    reference = clean_text(reference)
    hypothesis = clean_text(hypothesis)

    error = wer(reference, hypothesis)
    error = min(error, 1.0)
    accuracy = (1 - error) * 100
    return error, accuracy

def calculate_ber(reference, hypothesis):
    length = min(len(reference), len(hypothesis))
    errors = sum(1 for i in range(length) if reference[i] != hypothesis[i])
    errors += abs(len(reference) - len(hypothesis))
    return errors / len(reference)


def get_confidence(result):
    if "segments" not in result:
        return 0

    scores = []
    for seg in result["segments"]:
        if "avg_logprob" in seg:
            scores.append(seg["avg_logprob"])

    if not scores:
        return 0

    avg_logprob = sum(scores) / len(scores)

    # realistic mapping using sigmoid (BEST)
    import math
    confidence = 100 / (1 + math.exp(-3 * (avg_logprob + 0.7)))

    return confidence

def count_tokens(text):
    return len(text.split())

# ------------------ START ------------------
print("🎤 Listening...\n")

try:
    with stream:
        while True:
            audio_data = []

            for _ in range(int(samplerate / 1024 * chunk_duration)):
                audio_data.append(audio_queue.get())

            audio_np = np.concatenate(audio_data, axis=0).flatten()

            # normalize
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)

            # TRANSCRIBE
            result = model.transcribe(
                audio_np,
                language="ml",
                task="transcribe"
            )

            if "segments" in result:
                text = " ".join([seg["text"] for seg in result["segments"]])
            else:
                text = ""

            # Malayalam filter
            text_ml = "".join([c for c in text if '\u0D00' <= c <= '\u0D7F' or c == " "]).strip()

            if text_ml:
                tokens = count_tokens(text_ml)
                confidence = get_confidence(result)

                print(f"📝 {text_ml}")
                print(f"🔢 Tokens: {tokens}")
                print(f"🎯 Confidence: {confidence:.2f}%")
                print("-" * 40)

                # save
                with open("transcript.txt", "a", encoding="utf-8") as f:
                    f.write(text_ml + "\n")

                full_transcript += " " + text_ml
                confidence_list.append(confidence)   # ⭐ store

except KeyboardInterrupt:
    print("\n🛑 Stopped")

    full_transcript = full_transcript.strip()
    full_transcript = clean_transcript(full_transcript)

    print("\n📄 FINAL TRANSCRIPT:\n")
    print(full_transcript)

    total_tokens = count_tokens(full_transcript)
    print(f"\n🔢 Total Tokens: {total_tokens}")

    # ⭐ AVERAGE CONFIDENCE
    if confidence_list:
        avg_conf = sum(confidence_list) / len(confidence_list)
        print(f"🎯 Confidence: {avg_conf:.2f}%")

    # -------- SMART EVALUATION --------
    # -------- SMART EVALUATION --------
print("\n📊 EVALUATION:\n")

ref = clean_text(reference_text)
hyp = clean_text(full_transcript)

# ⭐ CHECK if same text
if is_same_text(ref, hyp, threshold=0.75):

    print("📖 Reference text detected\n")

    # WER
    error, accuracy = calculate_wer(ref, hyp)
    print(f"📉 WER: {error:.3f}")
    print(f"✅ Word Accuracy: {accuracy:.2f}%")

    # Character Accuracy
    char_acc = char_accuracy(ref, hyp)
    print(f"🔤 Character Accuracy: {char_acc:.2f}%")

    # BER
    ber = calculate_ber(ref, hyp)
    print(f"📊 BER: {ber:.3f}")

    # ⭐ FINAL ACCURACY (best representation)
    final_acc = (accuracy * 0.4) + (char_acc * 0.6)
    print(f"🎯 Final Accuracy: {final_acc:.2f}%")

else:
    print("🧠 Free speech detected")
    print("📊 WER, BER and CER not applicable")
    print(f"🎯 Confidence: {avg_conf:.2f}%")