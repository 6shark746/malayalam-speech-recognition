import re
import csv
import os

# 1. Configuration (Matches your VS Code sidebar exactly)
input_txt = "transcripts.txt"  # The file with (07_F_set_02_156 "text")
output_csv = "train.csv"
audio_folder = "audio"         # The folder containing your .wav files

# Regex to capture: (filename "transcription")
# This handles the exact format seen in your screenshots
pattern = r"\(([\w\d_]+)\s+\"(.+)\"\)"

data_rows = []
missing_files = 0

print(f"--- Starting Conversion ---")
print(f"Checking for audio files in: {os.path.abspath(audio_folder)}")

if not os.path.exists(input_txt):
    print(f"ERROR: {input_txt} not found in current directory!")
else:
    with open(input_txt, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            match = re.search(pattern, line)
            if match:
                file_id = match.group(1)
                transcription = match.group(2)
                
                # Construct the path to the wav file
                audio_path = f"{audio_folder}/{file_id}.wav"
                
                # Verify the file physically exists before adding to CSV
                if os.path.exists(audio_path):
                    data_rows.append([audio_path, transcription])
                else:
                    # Optional: Print which files are missing to help you debug
                    # print(f"Missing: {audio_path}")
                    missing_files += 1
            else:
                print(f"Skipping line {line_num} (Format mismatch): {line[:50]}...")

    # 2. Write the verified pairs to train.csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription"]) # Headers
        writer.writerows(data_rows)

    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Total entries in {output_csv}: {len(data_rows)}")
    print(f"Files skipped because audio was not found: {missing_files}")
    print("-" * 30)