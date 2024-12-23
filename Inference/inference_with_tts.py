## 감정TTS.py
## pip install git+https://github.com/myshell-ai/MeloTTS.git
## python -m unidic download

from melo.api import TTS

# Set TTS parameters
text = "이것은 화난 목소리로 말하는 테스트입니다. 화난 감정을 표현하세요."
output_path = "korean_angry.wav"  # Output file path for generated audio
language = "KR"  # Specify language
speed = 1.0  # Set the speaking speed
device = "auto"  # Automatically selects GPU if available, otherwise CPU

# Initialize the TTS model
model = TTS(language=language, device=device)
speaker_ids = model.hps.data.spk2id  # Retrieve available speaker IDs

# Check available speaker IDs
print("Available speaker IDs:", speaker_ids)

# Access a specific speaker ID for Korean (adjust based on your dataset/model configuration)
korean_speaker = speaker_ids["KR"]  # Replace with the correct key for Korean speakers if different

# Perform TTS inference and save the output audio
model.tts_to_file(text, korean_speaker, output_path, speed=speed)

print(f"Audio file saved at {output_path}")