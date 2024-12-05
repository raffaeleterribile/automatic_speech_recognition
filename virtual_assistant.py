""" Virtual Assistant Demo """
import sys
import torch
import requests
from IPython.display import Audio
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from huggingface_hub import HfFolder

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
	"audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device=DEVICE
)

print(classifier.model.config.id2label[27]) # 30 = "Sheila", 27 = "Marvin"

def launch_fn(
	wake_word="sheila", # previously was "marvin"
	prob_threshold=0.5,
	chunk_length_s=2.0,
	stream_chunk_s=0.25,
	debug=False,
):
	"""  Launches a live audio stream and listens for a wake word. """
	if wake_word not in classifier.model.config.label2id.keys():
		raise ValueError(
			f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
		)

	sampling_rate = classifier.feature_extractor.sampling_rate

	mic = ffmpeg_microphone_live(
		sampling_rate=sampling_rate,
		chunk_length_s=chunk_length_s,
		stream_chunk_s=stream_chunk_s,
	)

	print("Listening for wake word...")
	for prediction in classifier(mic):
		prediction = prediction[0]
		if debug:
			print(prediction)
		if prediction["label"] == wake_word:
			if prediction["score"] > prob_threshold:
				return True

transcriber = pipeline(
	"automatic-speech-recognition", model="openai/whisper-tiny", device=DEVICE
)

def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
	"""  Transcribes live audio stream. """
	sampling_rate = transcriber.feature_extractor.sampling_rate

	mic = ffmpeg_microphone_live(
		sampling_rate=sampling_rate,
		chunk_length_s=chunk_length_s,
		stream_chunk_s=stream_chunk_s,
	)

	print("Start speaking...")
	text = ""
	for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128, "language": "it"}):
		text += item["text"]
		sys.stdout.write("\033[K")
		print(item["text"], end="\r")
		if not item["partial"][0]:
			break

	return text

def query(text, model_id="tiiuae/falcon-7b-instruct"):
	"""  Queries a text-to-text model. """
	api_url = f"https://api-inference.huggingface.co/models/{model_id}"
	headers = {"Authorization": f"Bearer {HfFolder().get_token()}"}
	payload = {"inputs": text}

	print(f"Querying...: {text}")
	result = requests.post(api_url, headers=headers, json=payload, timeout=60)
	return result.json()[0]["generated_text"][len(text) + 1 :]

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def synthesise(text):
	"""  Synthesises speech from text. """
	inputs = processor(text=text, return_tensors="pt")
	speech = model.generate_speech(
		inputs["input_ids"].to(DEVICE), speaker_embeddings.to(DEVICE), vocoder=vocoder
	)
	return speech.cpu()

# Original demo code
# launch_fn(debug=True)
# transcribe()
# query("What does Hugging Face do?")
# audio = synthesise(
# 	"Hugging Face is a company that provides natural language processing and machine learning tools for developers."
# )
# Audio(audio, rate=16000)

launch_fn()
transcription = transcribe()
response = query(transcription)
audio = synthesise(response)

Audio(audio, rate=16000, autoplay=True)
