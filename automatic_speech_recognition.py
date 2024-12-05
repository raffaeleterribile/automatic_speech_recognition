""" Automatic Speech Recognition with 🤗 Transformers and OpenAI's Whisper """
import torch
from transformers import pipeline
import gradio as gr

MODEL_NAME = "openai/whisper-tiny"
BATCH_SIZE = 8

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
	task="automatic-speech-recognition",
	model=MODEL_NAME,
	chunk_length_s=30,
	device=DEVICE
)


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
	""" Format a timestamp in seconds to a human-readable format. """
	if seconds is not None:
		milliseconds = round(seconds * 1000.0)

		hours = milliseconds // 3_600_000
		milliseconds -= hours * 3_600_000

		minutes = milliseconds // 60_000
		milliseconds -= minutes * 60_000

		seconds = milliseconds // 1_000
		milliseconds -= seconds * 1_000

		hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
		return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
	else:
		# we have a malformed timestamp so just return it as is
		return seconds

def transcribe(file, task, return_timestamps):
	""" Transcribe audio file. """
	outputs = pipe(file, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=return_timestamps)
	text = outputs["text"]
	if return_timestamps:
		timestamps = outputs["chunks"]
		timestamps = [
			f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
			for chunk in timestamps
		]
		text = "\n".join(str(feature) for feature in timestamps)
	return text

demo = gr.Blocks()

mic_transcribe = gr.Interface(
	fn=transcribe,
	inputs=[
		gr.Audio(sources="microphone", type="filepath"),
		gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
		gr.Checkbox(value=False, label="Return timestamps"),
	],
	outputs="text",
	theme="default",
	title="Whisper Demo: Transcribe Audio",
	description=(
		"Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
		f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
		" of arbitrary length."
	),
	flagging_mode="never",
)

file_transcribe = gr.Interface(
	fn=transcribe,
	inputs=[
		gr.Audio(sources="upload", label="Audio file"),
		gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
		gr.Checkbox(value=False, label="Return timestamps"),
	],
	outputs="text",
	theme="default",
	title="Whisper Demo: Transcribe Audio",
	description=(
		"Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
		f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
		" of arbitrary length."
	),
	examples=[
		["./example.flac", "transcribe", False],
		["./example.flac", "transcribe", True],
	],
	cache_examples=True,
	flagging_mode="never",
)

demo = gr.TabbedInterface([mic_transcribe, file_transcribe], ["Transcribe Microphone", "Transcribe Audio File"])

demo.launch()
