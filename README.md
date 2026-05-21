# GraniteCLI (Transcriber)

A lightweight transcription/translation workflow for long audio and video files using **IBM Granite 4.1-2B** with **vLLM** in Docker.

## What’s in this repo

- `transcribe.py` – CLI script that runs Granite via `vLLM`, supports:
  - transcription
  - translation to a target language
  - punctuation/capitalization prompting
  - keyword biasing
  - optional chunked processing for long audio
- `Dockerfile` – container image based on `vllm/vllm-openai`, with required audio dependencies and pre-downloaded model.
- `TranscribeVideo.ps1` – PowerShell helper that:
  - extracts mono 16kHz WAV audio with `ffmpeg`
  - runs transcription in a transient GPU Docker container
  - cleans temporary files

## Requirements

- NVIDIA GPU (CUDA-compatible)
- Docker with GPU support (`--gpus all`)
- PowerShell (for `TranscribeVideo.ps1`)
- `ffmpeg` available on host (used by the PowerShell helper)

## Build

```powershell
docker build -t granite-cli .
```

## PowerShell helper usage

Import the function:

```powershell
. .\TranscribeVideo.ps1
```

Basic transcription:

```powershell
Transcribe-Video -InputFile "your_video.mp4" -OutputFile "transcript.txt"
```

With punctuation and capitalization:

```powershell
Transcribe-Video -InputFile "your_video.mp4" -OutputFile "transcript.txt" -Punctuation
```

With chunking for very long files:

```powershell
Transcribe-Video -InputFile "your_video.mp4" -OutputFile "transcript.txt" -Chunk 30
```

With translation:

```powershell
Transcribe-Video -InputFile "french_audio.wav" -Language "English" -OutputFile "translation.txt"
```

With keyword biasing:

```powershell
Transcribe-Video -InputFile "tech_talk.mp4" -Keywords "Kubernetes, Docker, CI/CD" -OutputFile "transcript.txt"
```

## Direct Docker usage

```powershell
docker run --rm --gpus all `
    -v "${env:TEMP}:/audio" `
    -v "${PWD}:/output" `
    -v "granite_model_cache:/root/.cache/huggingface" `
    granite-cli python /app/transcribe.py "/audio/input.wav" "/output/output.txt" --punctuation --language "Spanish" --keywords "Hola"
```

## `transcribe.py` CLI options

| Flag | Description | Default |
|---|---|---|
| `--punctuation` | Prompt model for punctuation and capitalization | off |
| `--chunk SECONDS` | Chunk duration in seconds (`0` = process as one segment unless safety auto-chunking is triggered for very long audio) | `0` |
| `--max-tokens N` | Max new tokens per generation | `200` (auto-adjusted if left default) |
| `--language LANG` | Translate speech into target language | `None` |
| `--keywords KWS` | Comma-separated keywords for biasing | `None` |

## Notes

- Audio is normalized to mono and resampled to 16kHz if needed.
- For very long audio (> ~25 min), the script applies a safety fallback: if `--chunk` is `0`, it switches to 30s chunks automatically.
- Containers are run with `--rm` so VRAM and resources are released when finished.
