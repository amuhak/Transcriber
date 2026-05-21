# GraniteCLI

A specialized tool for transcribing long audio and video files using **IBM Granite 4.1-2B**, containerized for easy deployment and optimized for NVIDIA GPUs (e.g., RTX 3090).

## Project Overview

- **Core Technology:** [IBM Granite 4.1-2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) via Hugging Face Transformers.
- **Architecture:**
    - `transcribe.py`: Python script utilizing `AutoProcessor` and `AutoModelForSpeechSeq2Seq` with prompt-conditioned ASR.
    - `Dockerfile`: Containerizes the environment with CUDA 12.4 and stable PyTorch 2.5+.
    - `TranscribeVideo.ps1`: PowerShell helper to automate audio extraction via `ffmpeg` and transient execution within Docker.

## Setup & Building

### 1. Build the Docker Image
```powershell
docker build -t granite-cli .
```

### 2. Prepare PowerShell Environment
Import the helper function from `TranscribeVideo.ps1`:
```powershell
. .\TranscribeVideo.ps1
```

## Usage

### Using the PowerShell Helper
The `Transcribe-Video` function automates the entire process (extracting audio, running Docker, and cleaning up).

#### Basic transcription
```powershell
Transcribe-Video -InputFile "your_video.mp4" -OutputFile "transcript.txt"
```

#### With punctuation and capitalization
```powershell
Transcribe-Video -InputFile "your_video.mp4" -OutputFile "transcript.txt" -Punctuation
```

#### With chunking for very long files
```powershell
Transcribe-Video -InputFile "your_video.mp4" -OutputFile "transcript.txt" -Chunk 30
```

#### With Translation (Automatic Speech Translation)
```powershell
Transcribe-Video -InputFile "french_audio.wav" -Language "English" -OutputFile "translation.txt"
```

#### With Keyword Biasing
```powershell
Transcribe-Video -InputFile "tech_talk.mp4" -Keywords "Kubernetes, Docker, CI/CD" -OutputFile "transcript.txt"
```

### Direct Docker Execution
If you prefer running the container manually:
```powershell
docker run --rm --gpus all `
    -v "${env:TEMP}:/audio" `
    -v "${PWD}:/output" `
    -v "granite_model_cache:/root/.cache/huggingface" `
    granite-cli python /app/transcribe.py "/audio/input.wav" "/output/output.txt" --punctuation --language "Spanish" --keywords "Hola"
```

### CLI Options (transcribe.py)
| Flag | Description | Default |
|---|---|---|
| `--punctuation` | Enable punctuation and truecasing in output | off |
| `--chunk SECONDS` | Chunk audio into segments (0 = no chunking) | 0 |
| `--max-tokens N` | Max new tokens per generation | 200 |
| `--language LANG` | Translate speech to specified language (e.g., English, French, German) | None |
| `--keywords KWS` | Comma-separated keywords for biasing | None |

## Development Conventions

- **Hardware Targeting:** Configured for NVIDIA GPUs (`device="cuda"`, `torch.bfloat16`).
- **Transient Lifecycle:** Docker containers are started on-demand (`--rm`) and shut down immediately after use to release VRAM.
- **Memory Management:** No chunking by default (128k context handles ~30min+ audio). Use `--chunk` for very long files.
- **Self-Contained Image:** The ~4GB model is downloaded and baked directly into the Docker image during the build process, making the image 100% portable and offline-ready.
