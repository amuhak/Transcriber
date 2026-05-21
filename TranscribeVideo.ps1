function Transcribe-Video {
    <#
    .SYNOPSIS
        Transcribes a video or audio file using IBM Granite 4.1-2B in a transient Docker container.

    .PARAMETER InputFile
        The path to the source video or audio file.

    .PARAMETER OutputFile
        The path for the resulting transcript. Defaults to the input filename with '_transcript.txt'.

    .PARAMETER Punctuation
        Enable punctuation and capitalization in the transcript.

    .PARAMETER Chunk
        Chunk duration in seconds for processing long files (0 = no chunking). If the file is longer than 25 minutes, it defaults to 30s.

    .PARAMETER Language
        Target language for translation (e.g., 'English', 'French', 'German'). If omitted, performs transcription in the original language.

    .PARAMETER Keywords
        Comma-separated list of keywords for biasing the recognition.
    #>
    param (
        [Parameter(Mandatory=$true, Position=0)]
        [string]$InputFile,

        [Parameter(Mandatory=$false, Position=1)]
        [string]$OutputFile,

        [Parameter(Mandatory=$false)]
        [switch]$Punctuation,

        [Parameter(Mandatory=$false)]
        [int]$Chunk = 0,

        [Parameter(Mandatory=$false)]
        [string]$Language,

        [Parameter(Mandatory=$false)]
        [string]$Keywords
    )

    if (-not (Test-Path $InputFile)) {
        Write-Error "Input file '$InputFile' not found."
        return
    }

    if ([string]::IsNullOrEmpty($OutputFile)) {
        $OutputFile = [System.IO.Path]::GetFileNameWithoutExtension($InputFile) + "_transcript.txt"
    } elseif (-not $OutputFile.EndsWith(".txt")) {
        $OutputFile += ".txt"
    }

    $TempFileName = "temp_$([System.IO.Path]::GetRandomFileName()).wav"
    $TempWav = Join-Path $env:TEMP $TempFileName

    try {
        Write-Host "=> Extracting audio to temporary buffer..." -ForegroundColor Cyan
        & ffmpeg -i "$InputFile" -y -vn -acodec pcm_s16le -ar 16000 -ac 1 "$TempWav" -hide_banner -loglevel error
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed to extract audio." }

        Write-Host "=> Starting Granite Transcription (Docker)..." -ForegroundColor Cyan
        Write-Host "=> Model: ibm-granite/granite-speech-4.1-2b" -ForegroundColor Gray

        $pythonArgs = @("/app/transcribe.py", "/audio/$TempFileName", "/output/$OutputFile")
        if ($Punctuation) {
            $pythonArgs += "--punctuation"
        }
        if ($Chunk -gt 0) {
            $pythonArgs += "--chunk", "$Chunk"
        }
        if (-not [string]::IsNullOrEmpty($Language)) {
            $pythonArgs += "--language", "$Language"
        }
        if (-not [string]::IsNullOrEmpty($Keywords)) {
            $pythonArgs += "--keywords", "$Keywords"
        }

        docker run --rm --gpus all `
            -v "$($env:TEMP):/audio" `
            -v "${PWD}:/output" `
            granite-cli python3 @pythonArgs

        if ($LASTEXITCODE -ne 0) { throw "Docker transcription failed." }

        Write-Host "=> Success! Transcript saved to: $OutputFile" -ForegroundColor Green

    } catch {
        Write-Error "Transcription failed: $_"
    } finally {
        if (Test-Path $TempWav) {
            Write-Host "=> Releasing resources..." -ForegroundColor Gray
            Remove-Item $TempWav -ErrorAction SilentlyContinue
        }
    }
}
