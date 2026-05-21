import sys
import argparse
import numpy as np
import soundfile as sf
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with IBM Granite 4.1-2B using vLLM")
    parser.add_argument("input_audio", help="Path to input audio file")
    parser.add_argument("output_text", help="Path to output transcript file")
    parser.add_argument("--punctuation", action="store_true", help="Enable punctuation and capitalization")
    parser.add_argument("--chunk", type=int, default=0, help="Chunk duration in seconds (0 = no chunking)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens per generation (default: 200)")
    parser.add_argument("--language", type=str, help="Translate speech to specified language")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords for biasing")
    args = parser.parse_args()

    input_file = args.input_audio
    output_file = args.output_text
    use_punctuation = args.punctuation
    chunk_seconds = args.chunk
    max_tokens = args.max_tokens
    target_language = args.language
    keywords_list = args.keywords

    print(f"=> Initializing vLLM Offline Engine (Laptop 8GB VRAM Optimized)...", flush=True)
    # enforce_eager=True disables CUDA graph capture for instant startup and low VRAM
    # gpu_memory_utilization=0.8 restricts VRAM usage to fit on an 8GB laptop GPU
    llm = LLM(
        model="ibm-granite/granite-speech-4.1-2b",
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True
    )

    print(f"=> Loading Audio: {input_file}...", flush=True)
    wav_np, sr = sf.read(input_file, dtype='float32')
    if wav_np.ndim > 1:
        wav_np = np.mean(wav_np, axis=1)

    if sr != 16000:
        import librosa
        print("=> Resampling audio to 16kHz...", flush=True)
        wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Build prompt instructions
    if target_language:
        if use_punctuation and keywords_list:
            user_prompt = f"<|audio|>translate the speech to {target_language} with proper punctuation and capitalization. Keywords: {keywords_list}"
        elif use_punctuation:
            user_prompt = f"<|audio|>translate the speech to {target_language} with proper punctuation and capitalization."
        elif keywords_list:
            user_prompt = f"<|audio|>translate the speech to {target_language}. Keywords: {keywords_list}"
        else:
            user_prompt = f"<|audio|>translate the speech to {target_language}."
    else:
        if use_punctuation and keywords_list:
            user_prompt = f"<|audio|>transcribe the speech with proper punctuation and capitalization. Keywords: {keywords_list}"
        elif use_punctuation:
            user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
        elif keywords_list:
            user_prompt = f"<|audio|>transcribe the speech to text. Keywords: {keywords_list}"
        else:
            user_prompt = "<|audio|>can you transcribe the speech into a written format?"

    # Format the prompt using the model's standard template format via the tokenizer
    chat = [{"role": "user", "content": user_prompt}]
    prompt = llm.get_tokenizer().apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    duration_sec = len(wav_np) / 16000

    if chunk_seconds == 0 and duration_sec > 1500:
        print(f"=> Warning: Audio duration ({duration_sec:.1f}s) exceeds safe context limit (~25m). Auto-enabling 30s chunks.", flush=True)
        chunk_seconds = 30

    if max_tokens == 200:
        ref_seconds = chunk_seconds if chunk_seconds > 0 else duration_sec
        max_tokens = min(4096, max(200, int(ref_seconds * 7)))
        print(f"=> Adjusted max_tokens to {max_tokens} per segment.", flush=True)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0 # Deterministic greedy generation matching original setup
    )

    if chunk_seconds > 0:
        chunk_len_samples = chunk_seconds * 16000
        total_samples = len(wav_np)
        
        chunks = []
        for i in range(0, total_samples, chunk_len_samples):
            chunk = wav_np[i : i + chunk_len_samples]
            if len(chunk) >= 1600:  # At least 0.1s
                chunks.append(chunk)
                
        total_chunks = len(chunks)
        print(f"=> Splitting audio into {total_chunks} chunks of {chunk_seconds}s...", flush=True)
        
        inputs_list = []
        for chunk in chunks:
            inputs_list.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": (chunk, 16000)}
            })
            
        print(f"=> Submitting {total_chunks} chunks to vLLM batch engine...", flush=True)
        outputs = llm.generate(inputs_list, sampling_params=sampling_params)
        full_transcript = " ".join([o.outputs[0].text.strip() for o in outputs if o.outputs[0].text.strip()])
    else:
        print(f"=> Submitting full audio to vLLM engine...", flush=True)
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"audio": (wav_np, 16000)}
        }
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        full_transcript = outputs[0].outputs[0].text.strip()

    print(f"=> Saving transcript to {output_file}", flush=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_transcript)

if __name__ == "__main__":
    main()
