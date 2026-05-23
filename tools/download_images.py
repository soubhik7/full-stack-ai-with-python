import os
import sys
from urllib.request import Request, urlopen
import ssl

mapping = {
    "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*d5MiUkYVQkuCfLIEuhALGg.png": "/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/images/intro.png",
    "https://miro.medium.com/v2/resize:fit:640/0*gKZNVxDH-WQjVoBZ.png": "/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/images/next_token_predictor.png",
    "https://miro.medium.com/v2/resize:fit:1048/format:webp/0*j6n2kW3Y32VGVwJc.png": "/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/images/autoregressive_window.png",
    "https://miro.medium.com/v2/resize:fit:1260/0*01biJyKsA-ccx8In": "/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/images/rnn.png",
    "https://miro.medium.com/v2/resize:fit:1260/0*ORhUgV_p_SVCuN9R.png": "/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/images/lstm.png",
}

os.makedirs("/Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/images", exist_ok=True)

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

for url, out_path in mapping.items():
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30, context=context) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved {out_path} ({len(data)} bytes)")
    except Exception as e:
        print(f"Failed {url} -> {out_path}: {e}")
        sys.exit(1)

print("Done")
