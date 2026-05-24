from regex_tokenizer import RegexTokenizer
from serialization import save_tokenizer
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
file_path = project_root / "datasets" / "taylorswift.txt"

tokenizer = RegexTokenizer()
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size = 1024
tokenizer.train(text, vocab_size)

save_tokenizer(tokenizer, "checkpoints/tokenizer/gpt4tokenizer")