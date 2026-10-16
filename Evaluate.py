"""Fixed sampled-window evaluation; perplexity is per character, not per word."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from Runtime import LoadCheckpoint, LoadData, EvaluateLoss, SeedAll
from Generate import GenerateText

Prompts = ['The human heart is', 'The nervous system', 'Blood flows through', 'The respiratory system', 'The function of the kidneys is']

def MemorizationCheck(Text, Corpus, Width=50):
    Windows = [Text[Index:Index + Width] for Index in range(0, max(0, len(Text) - Width + 1), 10)]
    return dict(WindowCharacters=Width, Windows=len(Windows), ExactMatches=sum(Window in Corpus for Window in Windows),
                Note='Exact overlapping windows only; not a semantic or exhaustive near-duplicate detector')

def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--checkpoints', nargs='+', default=['MedGPT.pt'])
    Parser.add_argument('--output', default='Runs/evaluation.json')
    Parser.add_argument('--data', default='Data')
    Parser.add_argument('--batches', type=int, default=20)
    Parser.add_argument('--batch-size', type=int, default=16)
    Parser.add_argument('--max-new-tokens', type=int, default=200)
    Parser.add_argument('--seed', type=int, default=123)
    Parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    Args = Parser.parse_args()
    if Args.batches <= 0 or Args.batch_size <= 0 or Args.max_new_tokens <= 0:
        raise ValueError('Evaluation sizes must be positive')
    SeedAll(Args.seed)
    Corpus = (Path(Args.data) / 'train.txt').read_text(encoding='utf-8')
    Reports = []
    for CheckpointPath in Args.checkpoints:
        Model, Tokenizer, Checkpoint = LoadCheckpoint(CheckpointPath, Args.device)
        Losses = EvaluateLoss(Model, LoadData(Tokenizer, Args.data), Args.batch_size, Args.batches, Args.seed)
        Samples = []
        for Prompt in Prompts:
            Start = time.perf_counter()
            Text = GenerateText(Model, Tokenizer, Prompt, Args.max_new_tokens, Seed=Args.seed)
            Elapsed = time.perf_counter() - Start
            Samples.append(dict(Prompt=Prompt, Text=Text, TokensPerSecond=Args.max_new_tokens / Elapsed,
                                Memorization=MemorizationCheck(Text[len(Prompt):], Corpus)))
        Reports.append(dict(Checkpoint=CheckpointPath, Step=Checkpoint.get('Step'), Seed=Args.seed, Batches=Args.batches,
                            BatchSize=Args.batch_size, CrossEntropy=Losses, CharacterPerplexity={Key: math.exp(Value) for Key, Value in Losses.items()}, Samples=Samples))
    Destination = Path(Args.output)
    Destination.parent.mkdir(parents=True, exist_ok=True)
    Destination.write_text(json.dumps(Reports, indent=2, ensure_ascii=False), encoding='utf-8')
    for Report in Reports:
        print(Report['Checkpoint'], Report['CrossEntropy'], Report['CharacterPerplexity'])

if __name__ == '__main__':
    Main()
