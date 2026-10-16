import argparse
import math
import torch
from Runtime import LoadCheckpoint, SeedAll

def FilterLogits(Logits, History, Temperature=0.8, TopK=20, TopP=1.0, RepetitionPenalty=1.0):
    if not math.isfinite(Temperature) or Temperature <= 0 or not 0 < TopP <= 1 or not math.isfinite(RepetitionPenalty) or RepetitionPenalty < 1 or TopK < 0:
        raise ValueError('Require temperature > 0, top-k >= 0, 0 < top-p <= 1, repetition penalty >= 1')
    Scores = Logits.clone()
    # Sign-aware penalty reduces both positive and negative seen-token logits.
    Seen = History.unique()
    Scores[:, Seen] = torch.where(Scores[:, Seen] < 0, Scores[:, Seen] * RepetitionPenalty, Scores[:, Seen] / RepetitionPenalty)
    Scores /= Temperature
    if TopK:
        Values, Indices = Scores.topk(min(TopK, Scores.size(-1)))
        Scores = torch.full_like(Scores, float('-inf')).scatter(1, Indices, Values)
    if TopP < 1:
        Values, Indices = Scores.sort(descending=True)
        Remove = Values.softmax(-1).cumsum(-1) > TopP
        # Keep the first token crossing the probability threshold, including top-1.
        Remove[:, 1:] = Remove[:, :-1].clone()
        Remove[:, 0] = False
        Values = Values.masked_fill(Remove, float('-inf'))
        Scores = torch.full_like(Scores, float('-inf')).scatter(1, Indices, Values)
    return Scores

@torch.inference_mode()
def GenerateText(Model, Tokenizer, Prompt, MaxNewTokens=300, Temperature=0.8, TopK=20, TopP=1.0, RepetitionPenalty=1.0, Greedy=False, Seed=42):
    if not Prompt or MaxNewTokens < 0:
        raise ValueError('Provide a nonempty prompt and nonnegative generation length')
    Device = next(Model.parameters()).device
    InputTensor = torch.tensor([Tokenizer.Encode(Prompt)], dtype=torch.long, device=Device)
    Generator = torch.Generator(device=Device).manual_seed(Seed)
    WasTraining = Model.training
    Model.eval()
    try:
        for Index in range(MaxNewTokens):
            Context = InputTensor[:, -Model.ContextSize:]
            Scores = FilterLogits(Model(Context)[0][:, -1, :], Context, Temperature, TopK, TopP, RepetitionPenalty)
            Token = Scores.argmax(-1, keepdim=True) if Greedy else torch.multinomial(Scores.softmax(-1), 1, generator=Generator)
            InputTensor = torch.cat((InputTensor, Token), dim=1)
        return Tokenizer.Decode(InputTensor[0].tolist())
    finally:
        Model.train(WasTraining)

def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--checkpoint', default='MedGPT.pt')
    Parser.add_argument('--prompt')
    Parser.add_argument('--max-new-tokens', type=int, default=300)
    Parser.add_argument('--temperature', type=float, default=0.8)
    Parser.add_argument('--top-k', type=int, default=20)
    Parser.add_argument('--top-p', type=float, default=1.0)
    Parser.add_argument('--repetition-penalty', type=float, default=1.0)
    Parser.add_argument('--greedy', action='store_true')
    Parser.add_argument('--compare', action='store_true')
    Parser.add_argument('--seed', type=int, default=42)
    Parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    Args = Parser.parse_args()
    SeedAll(Args.seed)
    Model, Tokenizer, _ = LoadCheckpoint(Args.checkpoint, Args.device)
    Settings = dict(Temperature=Args.temperature, TopK=Args.top_k, TopP=Args.top_p, RepetitionPenalty=Args.repetition_penalty, Greedy=Args.greedy)
    while True:
        try:
            Prompt = Args.prompt if Args.prompt is not None else input('Prompt (exit to quit): ')
            if Prompt.lower() == 'exit':
                break
            Variants = [Settings] if not Args.compare else [dict(Greedy=True, TopK=0, Temperature=1), Settings, dict(Temperature=0.9, TopK=0, TopP=0.9)]
            for Variant in Variants:
                print(Variant)
                print(GenerateText(Model, Tokenizer, Prompt, Args.max_new_tokens, Seed=Args.seed, **Variant))
        except ValueError as Error:
            print(Error)
        except (EOFError, KeyboardInterrupt):
            break
        if Args.prompt is not None:
            break

if __name__ == '__main__':
    Main()
