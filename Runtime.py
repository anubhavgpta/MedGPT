"""Shared FP32 training, checkpoint and evaluation primitives."""
from pathlib import Path
import math
import random
import torch
from Model import MedGPT
from Tokenizer import MedicalTokenizer

ArchitectureKeys = ('VocabSize', 'EmbeddingSize', 'NumHeads', 'NumLayers', 'ContextSize')

def SeedAll(Seed):
    random.seed(Seed)
    torch.manual_seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed)
    torch.set_num_threads(4)

def LoadCheckpoint(PathName, Device='cpu'):
    Checkpoint = torch.load(PathName, map_location='cpu', weights_only=True)
    Tokenizer = MedicalTokenizer(Characters=Checkpoint['Characters'])
    if Tokenizer.VocabSize != Checkpoint['VocabSize']:
        raise ValueError('Checkpoint vocabulary size mismatch')
    Model = MedGPT(**{Key: Checkpoint[Key] for Key in ArchitectureKeys}).to(Device)
    Model.load_state_dict(Checkpoint['ModelState'], strict=True)
    return Model, Tokenizer, Checkpoint

def SaveCheckpoint(PathName, Model, Tokenizer, Architecture, Optimizer, Step, Config, BestLoss, BadEvaluations):
    Destination = Path(PathName)
    Destination.parent.mkdir(parents=True, exist_ok=True)
    State = dict(Architecture, FormatVersion=2, ModelState=Model.state_dict(),
                 Characters=Tokenizer.Characters, OptimizerState=Optimizer.state_dict(),
                 Step=Step, Config=Config, BestLoss=BestLoss, BadEvaluations=BadEvaluations,
                 TorchRng=torch.get_rng_state(),
                 CudaRng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    Temporary = Destination.with_suffix('.tmp')
    torch.save(State, Temporary)
    Temporary.replace(Destination)

def LoadData(Tokenizer, Directory):
    return {Name: torch.tensor(Tokenizer.Encode((Path(Directory) / File).read_text(encoding='utf-8')), dtype=torch.long)
            for Name, File in [('Train', 'train.txt'), ('Validation', 'validation.txt')]}

def GetBatch(Data, BatchSize, ContextSize, Device, Generator=None):
    if len(Data) <= ContextSize:
        raise ValueError('Dataset must exceed context length')
    Starts = torch.randint(len(Data) - ContextSize, (BatchSize,), generator=Generator)
    # A single gather replaces Python slicing; targets are shifted exactly one character.
    Indices = Starts[:, None] + torch.arange(ContextSize + 1)
    Windows = Data[Indices].to(Device)
    return Windows[:, :-1], Windows[:, 1:]

@torch.no_grad()
def EvaluateLoss(Model, Datasets, BatchSize=16, Batches=20, Seed=123):
    WasTraining = Model.training
    Model.eval()
    Results = {}
    try:
        for Name, Data in Datasets.items():
            Generator = torch.Generator().manual_seed(Seed)
            Total = 0.0
            for Index in range(Batches):
                Inputs, Targets = GetBatch(Data, BatchSize, Model.ContextSize, next(Model.parameters()).device, Generator)
                Total += Model(Inputs, Targets)[1].item()
            Results[Name] = Total / Batches
    finally:
        Model.train(WasTraining)
    return Results

def LearningRateAt(Step, Config):
    # Step is the zero-based absolute optimizer update, never restarted on resume.
    Start = Config.get('ScheduleStartStep', 0)
    Step = max(0, Step - Start)
    Warmup = Config['WarmupSteps']
    if Step < Warmup:
        return Config['LearningRate'] * (Step + 1) / max(1, Warmup)
    Progress = min(1.0, max(0.0, (Step - Warmup) / max(1, Config['TotalSteps'] - Start - Warmup - 1)))
    return Config['MinLearningRate'] + 0.5 * (Config['LearningRate'] - Config['MinLearningRate']) * (1 + math.cos(math.pi * Progress))

def ValidateResumeConfig(Checkpoint, Config, NewPhase=False):
    Previous = Checkpoint['Config']
    if not NewPhase:
        if Previous != Config:
            raise ValueError('Resume configuration differs; use the saved config or explicitly start --new-phase')
        return
    Step = Checkpoint['Step']
    if Step < Previous['TotalSteps']:
        raise ValueError('Finish the previous training plan before starting a new phase')
    if Config.get('ScheduleStartStep') != Step or Config['TotalSteps'] <= Step:
        raise ValueError('New phase must start at the checkpoint step and extend TotalSteps')
    Allowed = {'ScheduleStartStep', 'TotalSteps', 'LearningRate', 'MinLearningRate', 'WarmupSteps'}
    if any(Previous.get(Key) != Config.get(Key) for Key in (Previous.keys() | Config.keys()) - Allowed):
        raise ValueError('A continuation phase may change only schedule settings')
