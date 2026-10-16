"""Resume the original architecture; never overwrite the source checkpoint."""
import argparse
import json
import time
from pathlib import Path
import torch
from Runtime import *

def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--config', default='TrainingConfig.json')
    Parser.add_argument('--resume', default='MedGPT.pt')
    Parser.add_argument('--new-phase', action='store_true', help='Explicitly extend a completed plan using a new schedule; retain optimizer and weights')
    Parser.add_argument('--output', default='Runs/continued.pt')
    Parser.add_argument('--legacy-step', type=int, default=1000, help='Assumed completed updates for old checkpoints without Step')
    Parser.add_argument('--max-updates', type=int, help='Stop early for smoke testing without changing the LR horizon')
    Parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    Args = Parser.parse_args()
    if Path(Args.resume).resolve() == Path(Args.output).resolve() or Path(Args.output).resolve() == Path('MedGPT.pt').resolve():
        raise ValueError('Use a separate output checkpoint')
    Output = Path(Args.output)
    BestPath = Output.with_name(Output.stem + '.best.pt')
    if any(PathName.exists() for PathName in (Output, BestPath, Output.with_suffix('.jsonl'))):
        raise FileExistsError('Output checkpoint/log already exists; choose a new output name')
    Config = json.loads(Path(Args.config).read_text(encoding='utf-8-sig'))
    for Key in ('TotalSteps', 'BatchSize', 'AccumulationSteps', 'EvaluationInterval', 'EvaluationBatches', 'SaveInterval', 'Patience'):
        if Config[Key] <= 0:
            raise ValueError(f'{Key} must be positive')
    if not 0 <= Config['WarmupSteps'] < Config['TotalSteps'] - Config.get('ScheduleStartStep', 0) or not 0 <= Config['MinLearningRate'] <= Config['LearningRate']:
        raise ValueError('Invalid learning rate schedule')
    SeedAll(Config['Seed'])
    Model, Tokenizer, Checkpoint = LoadCheckpoint(Args.resume, Args.device)
    Architecture = {Key: Checkpoint[Key] for Key in ArchitectureKeys}
    Optimizer = torch.optim.AdamW(Model.parameters(), lr=Config['LearningRate'])
    Step = Checkpoint.get('Step', Args.legacy_step)
    if Step >= Config['TotalSteps']:
        raise ValueError('Checkpoint has already reached TotalSteps; configure an explicit new phase')
    if Args.new_phase and 'OptimizerState' not in Checkpoint:
        raise ValueError('New phases require a checkpoint with optimizer state and completed steps')
    if 'OptimizerState' in Checkpoint:
        ValidateResumeConfig(Checkpoint, Config, Args.new_phase)
        Optimizer.load_state_dict(Checkpoint['OptimizerState'])
        torch.set_rng_state(Checkpoint['TorchRng'])
        if Args.device.startswith('cuda') and Checkpoint.get('CudaRng'):
            torch.cuda.set_rng_state_all(Checkpoint['CudaRng'])
    else:
        print(f'Legacy checkpoint: preserving weights, using a FRESH optimizer; assuming {Step} completed updates.')
    Datasets = LoadData(Tokenizer, Config['DataDirectory'])
    BestLoss = Checkpoint.get('BestLoss', float('inf'))
    BadEvaluations = Checkpoint.get('BadEvaluations', 0)
    if Args.new_phase:
        BadEvaluations = 0
        print(f'New phase from step {Step}: retaining weights, optimizer and RNG; LR starts at {LearningRateAt(Step, Config):.8g}')
    Output = Path(Args.output)
    Output.parent.mkdir(parents=True, exist_ok=True)
    BestPath = Output.with_name(Output.stem + '.best.pt')
    def Save(PathName):
        SaveCheckpoint(PathName, Model, Tokenizer, Architecture, Optimizer, Step, Config, BestLoss, BadEvaluations)
    def Evaluate():
        nonlocal BestLoss, BadEvaluations
        Results = EvaluateLoss(Model, Datasets, Config['BatchSize'], Config['EvaluationBatches'])
        if Results['Validation'] < BestLoss - Config['MinDelta']:
            BestLoss = Results['Validation']
            BadEvaluations = 0
            Save(BestPath)
        else:
            BadEvaluations += 1
        Record = dict(Step=Step, **Results, LearningRate=Optimizer.param_groups[0]['lr'], TokensPerSecond=Throughput,
                      PeakGpuMiB=torch.cuda.max_memory_allocated() / 2**20 if Args.device.startswith('cuda') else 0)
        print(json.dumps(Record), flush=True)
        with Output.with_suffix('.jsonl').open('a', encoding='utf-8') as File:
            File.write(json.dumps(Record) + '\n')
    Throughput = 0.0
    # Baseline evaluation must not increment the saved early-stop counter on every resume.
    if 'BestLoss' not in Checkpoint:
        Evaluate()
    StopStep = min(Config['TotalSteps'], Step + Args.max_updates) if Args.max_updates else Config['TotalSteps']
    if Args.max_updates is not None and Args.max_updates <= 0:
        raise ValueError('max-updates must be positive')
    try:
        while Step < StopStep and BadEvaluations < Config['Patience']:
            Start = time.perf_counter()
            Model.train()
            Optimizer.zero_grad(set_to_none=True)
            for Group in Optimizer.param_groups:
                Group['lr'] = LearningRateAt(Step, Config)
            for MicroStep in range(Config['AccumulationSteps']):
                Inputs, Targets = GetBatch(Datasets['Train'], Config['BatchSize'], Model.ContextSize, Args.device)
                Loss = Model(Inputs, Targets)[1] / Config['AccumulationSteps']
                if not torch.isfinite(Loss):
                    raise FloatingPointError('Non-finite loss')
                Loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), Config['GradientClip'], error_if_nonfinite=True)
            Optimizer.step()
            Step += 1
            if Args.device.startswith('cuda'):
                torch.cuda.synchronize()
            Throughput = Config['BatchSize'] * Model.ContextSize * Config['AccumulationSteps'] / (time.perf_counter() - Start)
            if Step % Config['EvaluationInterval'] == 0 or Step == StopStep:
                Evaluate()
            if Step % Config['SaveInterval'] == 0:
                Save(Output)
    except KeyboardInterrupt:
        print('Interrupted; saving completed updates (partial accumulated gradients are discarded).')
    finally:
        Save(Output)
    print(f'Saved {Output} at completed step {Step}')

if __name__ == '__main__':
    Main()
