import tempfile
import unittest
from pathlib import Path
import torch
from Model import MedGPT
from Tokenizer import MedicalTokenizer
from Runtime import GetBatch, LoadCheckpoint, SaveCheckpoint, LearningRateAt, SeedAll, ValidateResumeConfig
from Generate import FilterLogits, GenerateText

class InvariantTests(unittest.TestCase):
    def setUp(self):
        SeedAll(42)
        self.Architecture = dict(VocabSize=5, EmbeddingSize=16, NumHeads=4, NumLayers=2, ContextSize=8)
        self.Model = MedGPT(**self.Architecture)
        self.Tokenizer = MedicalTokenizer(Characters=list('edcba'))

    def test_CausalAndBackward(self):
        Inputs = torch.randint(5, (2, 8))
        Changed = Inputs.clone()
        Changed[:, 4:] = (Changed[:, 4:] + 1) % 5
        torch.testing.assert_close(self.Model(Inputs)[0][:, :4], self.Model(Changed)[0][:, :4])
        Optimizer = torch.optim.AdamW(self.Model.parameters())
        Before = self.Model.TokenEmbedding.weight.detach().clone()
        Loss = self.Model(Inputs, Changed)[1]
        Loss.backward()
        self.assertTrue(all(Parameter.grad is not None and torch.isfinite(Parameter.grad).all() for Parameter in self.Model.parameters()))
        Optimizer.step()
        self.assertFalse(torch.equal(Before, self.Model.TokenEmbedding.weight))

    def test_ShiftAndSchedule(self):
        Inputs, Targets = GetBatch(torch.arange(20), 4, 8, 'cpu')
        torch.testing.assert_close(Inputs + 1, Targets)
        Config = dict(WarmupSteps=2, TotalSteps=10, LearningRate=0.001, MinLearningRate=0.0001)
        self.assertAlmostEqual(LearningRateAt(0, Config), 0.0005)
        self.assertAlmostEqual(LearningRateAt(1, Config), 0.001)
        self.assertAlmostEqual(LearningRateAt(9, Config), 0.0001)

    def test_ContinuationPhase(self):
        Previous = dict(TotalSteps=10000, WarmupSteps=200, LearningRate=0.0003, MinLearningRate=0.00003, BatchSize=16)
        Config = dict(Previous, TotalSteps=20000, ScheduleStartStep=10000, WarmupSteps=0, LearningRate=0.00003, MinLearningRate=0.00001)
        Checkpoint = dict(Config=Previous, Step=10000)
        ValidateResumeConfig(Checkpoint, Config, True)
        self.assertAlmostEqual(LearningRateAt(10000, Config), 0.00003)
        self.assertAlmostEqual(LearningRateAt(19999, Config), 0.00001)
        self.assertGreater(LearningRateAt(12000, Config), LearningRateAt(17000, Config))
        ValidateResumeConfig(dict(Config=Config, Step=10020), Config)
        with self.assertRaises(ValueError):
            ValidateResumeConfig(Checkpoint, Config)
        with self.assertRaises(ValueError):
            ValidateResumeConfig(dict(Checkpoint, Step=9999), Config, True)
        with self.assertRaises(ValueError):
            ValidateResumeConfig(Checkpoint, dict(Config, BatchSize=8), True)
        with self.assertRaises(ValueError):
            ValidateResumeConfig(Checkpoint, dict(Config, ScheduleStartStep=0), True)

    def test_LossAndTinyOverfit(self):
        Inputs = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
        Targets = (Inputs + 1) % 5
        Logits, Initial = self.Model(Inputs, Targets)
        Expected = torch.nn.functional.cross_entropy(Logits.reshape(-1, 5), Targets.reshape(-1))
        torch.testing.assert_close(Initial, Expected)
        Optimizer = torch.optim.AdamW(self.Model.parameters(), lr=0.01)
        for Index in range(40):
            Optimizer.zero_grad()
            self.Model(Inputs, Targets)[1].backward()
            Optimizer.step()
        self.assertLess(self.Model(Inputs, Targets)[1].item(), Initial.item() * 0.2)

    def test_CheckpointAndOptimizer(self):
        Optimizer = torch.optim.AdamW(self.Model.parameters())
        Inputs = torch.randint(5, (2, 8))
        self.Model(Inputs, Inputs)[1].backward()
        Optimizer.step()
        with tempfile.TemporaryDirectory() as Directory:
            Destination = Path(Directory) / 'test.pt'
            SaveCheckpoint(Destination, self.Model, self.Tokenizer, self.Architecture, Optimizer, 1, {}, 2.0, 0)
            Loaded, Tokenizer, Checkpoint = LoadCheckpoint(Destination)
            torch.testing.assert_close(self.Model(Inputs)[0], Loaded(Inputs)[0], rtol=0, atol=0)
            self.assertEqual(Tokenizer.Characters, list('edcba'))
            Restored = torch.optim.AdamW(Loaded.parameters())
            Restored.load_state_dict(Checkpoint['OptimizerState'])
            for Model, Opt in [(self.Model, Optimizer), (Loaded, Restored)]:
                Opt.zero_grad()
                Model(Inputs, Inputs)[1].backward()
                Opt.step()
            for Left, Right in zip(self.Model.parameters(), Loaded.parameters()):
                torch.testing.assert_close(Left, Right, rtol=0, atol=0)
            Legacy = {Key: Checkpoint[Key] for Key in (*self.Architecture, 'Characters', 'ModelState')}
            torch.save(Legacy, Destination)
            LoadCheckpoint(Destination)

    def test_Sampling(self):
        Scores = torch.tensor([[3., 2., -1., -2., 0.]])
        Filtered = FilterLogits(Scores, torch.tensor([[0, 2]]), 1, 0, 1, 2)
        self.assertEqual(Filtered[0, 0].item(), 1.5)
        self.assertEqual(Filtered[0, 2].item(), -2)
        Top = FilterLogits(Scores, torch.tensor([[4]]), 1, 2, 1)
        self.assertEqual(torch.isfinite(Top).sum().item(), 2)
        Nucleus = FilterLogits(torch.log(torch.tensor([[0.5, 0.3, 0.2]])), torch.tensor([[0]]), 1, 0, 0.6)
        self.assertEqual(torch.isfinite(Nucleus).sum().item(), 2)
        self.assertEqual(GenerateText(self.Model, self.Tokenizer, 'abc', 10), GenerateText(self.Model, self.Tokenizer, 'abc', 10))
        with self.assertRaises(ValueError):
            GenerateText(self.Model, self.Tokenizer, '?')
        with self.assertRaises(ValueError):
            GenerateText(self.Model, self.Tokenizer, '')

if __name__ == '__main__':
    unittest.main()
