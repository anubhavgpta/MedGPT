import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        EmbeddingSize,
        NumHeads,
        ContextSize
    ):

        super().__init__()

        assert EmbeddingSize % NumHeads == 0

        self.NumHeads = NumHeads
        self.HeadSize = EmbeddingSize // NumHeads

        self.QueryKeyValue = nn.Linear(
            EmbeddingSize,
            3 * EmbeddingSize
        )

        self.OutputProjection = nn.Linear(
            EmbeddingSize,
            EmbeddingSize
        )

        CausalMask = torch.tril(
            torch.ones(ContextSize, ContextSize)
        )

        self.register_buffer("CausalMask", CausalMask)

    def forward(self, InputTensor):

        BatchSize, SequenceLength, EmbeddingSize = InputTensor.shape

        Query, Key, Value = self.QueryKeyValue(
            InputTensor
        ).chunk(3, dim=-1)

        Query = Query.view(
            BatchSize,
            SequenceLength,
            self.NumHeads,
            self.HeadSize
        ).transpose(1, 2)

        Key = Key.view(
            BatchSize,
            SequenceLength,
            self.NumHeads,
            self.HeadSize
        ).transpose(1, 2)

        Value = Value.view(
            BatchSize,
            SequenceLength,
            self.NumHeads,
            self.HeadSize
        ).transpose(1, 2)

        AttentionScores = (
            Query @ Key.transpose(-2, -1)
        ) / (self.HeadSize ** 0.5)

        AttentionScores = AttentionScores.masked_fill(
            self.CausalMask[:SequenceLength, :SequenceLength] == 0,
            float("-inf")
        )

        AttentionWeights = F.softmax(
            AttentionScores,
            dim=-1
        )

        AttentionOutput = AttentionWeights @ Value

        AttentionOutput = AttentionOutput.transpose(
            1, 2
        ).contiguous().view(
            BatchSize,
            SequenceLength,
            EmbeddingSize
        )

        return self.OutputProjection(AttentionOutput)

class TransformerBlock(nn.Module):

    def __init__(
        self,
        EmbeddingSize,
        NumHeads,
        ContextSize
    ):
        super().__init__()

        self.Attention = CausalSelfAttention(
            EmbeddingSize,
            NumHeads,
            ContextSize
        )

        self.FeedForward = nn.Sequential(
            nn.Linear(EmbeddingSize, 4 * EmbeddingSize),
            nn.GELU(),
            nn.Linear(4 * EmbeddingSize, EmbeddingSize)
        )

        self.LayerNorm1 = nn.LayerNorm(EmbeddingSize)
        self.LayerNorm2 = nn.LayerNorm(EmbeddingSize)

    def forward(self, InputTensor):

        InputTensor = InputTensor + self.Attention(
            self.LayerNorm1(InputTensor)
        )

        InputTensor = InputTensor + self.FeedForward(
            self.LayerNorm2(InputTensor)
        )

        return InputTensor

class MedGPT(nn.Module):

    def __init__(
        self,
        VocabSize,
        EmbeddingSize=128,
        NumHeads=4,
        NumLayers=3,
        ContextSize=128
    ):
        super().__init__()

        self.ContextSize = ContextSize

        self.TokenEmbedding = nn.Embedding(
            VocabSize,
            EmbeddingSize
        )

        self.PositionEmbedding = nn.Embedding(
            ContextSize,
            EmbeddingSize
        )

        self.TransformerBlocks = nn.Sequential(
            *[
                TransformerBlock(
                    EmbeddingSize,
                    NumHeads,
                    ContextSize
                )
                for LayerIndex in range(NumLayers)
            ]
        )

        self.FinalNorm = nn.LayerNorm(EmbeddingSize)

        self.LanguageHead = nn.Linear(
            EmbeddingSize,
            VocabSize
        )

    def forward(self, InputIds, TargetIds=None):

        BatchSize, SequenceLength = InputIds.shape

        if SequenceLength == 0 or SequenceLength > self.ContextSize:
            raise ValueError("Sequence exceeds context size")

        Positions = torch.arange(
            SequenceLength,
            device=InputIds.device
        )

        TokenVectors = self.TokenEmbedding(InputIds)
        PositionVectors = self.PositionEmbedding(Positions)

        HiddenStates = TokenVectors + PositionVectors

        HiddenStates = self.TransformerBlocks(HiddenStates)

        HiddenStates = self.FinalNorm(HiddenStates)

        Logits = self.LanguageHead(HiddenStates)

        Loss = None

        if TargetIds is not None:

            BatchSize, SequenceLength, VocabSize = Logits.shape

            Loss = F.cross_entropy(
                Logits.reshape(-1, VocabSize),
                TargetIds.reshape(-1)
            )

        return Logits, Loss

if __name__ == "__main__":

    Device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    Model = MedGPT(
        VocabSize=100,
        EmbeddingSize=128,
        NumHeads=4,
        NumLayers=3,
        ContextSize=128
    ).to(Device)

    ParameterCount = sum(
        Parameter.numel()
        for Parameter in Model.parameters()
    )

    InputIds = torch.randint(
        0, 100, (4, 128), device=Device
    )

    TargetIds = torch.randint(
        0, 100, (4, 128), device=Device
    )

    Logits, Loss = Model(InputIds, TargetIds)

    print("Device:", Device)
    print("Parameters:", f"{ParameterCount:,}")
    print("Output Shape:", Logits.shape)
    print("Loss:", Loss.item())

    assert Logits.shape == (4, 128, 100)
    assert torch.isfinite(Loss)

    print("MedGPT forward pass successful!")
