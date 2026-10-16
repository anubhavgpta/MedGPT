from pathlib import Path

class MedicalTokenizer:

    def __init__(self, TextData="", Characters=None):
        self.Characters = list(Characters) if Characters is not None else sorted(set(TextData))
        if not self.Characters or len(set(self.Characters)) != len(self.Characters):
            raise ValueError("Vocabulary must be nonempty and unique")
        self.VocabSize = len(self.Characters)
        self.CharToIndex = {
            Character: Index
            for Index, Character in enumerate(self.Characters)
        }

        self.IndexToChar = {
            Index: Character
            for Index, Character in enumerate(self.Characters)
        }

    def Encode(self, InputText):
        Unknown = set(InputText) - set(self.Characters)
        if Unknown:
            raise ValueError(f"Unsupported characters: {ascii(sorted(Unknown))}")
        return [
            self.CharToIndex[Character]
            for Character in InputText
        ]

    def Decode(self, TokenIds):
        return "".join(
            self.IndexToChar[TokenId]
            for TokenId in TokenIds
        )

if __name__ == "__main__":

    TrainText = Path("Data/train.txt").read_text(
        encoding="utf-8"
    )

    ValidationText = Path("Data/validation.txt").read_text(
        encoding="utf-8"
    )

    Tokenizer = MedicalTokenizer(
        TrainText + ValidationText
    )

    SampleText = TrainText[:50]

    EncodedText = Tokenizer.Encode(SampleText)
    DecodedText = Tokenizer.Decode(EncodedText)

    print("Vocabulary Size:", Tokenizer.VocabSize)
    print("Original Text:", repr(SampleText))
    print("Encoded Tokens:", EncodedText)
    print("Decoded Text:", repr(DecodedText))

    assert SampleText == DecodedText

    print("Tokenizer test passed!")
