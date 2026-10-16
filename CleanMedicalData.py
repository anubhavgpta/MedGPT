"""Conservative, opt-in cleaning into a NEW directory. Original corpus is untouched."""
import argparse
import json
import re
from pathlib import Path
from collections import Counter

def CleanText(Text):
    # Keep line boundaries: extraction alone cannot distinguish prose from tables.
    Text = Text.replace('\r\n', '\n').replace('\r', '\n').replace('\u00a0', ' ')
    Text = re.sub(r'[ \t]+', ' ', Text)
    return re.sub(r'\n{3,}', '\n\n', Text).strip()

def Audit(TrainText, ValidationText):
    def Describe(Text):
        Paragraphs = [Part.strip() for Part in Text.split('\n\n') if Part.strip()]
        Counts = Counter(Paragraphs)
        return dict(Characters=len(Text), Paragraphs=len(Paragraphs), DuplicateParagraphOccurrences=sum(Count - 1 for Count in Counts.values()),
                    RepeatedShortParagraphs=Counter({Part: Count for Part, Count in Counts.items() if len(Part) < 140 and Count > 2}).most_common(20),
                    NonAscii=Counter(Character for Character in Text if ord(Character) > 127).most_common(),
                    BrokenHyphenCandidates=len(re.findall(r'[A-Za-z]-\s+[a-z]', Text)),
                    LongParagraphs=sum(len(Part) > 5000 for Part in Paragraphs))
    TrainParts = set(re.sub(r'\s+', ' ', Part).strip().casefold() for Part in TrainText.split('\n\n') if len(Part) >= 100)
    ValidationParts = [re.sub(r'\s+', ' ', Part).strip().casefold() for Part in ValidationText.split('\n\n') if len(Part) >= 100]
    return dict(Train=Describe(TrainText), Validation=Describe(ValidationText),
                SharedLongParagraphs=sum(Part in TrainParts for Part in ValidationParts),
                Note='Paragraph overlap is a lower bound on leakage; manually review tables, headers and chapter boundaries.')

def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data', default='Data')
    Parser.add_argument('--report', default='Runs/data-audit.json')
    Parser.add_argument('--clean-output', help='Optional new directory; refuses existing output files')
    Args = Parser.parse_args()
    Directory = Path(Args.data)
    Texts = {Name: (Directory / Name).read_text(encoding='utf-8') for Name in ('train.txt', 'validation.txt')}
    Report = Audit(Texts['train.txt'], Texts['validation.txt'])
    if Args.clean_output:
        Output = Path(Args.clean_output)
        if any((Output / Name).exists() for Name in Texts):
            raise ValueError('Output dataset already exists')
        Output.mkdir(parents=True, exist_ok=True)
        Report['Cleaning'] = {}
        for Name, Text in Texts.items():
            Cleaned = CleanText(Text)
            (Output / Name).write_text(Cleaned, encoding='utf-8')
            Report['Cleaning'][Name] = dict(Before=len(Text), After=len(Cleaned), Removed=len(Text) - len(Cleaned))
    Destination = Path(Args.report)
    Destination.parent.mkdir(parents=True, exist_ok=True)
    Destination.write_text(json.dumps(Report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Audit saved:', Destination, 'Shared long paragraphs:', Report['SharedLongParagraphs'])

if __name__ == '__main__':
    Main()
