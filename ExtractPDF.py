"""Extract each PDF into its own text plus page boundaries; never modify PDFs."""
import argparse
import json
from pathlib import Path

def Main():
    import pymupdf
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--input', default='Data/pdfs')
    Parser.add_argument('--output', default='Data/extracted')
    Args = Parser.parse_args()
    Output = Path(Args.output)
    Output.mkdir(parents=True, exist_ok=True)
    for PdfPath in sorted(Path(Args.input).glob('*.pdf')):
        Destination = Output / (PdfPath.stem + '.txt')
        Manifest = Output / (PdfPath.stem + '.pages.json')
        if Destination.exists() or Manifest.exists():
            raise FileExistsError(f'Refusing overwrite: {Destination}')
        Parts, Pages, Offset = [], [], 0
        with pymupdf.open(PdfPath) as Document:
            for Number, Page in enumerate(Document, 1):
                Text = Page.get_text('text', sort=True) + '\n\n'
                Pages.append(dict(Page=Number, Start=Offset, End=Offset + len(Text)))
                Parts.append(Text)
                Offset += len(Text)
        Destination.write_text(''.join(Parts), encoding='utf-8')
        Manifest.write_text(json.dumps(dict(Source=str(PdfPath), Pages=Pages), indent=2), encoding='utf-8')
        print(Destination, Offset)

if __name__ == '__main__':
    Main()
