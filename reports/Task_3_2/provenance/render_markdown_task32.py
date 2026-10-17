"""Repair literal table-header pipes without changing experimental numbers.

The GPU source snapshot is frozen. Its tabulate output places unescaped | in
three mathematical headers, which Markdown reads as extra columns. This
postprocessing step substitutes abs(...) in those headers, retaining all rows.
"""
from pathlib import Path
import hashlib
import json


def repair(folder):
    path=folder/'exp2_circular_report.md'
    original=path.read_text(encoding='utf-8')
    updated=original.replace('|slope+1.5|','abs(slope+1.5)').replace('|QᵀQ−I|','abs(QᵀQ−I)')
    receipt={'purpose':'Markdown table-header correction only; no changed numerical measurements',
             'before_sha256':hashlib.sha256(original.encode('utf-8')).hexdigest(),
             'after_sha256':hashlib.sha256(updated.encode('utf-8')).hexdigest(),
             'numeric_characters_unchanged': ''.join(c for c in original if c.isdigit()) == ''.join(c for c in updated if c.isdigit())}
    if not receipt['numeric_characters_unchanged']:
        raise ValueError('Unexpected numerical change')
    path.write_text(updated,encoding='utf-8')
    (folder/'provenance/markdown_formatting_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':
    repair(Path(__file__).resolve().parents[1])

