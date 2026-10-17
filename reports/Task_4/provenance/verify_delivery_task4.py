"""Final local delivery checks; scientific acceptance is recorded separately."""
from pathlib import Path
import hashlib,json,re
from PIL import Image
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
previous=json.loads((OUT/'provenance/previous_reports_sha256.json').read_text())
assert all(sha(ROOT/name)==h for name,h in previous.items())
audit=json.loads((OUT/'provenance/audit_task4_results.json').read_text());assert audit['status']=='verified'
links=[];tables=0
for name in ['task4_analysis.md','exp3_elliptic_report.md']:
    text=(OUT/name).read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in target:
            assert (OUT/target.split('#')[0]).exists(),(name,target);links.append(target)
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('|') and i+1<len(lines) and re.match(r'^\|[-: |]+\|$',lines[i+1]):
            count=line.count('|');j=i+1;tables+=1
            while j<len(lines) and lines[j].startswith('|'):
                assert lines[j].count('|')==count,(name,j+1);j+=1
with Image.open(OUT/'figures/exp3_elliptic_kepler.png') as im:
    size=im.size;dpi=im.info.get('dpi');assert size==(5100,3300) and all(abs(v-300)<1 for v in dpi)
for name in ['unit_tests_local.txt','unit_tests_idh.txt']:
    text=(OUT/name).read_text(encoding='utf-8');assert 'Ran 156 tests' in text and text.rstrip().endswith('OK')
assert len(list((OUT/'checkpoints').glob('*.pt')))==20
receipt=dict(status='verified',protected_reports=len(previous),working_local_links=len(links),markdown_tables=tables,
             figure_pixels=size,figure_dpi=dpi,unit_tests_local=156,unit_tests_idh=156,
             scientific_passed=audit['scientific_passed'],scientific_total=audit['scientific_total'])
(OUT/'provenance/delivery_checks.json').write_text(json.dumps(receipt,indent=2))
files=[p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_sha256.json']
(OUT/'provenance/delivery_sha256.json').write_text(json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in sorted(files)},indent=2))
print(json.dumps(receipt,indent=2));print('Final manifest:',len(files),'files')
