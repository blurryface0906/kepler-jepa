"""Presentation-only rerender with consistent model colors in panel (a)."""
from pathlib import Path
import hashlib,json,sys
from unittest.mock import patch
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from experiments.exp3_reporting import figure
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data=OUT/'exp3_elliptic_results.json';target=OUT/'figures/exp3_elliptic_kepler.png'
original=OUT/'provenance/exp3_elliptic_kepler_gpu_original.png'
if not original.exists():original.write_bytes(target.read_bytes())
before=sha(data);report=json.loads(data.read_text())
from matplotlib.axes import Axes
original_scatter=Axes.scatter
def colored_scatter(axis,*args,**kwargs):
    # Only panel (a) omits an explicit color and uses o/x model markers.
    if 'color' not in kwargs and 'c' not in kwargs:
        kwargs['color']={'o':'tab:blue','x':'tab:purple'}[kwargs['marker']]
    return original_scatter(axis,*args,**kwargs)
with patch.object(Axes,'scatter',colored_scatter):figure(report,OUT)
assert sha(data)==before
receipt=dict(change='Panel(a): consistent model colors across estimators; marker shapes unchanged. No numerical changes.',
             original_figure_sha256=sha(original),final_figure_sha256=sha(target),numerical_json_sha256=before,
             final_reporting_source_sha256=sha(ROOT/'experiments/exp3_reporting.py'),
             presentation_renderer_sha256=sha(Path(__file__)),
             frozen_GPU_reporting_source_sha256=report['source_sha256']['experiments/exp3_reporting.py'])
(OUT/'provenance/presentation_changes.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
