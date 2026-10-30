"""Execute the cached notebook on synthetic CPU data; CUDA is validated on the target host."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
os.environ.update(EVT_METRIC_MODE='synthetic_demo',EVT_METRIC_QUICK='1',EVT_METRIC_BACKEND='cpu',
                  MPLBACKEND='module://matplotlib_inline.backend_inline')
nb=nbformat.read(root/'event_stream_metric_benchmark_cached_gpu.ipynb',as_version=4)
nbformat.validate(nb)
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
out=root/'runs/cached_notebook_validation.ipynb'
out.parent.mkdir(parents=True,exist_ok=True)
nbformat.write(nb,out)
print('All cached notebook cells passed:',out)
