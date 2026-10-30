"""Explicit synthetic notebook validation; never edits the main notebook."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='runs/notebook_validation.ipynb')
args=parser.parse_args()
os.environ['EVT_METRIC_MODE']='synthetic_demo'
os.environ['EVT_METRIC_QUICK']='1'
os.environ['MPLBACKEND']='Agg'
nb=nbformat.read(ROOT/'event_stream_metric_benchmark.ipynb',as_version=4)
nbformat.validate(nb)
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
nbformat.write(nb,out)
print('Executed all cells successfully:',out)
