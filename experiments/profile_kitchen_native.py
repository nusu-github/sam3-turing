"""Profile the selected INT8 attention configuration, preserving its baseline."""
import os
import argparse
import subprocess
import sys
from run_kitchen_extended import environment

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--layout',choices=['head','sequence'],default='head')
args=parser.parse_args()
env = environment('kitchen_all')
env['SAM3_EXPERIMENT_KITCHEN_LAYOUT']=args.layout
name='kitchen-hotspots-ba018a0' if args.layout=='head' else 'kitchen-sequence-hotspots-ba018a0'
root = '.cache/native-perf/'+name
subprocess.run([sys.executable, 'experiments/profile_native_hotspots.py', root,
                'int8_boundary', 'int8_boundary'], env=env, check=True)
subprocess.run([sys.executable, 'experiments/analyze_native_hotspots.py', root,
                'experiments/results/native_rtx2060/'+name+'.json',
                '.cache/native-perf/kitchen-ba018a0/quality-kitchen_all-truck-1'], check=True)
