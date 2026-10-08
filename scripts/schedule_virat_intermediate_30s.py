#!/usr/bin/env python3
"""Detached local one-shot scheduler; waits for retry completion before text inference."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_inference.clip_caption import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-at',type=float,required=True)
    args=p.parse_args()
    out=ROOT/'outputs/virat_intermediate_30s';out.mkdir(parents=True,exist_ok=True)
    state=out/'schedule.json'
    def update(status,**extra):
        atomic_json(state,dict(status=status,scheduler_pid=__import__('os').getpid(),scheduled_unix_sec=args.start_at,
                    scheduled_local=datetime.fromtimestamp(args.start_at).isoformat(),updated_unix_sec=time.time(),**extra))
    with (out/'.scheduler.lock').open('a') as exclusive:
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        update('scheduled')
        while time.time()<args.start_at:
            time.sleep(min(30,args.start_at-time.time()))
        retry=ROOT/'outputs/virat_clip_caption'
        # Acquire the same source lock used by caption/retry jobs. Hold throughout this read-only stage.
        with (retry/'.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    update('waiting_for_caption_retry');time.sleep(30)
            retry_status=json.loads((retry/'retry_penalty_1_2_progress.json').read_text())
            if retry_status.get('stage')!='complete' or retry_status.get('failed')!=0:
                update('blocked_incomplete_captions',retry_progress=retry_status)
                print('Caption retry incomplete or failed; no caption inference or source changes performed',flush=True)
                return 1
            command=[str(ROOT/'.vlm-inf/bin/python'),'-u',str(ROOT/'scripts/run_virat_intermediate_30s.py')]
            with (out/'launcher.log').open('a') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                (out/'run.pid').write_text(str(proc.pid)+'\n')
                update('running',pipeline_pid=proc.pid,command=command,started_unix_sec=time.time())
                code=proc.wait()
                update('complete' if code==0 else 'failed',pipeline_pid=proc.pid,returncode=code)
            return code

if __name__=='__main__':raise SystemExit(main())
