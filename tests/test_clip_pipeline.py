from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
import json
import sys
import tempfile
import unittest

import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_inference.clip_caption import atomic_json, segment_video
from vlm_inference.clip_pipeline import materialize_segments, valid_record, preserve_failed
from vlm_inference.video_io import inspect_video


def make_video(path,frames=7,rate=2):
    with av.open(str(path),'w') as container:
        stream=container.add_stream('ffv1',rate=rate)
        stream.width,stream.height,stream.pix_fmt=32,32,'yuv420p'
        for i in range(frames):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),(40+i*10)%220+15,dtype=np.uint8),format='rgb24')
            frame.pts=i
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


def rgb_frames(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]


class SegmentTests(unittest.TestCase):
    def test_one_pass_segments_keep_source_pixels_and_resume_missing_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source.avi';make_video(source)
            info=inspect_video(source);windows=segment_video(info.frame_count,info.native_fps)
            materialize_segments(info,windows,base/'segments')
            original=rgb_frames(source)
            for w in windows:
                frames=rgb_frames(base/'segments'/f'{w.index:04d}.avi')
                self.assertEqual(len(frames),w.end_frame-w.start_frame)
                self.assertTrue(all(np.array_equal(a,b) for a,b in zip(frames,original[w.start_frame:w.end_frame])))
            first=base/'segments/0000.avi';timestamp=first.stat().st_mtime_ns
            (base/'segments/0001.avi').unlink()
            materialize_segments(info,windows,base/'segments')
            self.assertEqual(first.stat().st_mtime_ns,timestamp)

    def test_subframe_tail_is_retained_without_new_observation(self):
        windows=segment_video(504,23.97,allow_empty_tail=True)
        self.assertEqual(len(windows),8)
        self.assertEqual((windows[-1].start_frame,windows[-1].end_frame),(503,504))
        self.assertGreater(windows[-1].end_sec,21)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'tail.avi'
            make_video(source,504,Fraction(2397,100))
            info=inspect_video(source)
            materialize_segments(info,windows,base/'segments')
            tail=rgb_frames(base/'segments/0007.avi')
            self.assertEqual(len(tail),1)
            self.assertTrue(np.array_equal(tail[0],rgb_frames(source)[-1]))

    def test_resume_requires_fingerprint_and_normal_status_and_keeps_failed_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'record.json'
            atomic_json(path,dict(fingerprint='same',status='truncated',caption='raw repeated output'))
            self.assertIsNone(valid_record(path,'same'))
            with self.assertRaises(ValueError):valid_record(path,'different')
            preserve_failed(path)
            self.assertFalse(path.exists())
            self.assertEqual(json.loads((Path(tmp)/'record.failed_1.json').read_text())['caption'],'raw repeated output')


class PipelineTests(unittest.TestCase):
    def test_stage_barrier_one_model_load_and_resume_without_any_inference(self):
        spec=importlib.util.spec_from_file_location('pipeline_cli',ROOT/'scripts/run_virat_clip_caption_pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'configs').mkdir()
            paths=[]
            for i in range(59):
                source=base/f'video_{i:02d}.avi';make_video(source,3,30);paths.append(source)
            selection=base/'configs/virat_multi_roi_59.txt'
            selection.write_text('\n'.join(str(p) for p in paths))
            events=[];loads=[]
            class FakeCaptioner:
                def __init__(self,cfg):self.cfg=cfg;self.processor=object();loads.append(1)
                def generate(self,*args,**kwargs):
                    events.append('caption')
                    return SimpleNamespace(text='A person walks.',hit_token_limit=False,num_output_tokens=5)
                def generate_text(self,prompt):
                    self.assertion=None
                    assert events.count('caption')>=59
                    events.append('merge')
                    data=prompt.split('Input clip caption sequence (the only event evidence):\n',1)[1]
                    clips,_=json.JSONDecoder().raw_decode(data)
                    text=json.dumps([dict(start_sec=clips[0]['start_sec'],end_sec=clips[-1]['end_sec'],sentence='A person walks.',source_clips=[c['source_clip'] for c in clips])])
                    return SimpleNamespace(text=text,hit_token_limit=False,num_output_tokens=30,token_stats=lambda:{'num_visual_tokens':0})
            def prepare(*args,**kwargs):return SimpleNamespace(token_stats={},video_inputs=[{'sampled_frame_count':2}])
            arguments=['pipeline','--config',str(ROOT/'configs/virat_clip_caption_59.yaml'),
                       '--merge-config',str(ROOT/'configs/virat_temporal_event_merge.yaml'),
                       '--videos-file',str(selection),'--output-dir',str(base/'output')]
            with patch.object(module,'ROOT',base),patch('vlm_inference.model.Qwen3VLCaptioner',FakeCaptioner),patch('vlm_inference.model.prepare_multimodal_inputs',prepare),patch.object(sys,'argv',arguments):
                self.assertEqual(module.main(),0)
            self.assertEqual(loads,[1]);self.assertEqual(events,['caption']*59+['merge']*59)
            saved=(base/'output/captions/video_00.avi.json').read_bytes()
            (base/'output/merged_events/video_00.avi.json').unlink()
            with patch.object(module,'ROOT',base),patch('vlm_inference.model.Qwen3VLCaptioner',side_effect=AssertionError('Unexpected model reload')),patch.object(sys,'argv',arguments+['--resume']):
                self.assertEqual(module.main(),0)
            self.assertEqual((base/'output/captions/video_00.avi.json').read_bytes(),saved)
            self.assertTrue((base/'output/merged_events/video_00.avi.json').exists())
            events.clear();loads.clear()
            class FirstFailure(FakeCaptioner):
                def generate(self,*args,**kwargs):
                    result=super().generate(*args,**kwargs)
                    if args[0].job_name=='video_00':result.hit_token_limit=True
                    return result
            failed_args=arguments[:-1]+[str(base/'failed_output')]
            with patch.object(module,'ROOT',base),patch('vlm_inference.model.Qwen3VLCaptioner',FirstFailure),patch('vlm_inference.model.prepare_multimodal_inputs',prepare),patch.object(sys,'argv',failed_args):
                self.assertEqual(module.main(),1)
            self.assertEqual(events,['caption']*59)
            state=json.loads((base/'failed_output/progress.json').read_text())
            self.assertEqual(state['stage'],'caption_failed_merge_not_started')
            self.assertEqual(state['normal_caption_videos'],58)
            with patch.object(module,'ROOT',base),patch('vlm_inference.model.Qwen3VLCaptioner',FakeCaptioner),patch('vlm_inference.model.prepare_multimodal_inputs',prepare),patch.object(sys,'argv',failed_args+['--resume']):
                self.assertEqual(module.main(),0)
            self.assertEqual(events.count('caption'),60)
            self.assertEqual(events.count('merge'),59)
            self.assertTrue((base/'failed_output/inference/video_00/0000.failed_1.json').exists())
            atomic_json(base/'output/pilot_gate.json',dict(accepted_for_59_run=False))
            with patch.object(module,'ROOT',base),patch('vlm_inference.model.Qwen3VLCaptioner',side_effect=AssertionError('Gate must prevent loading')),patch.object(sys,'argv',arguments+['--resume']):
                with self.assertRaisesRegex(ValueError,'quality gate'):
                    module.main()




if __name__=='__main__':unittest.main()
