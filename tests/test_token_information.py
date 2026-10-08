"""Numerical and semantic checks for block information interventions."""
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoTokenizer
from vlm_inference.token_information import (
    CachedQwenSuffix, information_roi_metrics, make_blocks, roi_membership,
    roi_metrics, sentence_means, sentence_token_spans,
)


class MappingAndMetricsTests(unittest.TestCase):
    def test_spatial_groups_are_not_four_consecutive_tokens(self):
        blocks,frames=make_blocks(list(range(10,18))+list(range(20,28)),
            [[1,4,8],[1,4,8]],[{'timestamp':1},{'timestamp':2}])
        self.assertEqual(blocks[0]['token_indices'],[10,11,14,15])
        self.assertEqual(blocks[1]['token_indices'],[12,13,16,17])
        self.assertEqual(blocks[2]['token_indices'],[20,21,24,25])
        self.assertEqual(blocks[1]['x1'],64)
        self.assertEqual(blocks[2]['frame_idx'],1)
        flags=roi_membership(blocks,frames,{'bboxes':[[0,0,64,64]],'frames':{'1':{'blocks':[[0,1]]}}})
        self.assertEqual(flags.tolist(),[True,False,False,True])

    def test_incomplete_groups_are_explicit(self):
        blocks,frames=make_blocks(list(range(9)),[[1,6,6]],[{'timestamp':0}])
        self.assertEqual(len(blocks),1)
        self.assertEqual(frames[0]['ungrouped_visual_tokens'],5)
        self.assertEqual(blocks[0]['token_indices'],[0,1,3,4])

    def test_video_grid_maps_each_frame_to_four_spatial_tokens(self):
        blocks, frames = make_blocks(
            list(range(8)), [[2, 4, 4]],
            [{'timestamp': 1.5}, {'timestamp': 2.5}],
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]['frame_idx'], 0)
        self.assertEqual(blocks[0]['token_indices'], [0, 1, 2, 3])
        self.assertEqual(blocks[1]['frame_idx'], 1)
        self.assertEqual(blocks[1]['token_indices'], [4, 5, 6, 7])
        self.assertEqual([frame['block_rows'] for frame in frames], [1, 1])
        self.assertEqual([frame['block_cols'] for frame in frames], [1, 1])

    def test_metrics_sign_and_sentence_weighting(self):
        result=roi_metrics([[4,-2],[-2,6]], [True,False],8)
        self.assertAlmostEqual(result['roi_information_coverage'],1/3)
        self.assertEqual(result['roi_information_density'],1)
        self.assertEqual(result['non_roi_information_density'],2)
        self.assertAlmostEqual(result['information_enrichment'],2/3)
        self.assertEqual(result['density_ratio'],.5)
        self.assertEqual(result['event_information_coverage'],[1,0])
        self.assertEqual(result['roi_token_ratio'],.5)
        self.assertEqual(result['roi_signed_information_density'],1)
        undefined=roi_metrics([[-1,-3]],[False],4)
        self.assertIsNone(undefined['roi_information_coverage'])
        self.assertIsNone(undefined['roi_information_density'])
        self.assertIsNone(undefined['information_enrichment'])
        self.assertIsNone(undefined['density_ratio'])

    def test_frame_event_map_does_not_guess_uncertain_events(self):
        import yaml
        root=Path(__file__).resolve().parents[1]
        mapping=yaml.safe_load((root/'configs/token_information_frame_event_map.yaml').read_text())
        refs=json.loads((root/'configs/caption_refs_en_event_complete.json').read_text())
        paragraph=refs['references'][mapping['video_id']][0]
        for _event_id,meta in mapping['events'].items():
            self.assertIn(meta['sentence'], paragraph)
        used=set()
        for frame in mapping['frames']:
            matched=set(frame['matched_event_ids'])
            uncertain=set(frame.get('uncertain_event_ids') or [])
            self.assertTrue(matched)
            self.assertFalse(matched & uncertain)
            for event_id in matched:
                self.assertEqual(mapping['events'][event_id]['status'],'confirmed')
            used |= matched
        self.assertGreaterEqual(len(mapping['frames']),5)
        self.assertLessEqual(len(mapping['frames']),10)

    def test_max_event_map_uses_positive_clip(self):
        # Event maps [[4,-1],[0,2]] -> I_max = [4, 2]; ROI is first block.
        values=np.maximum([[4,-1],[0,2]],0).max(axis=1)
        result=information_roi_metrics(values,[True,False],8)
        self.assertEqual(result['roi_information_coverage'],4/6)
        self.assertEqual(result['roi_block_ratio'],.5)
        self.assertAlmostEqual(result['information_enrichment'],4/3)
        self.assertEqual(result['roi_information_density'],4)
        self.assertEqual(result['non_roi_information_density'],2)
        self.assertEqual(result['density_ratio'],2)

    def test_sentence_means_are_not_length_weighted(self):
        spans=[dict(token_start=0,token_end=1),dict(token_start=1,token_end=4)]
        means=sentence_means(torch.tensor([[-1.,-3.,-3.,-3.]]),spans)
        self.assertEqual(means.mean().item(),-2.)

    def test_real_tokenizer_sentence_alignment(self):
        path=Path('/data/models/Qwen3-VL-8B-Instruct-BF16')
        if not path.exists():self.skipTest('Local tokenizer unavailable')
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        text='A car turns left. Another person walks away.'
        ids,spans=sentence_token_spans(tok,text)
        self.assertEqual(ids,tok.encode(text,add_special_tokens=False))
        self.assertEqual(spans[0]['token_end'],spans[1]['token_start'])
        self.assertEqual(spans[-1]['token_end'],len(ids))
        self.assertIn('Another',tok.decode(ids[spans[1]['token_start']:]))


class CachedSuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cfg=Qwen3VLConfig(
            text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
                num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                rope_parameters={'rope_type':'default','rope_theta':10000,'mrope_section':[1,1,2]}),
            vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,
                out_hidden_size=32,num_position_embeddings=64,deepstack_visual_indexes=[0,1,2]),
            image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)
        cfg._attn_implementation='sdpa'
        cls.model=Qwen3VLForConditionalGeneration(cfg).eval()
        cls.inputs=dict(input_ids=torch.tensor([[5,122]+[120]*8+[123,6,7,8,9,10]]),
            image_grid_thw=torch.tensor([[1,4,8]]),pixel_values=torch.randn(32,1536),
            attention_mask=torch.ones(1,16,dtype=torch.long),
            mm_token_type_ids=torch.tensor([[0,0]+[1]*8+[0]*6]))
        cls.engine=CachedQwenSuffix(cls.model,[0,2],list(range(2,10)),[12,13,14],[8,9,10])
        cls.full=cls.engine.capture(cls.inputs)

    def test_retain_all_including_remaining_deepstack(self):
        for l in [0,2]:
            torch.testing.assert_close(self.engine.resume(l,None),self.full,atol=2e-6,rtol=1e-6)
        self.assertEqual(self.engine.counts['vision_forwards'],1)
        self.assertEqual(self.engine.counts['full_forwards'],1)

    def test_batch_equals_individual_and_baseline(self):
        groups=[[2,3,6,7],[4,5,8,9]]
        for l in [0,2]:
            batched=self.engine.resume(l,groups)
            singles=torch.cat([self.engine.resume(l,[g]) for g in groups])
            torch.testing.assert_close(batched,singles,atol=2e-6,rtol=1e-6)
            baseline=self.engine.resume(l,[[]])
            self.assertTrue(torch.isfinite(baseline).all())
            self.assertGreater(float((batched-baseline).abs().max()),0)
        self.assertEqual(self.engine.counts['vision_forwards'],1)

    def test_suffix_matches_independent_text_decoder_pruning(self):
        # Full multimodal forward for a SMALL TEST MODEL only: prune on entry to
        # decoder layer 1, then slice remaining DeepStack features. Independent
        # path tests positional/causal reconstruction and GT prediction indexing.
        keep=torch.tensor([0,1,2,3,6,7,10,11,12,13,14,15])
        handles=[]
        def prune(module,args,kwargs):
            args=(args[0][:,keep],)+args[1:]
            kwargs['position_embeddings']=tuple(x[:,keep] for x in kwargs['position_embeddings'])
            if kwargs.get('position_ids') is not None:
                kwargs['position_ids']=kwargs['position_ids'][:,keep]
            n=len(keep)
            kwargs['attention_mask']=torch.full((1,1,n,n),torch.finfo(torch.float32).min).triu(1)
            return args,kwargs
        # The decoder loop keeps passing the original full mask and positions;
        # replace them at every subsequent layer, but prune hidden only once.
        def later(module,args,kwargs):
            kwargs['position_embeddings']=tuple(x[:,keep] for x in kwargs['position_embeddings'])
            if kwargs.get('position_ids') is not None:kwargs['position_ids']=kwargs['position_ids'][:,keep]
            n=len(keep);kwargs['attention_mask']=torch.full((1,1,n,n),torch.finfo(torch.float32).min).triu(1)
            return args,kwargs
        original=self.model.model.language_model._deepstack_process
        def deepstack(hidden,mask,features):
            if hidden.shape[1]==len(keep):
                mask=mask[:,keep];features=features[[0,1,4,5]]
            return original(hidden,mask,features)
        try:
            decoder=self.model.model.language_model
            handles.append(decoder.layers[1].register_forward_pre_hook(prune,with_kwargs=True))
            for l in [2,3]:handles.append(decoder.layers[l].register_forward_pre_hook(later,with_kwargs=True))
            decoder._deepstack_process=deepstack
            with torch.inference_mode():
                out=self.model(**self.inputs,use_cache=False,logits_to_keep=torch.tensor([8,9,10]))
                expected=self.engine.log_probs(out.logits)
        finally:
            decoder._deepstack_process=original
            for h in handles:h.remove()
        actual=self.engine.resume(0,[[2,3,6,7]])
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
