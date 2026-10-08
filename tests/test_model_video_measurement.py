"""Verify actual video sampling metadata reaches Qwen's processor unchanged."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_inference.model import prepare_multimodal_inputs


class VideoMeasurementTests(unittest.TestCase):
    def test_actual_reader_fps_indices_and_processor_grid(self):
        metadata = {"fps": 30.0, "total_num_frames": 120, "frames_indices": [0, 40, 79, 119], "video_backend": "av"}
        frames = torch.zeros(4, 3, 32, 64)
        class Processor:
            image_token_id = 9
            video_token_id = 8
            video_processor = SimpleNamespace(temporal_patch_size=2)
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(input_ids=torch.tensor([[1, 8, 8, 2]]), video_grid_thw=torch.tensor([[2, 2, 4]]))
        processor = Processor()
        with patch("vlm_inference.model.process_vision_info", return_value=(None, [(frames, metadata)], {"do_sample_frames": False})):
            prepared = prepare_multimodal_inputs(processor, [])
        self.assertIs(processor.kwargs["videos"][0], frames)
        self.assertIs(processor.kwargs["video_metadata"][0], metadata)
        self.assertFalse(processor.kwargs["do_sample_frames"])
        self.assertFalse(processor.kwargs["do_resize"])
        measured = prepared.video_inputs[0]
        self.assertEqual(measured["sampled_frame_count"], 4)
        self.assertEqual(measured["effective_sample_fps"], 1.0)
        self.assertEqual(measured["source_frame_indices"], [0, 40, 79, 119])
        self.assertEqual(measured["model_frame_count_with_temporal_padding"], 4)
        self.assertAlmostEqual(measured["frame_timestamps_sec"][-1], 119/30)
        self.assertEqual(prepared.token_stats["num_visual_tokens"], 2)

class GenerateReuseTests(unittest.TestCase):
    def test_prepared_inputs_are_generated_without_second_decode_and_eos_not_truncation(self):
        from transformers import BatchFeature
        from vlm_inference.config import GenerationConfig
        from vlm_inference.model import PreparedInputs, Qwen3VLCaptioner
        captioner = Qwen3VLCaptioner.__new__(Qwen3VLCaptioner)
        captioner.cfg = SimpleNamespace(generation=GenerationConfig(max_new_tokens=3, do_sample=False))
        class Model:
            generation_config = SimpleNamespace(eos_token_id=[151645, 151643], pad_token_id=151643)
            def parameters(self):
                return iter([torch.zeros(1)])
            def generate(self, **kwargs):
                assert kwargs["do_sample"] is False
                self.received = kwargs["input_ids"]
                return torch.cat((kwargs["input_ids"], torch.tensor([[5, 6, 151645]])), dim=1)
        class Processor:
            def batch_decode(self, ids, **kwargs):
                self.decoded = ids[0].tolist()
                return ["[]"]
        captioner.model, captioner.processor = Model(), Processor()
        inputs = BatchFeature({"input_ids": torch.tensor([[1, 2]])})
        prepared = PreparedInputs(inputs, {"num_visual_tokens": 0, "num_text_tokens": 2, "num_total_input_tokens": 2}, [])
        with patch("vlm_inference.model.prepare_multimodal_inputs", side_effect=AssertionError("second decode")):
            result = captioner.generate(None, "prompt", prepared=prepared)
        self.assertEqual(captioner.processor.decoded, [5, 6, 151645])
        self.assertEqual(result.num_output_tokens, 3)
        self.assertFalse(result.hit_token_limit)
        self.assertEqual(result.num_total_input_tokens, 2)

class TextGenerateMeasurementTests(unittest.TestCase):
    def test_text_only_generate_counts_tokens_and_distinguishes_eos_from_limit(self):
        from transformers import BatchFeature
        from vlm_inference.config import GenerationConfig
        from vlm_inference.model import Qwen3VLCaptioner
        captioner = Qwen3VLCaptioner.__new__(Qwen3VLCaptioner)
        captioner.cfg = SimpleNamespace(generation=GenerationConfig(max_new_tokens=2, do_sample=False))
        class Model:
            generation_config = SimpleNamespace(eos_token_id=[151645,151643],pad_token_id=151643)
            final_token = 151645
            def parameters(self):
                return iter([torch.zeros(1)])
            def generate(self, **kwargs):
                return torch.cat((kwargs['input_ids'],torch.tensor([[5,self.final_token]])),dim=1)
        class Processor:
            image_token_id, video_token_id = 9, 8
            def apply_chat_template(self, messages, **kwargs):
                self.messages=messages
                return 'text prompt'
            def __call__(self, **kwargs):
                self.kwargs=kwargs
                return BatchFeature({'input_ids':torch.tensor([[1,2]])})
            def batch_decode(self, *args, **kwargs):
                return ['[]']
        captioner.model,captioner.processor=Model(),Processor()
        result=captioner.generate_text('Merge only these captions.')
        self.assertEqual(result.num_output_tokens,2)
        self.assertFalse(result.hit_token_limit)
        self.assertEqual(result.num_visual_tokens,0)
        self.assertIsNone(captioner.processor.kwargs['images'])
        self.assertIsNone(captioner.processor.kwargs['videos'])
        self.assertEqual(captioner.processor.messages[0]['content'],[{'type':'text','text':'Merge only these captions.'}])
        captioner.model.final_token=6
        self.assertTrue(captioner.generate_text('Same prompt').hit_token_limit)


if __name__ == "__main__":
    unittest.main()
