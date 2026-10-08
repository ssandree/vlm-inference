"""Target-only visual block information with cached Qwen3-VL decoder suffixes.

Layer l means AFTER zero-based decoder layer l, including its DeepStack update.
The text-only baseline removes visual positions at that boundary, not at input.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import torch


def sentence_token_spans(tokenizer, caption: str, sentences: Sequence[str] | None = None):
    """Tokenize the paragraph ONCE; assign every token to exactly one sentence.

    Explicit sentences must join with one space to reproduce the paragraph.
    Default splitting is punctuation followed by whitespace, suitable for the
    repository's event references. Abbreviations should use explicit sentences.
    """
    caption = caption.strip()
    if not caption:
        raise ValueError('GT caption must not be empty')
    sentences = list(sentences) if sentences else re.split(r'(?<=[.!?。！？])\s+', caption)
    if not sentences or any(not s.strip() for s in sentences):
        raise ValueError('Every event sentence must be nonempty')
    if ' '.join(sentences) != caption:
        raise ValueError('event_sentences joined by one space must equal gt_caption')
    encoded = tokenizer(caption, add_special_tokens=False, return_offsets_mapping=True)
    ends = np.cumsum([len(s)+1 for s in sentences])-1
    owners = []
    for a,b in encoded['offset_mapping']:
        if a == b:
            raise ValueError('Unexpected empty-offset target token')
        while a < b and caption[a].isspace():
            a += 1
        owners.append(min(int(np.searchsorted(ends, a, side='right')), len(sentences)-1))
    spans = []
    for i,sentence in enumerate(sentences):
        indices = [j for j,owner in enumerate(owners) if owner == i]
        if not indices or indices != list(range(indices[0],indices[-1]+1)):
            raise ValueError('Sentence must have a nonempty contiguous token span')
        spans.append(dict(sentence=sentence, token_start=indices[0], token_end=indices[-1]+1))
    return encoded['input_ids'], spans


def sentence_means(log_probs: torch.Tensor, spans: list[dict]) -> torch.Tensor:
    return torch.stack([log_probs[...,s['token_start']:s['token_end']].mean(-1) for s in spans], -1)


def make_blocks(visual_positions, grid_thw, frames, merge_size=2):
    """Make 64x64 spatial blocks for image or video visual-token grids.

    ``grid_thw`` contains Qwen's post-merge spatial grid. For video input a
    single ``[T, H, W]`` entry is accepted, while image input keeps the
    existing one ``[1, H, W]`` entry per frame.

    Odd final token rows/columns cannot form four-token groups and are recorded
    as ungrouped. They are still removed by every target-only intervention.
    """
    positions = list(map(int,visual_positions))
    grids = [list(map(int, grid)) for grid in grid_thw]
    if len(grids) == 1 and grids[0][0] > 1:
        temporal, grid_h, grid_w = grids[0]
        if temporal != len(frames):
            raise ValueError('Video grid temporal size and frame metadata disagree')
        grids = [[1, grid_h, grid_w] for _ in range(temporal)]
    elif len(grids) != len(frames):
        raise ValueError('One image grid is required per full frame')
    blocks,metadata = [],[]
    offset = 0
    for frame_idx,(grid,frame) in enumerate(zip(grids,frames)):
        temporal,h,w = map(int,grid)
        if temporal != 1 or h % merge_size or w % merge_size:
            raise ValueError('Visual grids must have temporal=1 and even spatial dimensions')
        h,w = h//merge_size,w//merge_size
        count = h*w
        frame_positions = positions[offset:offset+count]
        if len(frame_positions) != count:
            raise ValueError('Image grid and image token count disagree')
        for row in range(h//2):
            for col in range(w//2):
                local=[(2*row+dy)*w+2*col+dx for dy in range(2) for dx in range(2)]
                blocks.append(dict(frame_idx=frame_idx,timestamp=float(frame['timestamp']),
                    block_row=row,block_col=col,token_indices=[frame_positions[i] for i in local],
                    visual_token_indices=[offset+i for i in local],
                    x1=col*64,y1=row*64,x2=(col+1)*64,y2=(row+1)*64))
        metadata.append(dict(**frame,frame_idx=frame_idx,token_rows=h,token_cols=w,
            mapped_width=w*32,mapped_height=h*32,visual_tokens=count,
            block_rows=h//2,block_cols=w//2,ungrouped_visual_tokens=count-(h//2)*(w//2)*4))
        offset += count
    if offset != len(positions):
        raise ValueError('Unmapped visual tokens')
    return blocks,metadata


def roi_membership(blocks, frames, spec):
    """Union of boxes/blocks/masks; boxes select blocks by block-center inclusion.

    spec = {bboxes: [[x1,y1,x2,y2]], blocks: [[row,col]], block_mask: [[0,1]],
            frames: {"0": {bboxes/blocks/block_mask: ...}}}.
    Frame-specific entries replace the shared entry for that frame.
    """
    flags=[]
    for b in blocks:
        entry=spec.get('frames',{}).get(str(b['frame_idx']),spec)
        row,col=b['block_row'],b['block_col']
        cx,cy=(b['x1']+b['x2'])/2,(b['y1']+b['y2'])/2
        inside=False
        for box in entry.get('bboxes',[]):
            if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError('ROI bboxes require finite x1 < x2 and y1 < y2')
            inside |= box[0]<=cx<box[2] and box[1]<=cy<box[3]
        selected=set(map(tuple,entry.get('blocks',[])))
        shape=(frames[b['frame_idx']]['block_rows'],frames[b['frame_idx']]['block_cols'])
        if any(len(rc)!=2 or any(int(v)!=v for v in rc) or not (0<=rc[0]<shape[0] and 0<=rc[1]<shape[1]) for rc in selected):
            raise ValueError('ROI block coordinates outside analyzed grid')
        inside |= (row,col) in selected
        if 'block_mask' in entry:
            mask=np.asarray(entry['block_mask'])
            if mask.shape!=shape or not np.isin(mask,[0,1]).all():
                raise ValueError(f'ROI block_mask must be binary with shape {shape}')
            inside |= bool(mask[row,col])
        flags.append(bool(inside))
    return np.asarray(flags,dtype=bool)


def _finite_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    denominator=float(denominator)
    if not np.isfinite(denominator) or denominator==0:
        return None
    return float(numerator/denominator)


def information_roi_metrics(importance_1d, in_roi, total_visual_tokens):
    """ROI metrics for one block map. Densities use max(I, 0); undefined -> None."""
    values=np.asarray(importance_1d,dtype=float).reshape(-1)
    mask=np.asarray(in_roi,dtype=bool)
    if len(values)!=len(mask) or not np.isfinite(values).all():
        raise ValueError('Finite 1-D importance and aligned ROI mask required')
    positive=np.maximum(values,0)
    block_ratio=float(mask.mean()) if len(mask) else None
    coverage=float(positive[mask].sum()/positive.sum()) if positive.sum()>0 else None
    roi_density=float(positive[mask].mean()) if mask.any() else None
    non_roi_density=float(positive[~mask].mean()) if (~mask).any() else None
    return dict(
        roi_block_ratio=block_ratio,
        roi_token_ratio=float(mask.sum()*4/total_visual_tokens) if total_visual_tokens else None,
        analyzed_token_ratio=float(len(mask)*4/total_visual_tokens) if total_visual_tokens else None,
        roi_information_coverage=coverage,
        information_enrichment=_finite_ratio(coverage, block_ratio),
        roi_information_density=roi_density,
        non_roi_information_density=non_roi_density,
        density_ratio=_finite_ratio(roi_density, non_roi_density),
        positive_information_sum=float(positive.sum()),
        roi_signed_information_density=float(values[mask].mean()) if mask.any() else None,
        non_roi_signed_information_density=float(values[~mask].mean()) if (~mask).any() else None,
    )


def roi_metrics(importance, in_roi, total_visual_tokens):
    """Importance: [blocks, events]. Event-mean map is auxiliary; per-event coverage kept."""
    values=np.asarray(importance,dtype=float)
    mask=np.asarray(in_roi,dtype=bool)
    if values.ndim!=2 or len(values)!=len(mask) or not np.isfinite(values).all():
        raise ValueError('Finite [blocks, events] importance and aligned ROI mask required')
    metrics=information_roi_metrics(values.mean(axis=1), mask, total_visual_tokens)
    metrics['event_information_coverage']=[
        information_roi_metrics(values[:,i], mask, total_visual_tokens)['roi_information_coverage']
        for i in range(values.shape[1])]
    return metrics


@dataclass
class Boundary:
    hidden: torch.Tensor
    position_embeddings: tuple[torch.Tensor,torch.Tensor]
    position_ids: torch.Tensor | None


class CachedQwenSuffix:
    """One full multimodal forward, then batched compact-sequence decoder suffixes.

    Transformers Qwen3-VL dense decoder only; SDPA/eager, one model device.
    Original rotary positions survive physical removal. Remaining DeepStack
    injections are indexed to retained visual tokens, including layer-0 trials.
    """
    def __init__(self, model, layers, visual_positions, predictor_positions, target_ids):
        self.model=model
        self.decoder=model.model.language_model
        self.layers=sorted(set(layers))
        if not self.layers or any(l<0 or l>=len(self.decoder.layers)-1 for l in self.layers):
            raise ValueError('Each intervention layer must have at least one following decoder layer')
        if self.decoder.config._attn_implementation not in {'sdpa','eager'}:
            raise ValueError('Analysis requires SDPA or eager attention')
        devices={p.device for p in model.parameters()}
        if len(devices)!=1 or next(iter(devices)).type=='meta':
            raise ValueError('Feasibility analysis requires model on one device, without offload')
        self.device=next(iter(devices))
        self.visual_positions=torch.as_tensor(visual_positions,dtype=torch.long)
        self.predictors=torch.as_tensor(predictor_positions,dtype=torch.long)
        self.targets=torch.as_tensor(target_ids,dtype=torch.long,device=self.device)
        self.boundaries={}
        self.deepstack=[]
        self.counts=dict(vision_forwards=0,full_forwards=0,suffix_batches=0,suffix_examples=0)

    @torch.inference_mode()
    def capture(self, inputs):
        if inputs['input_ids'].shape[0] != 1 or not bool(inputs['attention_mask'].all()):
            raise ValueError('Capture requires one unpadded sequence')
        if self.boundaries:
            raise RuntimeError('capture may only run once')
        handles=[]
        def text_hook(module,args,kwargs):
            self.deepstack=[x.detach().cpu().clone() for x in (kwargs.get('deepstack_visual_embeds') or [])]
        def vision_hook(*unused):
            self.counts['vision_forwards']+=1
        def boundary_hook(layer):
            def hook(module,args,kwargs):
                hidden=args[0] if args else kwargs['hidden_states']
                pos=kwargs.get('position_ids')
                self.boundaries[layer]=Boundary(hidden.detach().cpu().clone(),
                    tuple(x.detach().cpu().clone() for x in kwargs['position_embeddings']),
                    pos.detach().cpu().clone() if pos is not None else None)
            return hook
        try:
            handles.append(self.decoder.register_forward_pre_hook(text_hook,with_kwargs=True))
            handles.append(self.model.model.visual.register_forward_pre_hook(vision_hook))
            for l in self.layers:
                handles.append(self.decoder.layers[l+1].register_forward_pre_hook(boundary_hook(l),with_kwargs=True))
            out=self.model(**inputs,use_cache=False,logits_to_keep=self.predictors.to(self.device),return_dict=True)
            self.counts['full_forwards']+=1
            log_probs=self.log_probs(out.logits).cpu()
            del out
        finally:
            for handle in handles:
                handle.remove()
        if set(self.boundaries)!=set(self.layers):
            raise RuntimeError('Decoder boundary capture incomplete')
        return log_probs

    def log_probs(self, logits):
        # Float32 log normalization and only GT-predicting positions, not full-sequence logits.
        logits=logits.float()
        ids=self.targets[None,:,None].expand(logits.shape[0],-1,1)
        return logits.gather(-1,ids).squeeze(-1)-torch.logsumexp(logits,dim=-1)

    @torch.inference_mode()
    def resume(self, layer: int, groups: list[list[int]] | None):
        """groups=[[]] -> text-only boundary; None -> retain-all replay check.
        All groups in a batch must have equal length (normally four).
        """
        cache=self.boundaries[layer]
        length=cache.hidden.shape[1]
        visual=torch.zeros(length,dtype=torch.bool)
        visual[self.visual_positions]=True
        text=torch.arange(length)[~visual]
        if groups is None:
            keep=torch.arange(length)[None,:]
        else:
            if not groups or len({len(g) for g in groups})!=1:
                raise ValueError('Batch must contain equal-sized nonempty list of interventions')
            indices=[]
            for group in groups:
                g=torch.tensor(group,dtype=torch.long)
                if len(g) and (len(set(group))!=len(group) or not torch.isin(g,self.visual_positions).all()):
                    raise ValueError('Intervention groups must contain distinct visual sequence indices')
                indices.append(torch.cat([text,g]).sort().values)
            keep=torch.stack(indices)
        batch,n=keep.shape
        hidden=cache.hidden[0][keep].to(self.device)
        rotary=tuple(x[0][keep].to(self.device) for x in cache.position_embeddings)
        position_ids=cache.position_ids[0][keep].to(self.device) if cache.position_ids is not None else None
        # Compact unpadded sequences preserve causal order. Use the same SDPA
        # is_causal path as the normal model (explicit masks select other kernels).
        mask=None if self.decoder.config._attn_implementation=='sdpa' else torch.full(
            (n,n),torch.finfo(hidden.dtype).min,dtype=hidden.dtype,device=self.device).triu(1)[None,None]
        retained_visual=visual[keep].to(self.device)
        # Original sequence index -> index in the concatenated visual embedding array.
        ordinal=torch.full((length,),-1,dtype=torch.long)
        ordinal[self.visual_positions]=torch.arange(len(self.visual_positions))
        selected_ordinals=ordinal[keep][visual[keep]]
        for index in range(layer+1,len(self.decoder.layers)):
            hidden=self.decoder.layers[index](hidden,position_embeddings=rotary,
                attention_mask=mask,position_ids=position_ids,use_cache=False)
            if index<len(self.deepstack) and len(selected_ordinals):
                features=self.deepstack[index][selected_ordinals].to(self.device,hidden.dtype)
                hidden=self.decoder._deepstack_process(hidden,retained_visual,features)
        hidden=self.decoder.norm(hidden)
        # Predict y_j at ORIGINAL position y_j-1; never shift labels across pruning.
        predicted=torch.searchsorted(keep.contiguous(),self.predictors[None,:].expand(batch,-1).contiguous())
        if not torch.equal(keep.gather(1,predicted),self.predictors[None,:].expand(batch,-1)):
            raise RuntimeError('A GT predictor position was removed')
        predicted=predicted.to(self.device)
        selected=hidden.gather(1,predicted[:,:,None].expand(-1,-1,hidden.shape[-1]))
        scores=self.log_probs(self.model.lm_head(selected)).cpu()
        self.counts['suffix_batches']+=1
        self.counts['suffix_examples']+=batch
        return scores
