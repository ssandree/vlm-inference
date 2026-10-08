#!/usr/bin/env python3
"""Validate and aggregate saved semantic decisions; no caption interpretation."""
import csv,hashlib,html,json,platform
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONDS=['full_frame','motion_roi']
def dump(name,x): (ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def tsv(name,rows):
 if not rows:return
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
def family(a):return 'locomotion' if a in {'walking','running'} else a
def div(a,b):return a/b if b else None
def units(js,kind,normalize=True):
 if not normalize:return js[kind]
 out={}
 for j in js[kind]:
  key=(family(j['action']),j['gt_direction']) if kind=='direction_judgments' else family(j['action'])
  # Same family means one unit; strongest saved semantic evidence wins.
  rank={'matched':3,'supported':3,'correct':3,'uncertain':2,'contradictory':1,'missed':0,'missing':0,'hallucinated_gt':0,'unscored_missing_gt':0}
  if key not in out or rank[j['label']]>rank[out[key]['label']]:out[key]={**j,'action':family(j['action'])}
 return list(out.values())
def normalized(r):
 out={**r}
 out['action_judgments']=units(r,'action_judgments')
 # walking and running cover the same GT family, including a fine-grained miss.
 for a in out['action_judgments']:
  ps=[p for p in r['predicted_events'] if family(p['action'])==a['action']]
  if ps:
   p=next((p for p in ps if not p['semantic_uncertain']),ps[0]);a.update(label='uncertain' if p['semantic_uncertain'] else 'matched',evidence=p['evidence'])
 out['direction_judgments']=units(r,'direction_judgments')
 out['predicted_events']=units(r,'predicted_events')
 for p in out['predicted_events']:
  if r['gt_status']!='missing_gt':p['label']='uncertain' if p['semantic_uncertain'] else 'supported' if any(a['action']==p['action'] for a in out['action_judgments']) else 'hallucinated_gt'
 return out
def metrics(rows,w=.5):
 a=[j['label'] for r in rows for j in r['action_judgments']];d=[j['label'] for r in rows for j in r['direction_judgments']];p=[j['label'] for r in rows for j in r['predicted_events']]
 R=div(a.count('matched')+w*a.count('uncertain'),len(a));P=div(p.count('supported')+w*p.count('uncertain'),len(p));D=div(d.count('correct')+w*d.count('uncertain'),len(d))
 H=div(p.count('hallucinated_gt')+(1-w)*p.count('uncertain'),len(p))
 return {'n_captions':len(rows),'gt_action_units':len(a),'gt_direction_units':len(d),'predicted_action_units':len(p),'action_matched':a.count('matched'),'action_uncertain':a.count('uncertain'),'action_missed':a.count('missed'),'pred_supported':p.count('supported'),'pred_uncertain':p.count('uncertain'),'pred_gt_unsupported':p.count('hallucinated_gt'),'direction_correct':d.count('correct'),'direction_uncertain':d.count('uncertain'),'direction_missing':d.count('missing'),'direction_contradictory':d.count('contradictory'),'action_recall':R,'action_precision_gt_only':P,'direction_recall':D,'gt_unsupported_rate':H,'score':100*(.5*R+.3*P+.2*D) if None not in (R,P,D) else None}
def pct(x):return 'N/A' if x is None else f'{x*100:.2f}%'
def table(head,rows):return '\n'.join([' | '.join(head),' | '.join(['---']*len(head))]+[' | '.join(map(str,r)) for r in rows])
def main():
 data=json.loads((ROOT/'inputs.json').read_text());jr=json.loads((ROOT/'judge_results.json').read_text());recs=data['records'];gt=data['ground_truth'];lookup={r['record_id']:r for r in recs}
 assert len(recs)==len(jr)==118 and len(lookup)==118 and len({r['record_id'] for r in jr})==118
 assert {r['record_id'] for r in jr}==set(lookup)
 assert len({(r['video_id'],r['condition']) for r in recs})==118
 expanded=[];events=[];actions=[];directions=[];preds=[]
 for j in jr:
  r={**lookup[j['record_id']],**j};ev=gt.get(r['video_id'],{}).get('caption_events',[])
  assert {a['action'] for a in j['action_judgments']}=={e['action'] for e in ev}
  assert {(d['action'],d['gt_direction']) for d in j['direction_judgments']}=={(e['action'],e['direction']) for e in ev if e['direction']}
  for kind,allowed in [('action_judgments',{'matched','missed','uncertain'}),('direction_judgments',{'correct','missing','uncertain','contradictory'}),('predicted_events',{'supported','hallucinated_gt','uncertain','unscored_missing_gt'})]:
   for v in j[kind]:
    assert v['label'] in allowed
    assert not v['evidence'] or v['evidence'] in r['description']
    if v['label'] not in {'missed','missing'}:assert v['evidence']
  r=normalized(r);r['metrics']=metrics([r]) if r['gt_status']!='missing_gt' else None
  base={k:r[k] for k in ['record_id','video_id','condition','gt_status']}
  for a in r['action_judgments']:actions.append({**base,**a})
  for d in r['direction_judgments']:directions.append({**base,**d})
  for p in r['predicted_events']:preds.append({**base,**p})
  projected=[]
  for e in ev:
   a=next(a for a in r['action_judgments'] if a['action']==family(e['action']))
   d=next((d for d in r['direction_judgments'] if d['action']==family(e['action']) and d['gt_direction']==e['direction']),None)
   joint='missed' if a['label']=='missed' or d and d['label'] in {'missing','contradictory'} else 'uncertain' if a['label']=='uncertain' or d and d['label']=='uncertain' else 'matched'
   row={**base,'gt_event_id':e['caption_event_id'],'original_action':e['action'],'action_family':family(e['action']),'gt_direction':e['direction'],'source_track_ids':','.join(e['source_track_ids']),'start_time':e['start_time'],'end_time':e['end_time'],'action_label':a['label'],'direction_label':d['label'] if d else 'not_applicable','joint_label':joint,'action_evidence':a['evidence'],'direction_evidence':d['evidence'] if d else ''};events.append(row);projected.append(row)
  r['gt_event_judgments']=projected;expanded.append(r)
 summaries=[]
 for scope,statuses in [('primary_nonempty_gt',{'nonempty_gt'}),('sensitivity_all_existing_gt',{'nonempty_gt','empty_gt'}),('empty_gt_only_diagnostic',{'empty_gt'})]:
  for c in CONDS:
   rs=[r for r in expanded if r['condition']==c and r['gt_status'] in statuses];m=metrics(rs)
   raw=[e for r in rs for e in r['gt_event_judgments']];raw_d=[e for e in raw if e['gt_direction']]
   summaries.append({'scope':scope,'condition':c,**m,'score_uncertain_0':metrics(rs,0)['score'],'score_uncertain_1':metrics(rs,1)['score'],'raw_gt_events':len(raw),'raw_direction_events':len(raw_d),'raw_event_action_coverage':div(sum(1 if e['action_label']=='matched' else .5 if e['action_label']=='uncertain' else 0 for e in raw),len(raw)),'raw_event_direction_coverage':div(sum(1 if e['direction_label']=='correct' else .5 if e['direction_label']=='uncertain' else 0 for e in raw_d),len(raw_d)),'raw_event_joint_coverage':div(sum(1 if e['joint_label']=='matched' else .5 if e['joint_label']=='uncertain' else 0 for e in raw),len(raw))})
 dump('summary.json',summaries);tsv('summary.tsv',summaries);dump('per_video_details.json',expanded)
 tsv('per_video_details.tsv',[{**{k:r[k] for k in ['record_id','video_id','condition','gt_status']},**(r['metrics'] or {k:None for k in metrics([])})} for r in expanded])
 tsv('gt_event_details.tsv',events);tsv('action_judgments.tsv',actions);tsv('direction_judgments.tsv',directions);tsv('predicted_events.tsv',preds)
 tsv('coverage.tsv',[{'video_id':r['video_id'],'gt_status':r['gt_status'],'n_gt_events':len(gt.get(r['video_id'],{}).get('caption_events',[]))} for r in recs[::2]])
 dump('uncertain_cases.json',[{'record_id':r['record_id'],'kind':kind,**v} for r in expanded for kind in ['action_judgments','direction_judgments','predicted_events'] for v in r[kind] if v['label']=='uncertain'])
 fine=[]
 for c in CONDS:
  rs=[{**lookup[j['record_id']],**j} for j in jr if lookup[j['record_id']]['condition']==c and lookup[j['record_id']]['gt_status']=='nonempty_gt'];fine.append({'condition':c,**metrics(rs)})
 dump('fine_grained_sensitivity.json',fine)
 paired=[]
 for v in sorted(gt):
  rs=[r for r in expanded if r['video_id']==v];f,m=sorted(rs,key=lambda r:CONDS.index(r['condition']))
  paired.append({'video_id':v,'gt_status':f['gt_status'],'full_frame_action_recall':f['metrics']['action_recall'],'motion_roi_action_recall':m['metrics']['action_recall'],'action_recall_delta':None if f['metrics']['action_recall'] is None else m['metrics']['action_recall']-f['metrics']['action_recall']})
 tsv('paired_video_comparison.tsv',paired)
 # Separate action-level diagnostics, retaining misses for all original units.
 ag=[]
 for c in CONDS:
  for a in sorted({a['action'] for a in actions}):
   js=[j for j in actions if j['condition']==c and j['action']==a];labels=Counter(j['label'] for j in js)
   ag.append({'condition':c,'action_family':a,'n':len(js),'matched':labels['matched'],'uncertain':labels['uncertain'],'missed':labels['missed'],'recall':(labels['matched']+.5*labels['uncertain'])/len(js)})
 tsv('per_action_summary.tsv',ag)
 meta={'judge':'current Codex session LLM semantic reasoning','external_api_calls':0,'video_verified':False,'blinded':False,'independent_review':False,'temperature':None,'seed':None,'date':'2026-10-01','python':platform.python_version(),'gt_root':str(ROOT.parent/'gt'),'original_gt_unchanged':True,'human_review_v2_applied':False,'n_input_videos':59,'n_captions':118,'n_nonempty_gt_videos':33,'n_empty_gt_videos':11,'n_missing_gt_videos':15,'primary_scope':'33 nonempty GT videos, equal paired conditions','unit':'unique video/action family; walking and running merged as locomotion; facility entering/exiting kept separate in original GT baseline','score_weights':{'action_recall':.5,'gt_only_precision':.3,'direction_recall':.2},'actual_factual_precision':None,'actual_hallucination_rate':None,'replay':'python evaluate.py; validates and aggregates saved decisions only','validation':{'all_caption_evidence_verbatim':True,'all_gt_units_covered':True,'all_gt_event_projections':len(events)},'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'inputs.json',ROOT/'input_manifest.json',ROOT/'judge_results.json',ROOT/'judge_prompt.txt',ROOT/'author_judgments.py',ROOT/'evaluate.py']}}
 dump('run_metadata.json',meta);dump('validation.json',meta['validation'])
 text=['59-video 8b-fp16 inference: 원본 GT 기준 의미 평가','',
 '입력: '+str(Path(json.loads((ROOT/'input_manifest.json').read_text())['source_files'][0]['path'])),
 'GT: '+str(ROOT.parent/'gt'),'',
 '전체 59개 영상 × 2조건 = 118개 설명. 추론 error 없음. GT가 있고 이벤트가 있는 33개 영상(66개 설명)을 주 평가 대상으로 사용. 빈 GT 11개 영상(22개 설명)은 별도 진단/민감도 표, GT 누락 15개 영상(30개 설명)은 채점 제외. 원본 GT 이벤트는 186개이며 두 조건 합계 372개 투영을 저장했다.',
 '빈 GT는 실제 사건이 없다는 뜻이 아니다. GT 누락과 빈 GT를 분리했으며 해당 캡션은 체크리스트에서 검토할 수 있다.',
 '', '판정 기준',
 '세션 LLM이 118개 설명 전체와 대응 GT를 읽고 의미 행동 및 인용 근거를 직접 작성했다. 키워드/정규식 채점, 새 API 호출, 원본 영상의 시각 검증은 하지 않았다. 저장 판정에 대한 집계만 결정적이다.',
 '이전 사용자 검토 기록의 walking/running 통합 규칙에 따라 locomotion으로 합친다. 자전거·스쿠터·카트 밀기는 별도. 원본 세부 행동 판정도 judge_results.json과 fine_grained_sensitivity.json에 남겼다.',
 '이번 요청에서 지정한 원본 gt를 사용한다. 사람 검토 GT v2를 자동 병합하지 않았다. 건물 진입/퇴장은 원본 GT처럼 별도 행동 단위이며 계단 이동/접근/화면 이탈만으로 인정하지 않는다. 이전 reviewed_v2의 사람 검토 명제 precision/출입극성 점수와 다른 기준이므로 점수 직접 비교 불가.',
 'R: 고유 (영상, 행동 계열) recall. P: 고유 예측 행동 계열의 원본 GT 지지 비율. D: 고유 (영상, 행동 계열, 방향)의 recall. H=1−P는 GT 비지원율이며 실제 환각률이 아니다. supported도 인원/색/시간/주체/서사까지 사실임을 보증하지 않는다.',
 'matched/supported/correct=1, uncertain=0.5, missed/missing/contradictory=0. 분모 0은 null/N/A이며 빈 예측의 precision을 임의로 100%로 채우지 않는다. 주 점수는 조건별 분자·분모 합산 후 100×(0.50R+0.30P+0.20D). uncertain=0/1 범위는 민감도이며 신뢰구간이 아니다.',
 '186개 원본 GT에 대한 행동·방향·결합 coverage는 동일 구절이 여러 트랙에 투영될 수 있는 보조치이며 instance 검출률이 아니다.',
 '', '결과 (주 평가와 민감도)',
 table(['범위','조건','캡션','행동 R','GT 지지 P','방향 D','GT 비지원 H','점수','uncertain 범위'],[[s['scope'],s['condition'],s['n_captions'],pct(s['action_recall']),pct(s['action_precision_gt_only']),pct(s['direction_recall']),pct(s['gt_unsupported_rate']),'N/A' if s['score'] is None else f"{s['score']:.2f}",'N/A' if s['score'] is None else f"{s['score_uncertain_0']:.2f}–{s['score_uncertain_1']:.2f}"] for s in summaries]),
 '', '원본 이벤트 투영 (주 평가)',
 table(['조건','GT 이벤트 수','행동 coverage','방향 coverage','행동+방향 coverage'],[[s['condition'],s['raw_gt_events'],pct(s['raw_event_action_coverage']),pct(s['raw_event_direction_coverage']),pct(s['raw_event_joint_coverage'])] for s in summaries[:2]]),
 '', '행동별 진단',table(['조건','행동','GT 단위','matched','uncertain','missed','recall'],[[s['condition'],s['action_family'],s['n'],s['matched'],s['uncertain'],s['missed'],pct(s['recall'])] for s in ag]),
 '', '해석',
 'ROI 설명은 일부 건물 출입을 더 명시하지만, 대부분의 계단/입구 주변 묘사는 실제 출입 경계를 설명하지 않아 출입 GT 누락이 많다. 차량 탑승 및 트렁크 열기/닫기도 명시적으로 포착되지 않았다.',
 '새로 서술한 자전거, 카트, 운반, 대화, 차량 이동이 많지만 원본 GT는 비포괄적이다. GT 비지원 증가를 실제 환각 증가로 해석할 수 없다. 사람 확인 후 GT를 보완하고 사실 precision/환각률을 다시 계산해야 한다.',
 '010106_05, 010113_07 등 빈 GT 영상에 물건 꺼내기/하차/박스 내려놓기 서술이 있어 우선 검토 대상으로 넣었다. 010208_09는 좌측으로 걷는 첫 사람과 문을 나와 가방을 든 둘째 사람의 방향을 혼합하지 않았다.',
 '', '산출물',
 'report.html: 요약과 전체 118개 원문 및 판정. CHECKLIST_ko.txt / review.html: 영상별 의문 항목, yes/no/unsure 입력, JSON 내보내기/가져오기. review_answers.tsv: 편집 가능한 응답표. 원본 GT 및 이전 검토 답변은 변경하지 않았다.',
 'per_video_details.json/tsv, action_judgments.tsv, direction_judgments.tsv, predicted_events.tsv, gt_event_details.tsv, coverage.tsv, paired_video_comparison.tsv, per_action_summary.tsv에서 분모와 근거를 확인할 수 있다.',
 '재계산: python evaluate.py. 판정 재직렬화: python author_judgments.py (새 LLM 추론이 아님). 체크리스트 재생성: python build_review.py. 입력 해시: input_manifest.json; 실행 해시: run_metadata.json.',
 '', 'GT 누락 영상',*[r['video_id'] for r in recs[::2] if r['gt_status']=='missing_gt'],
 '', '빈 GT 영상',*[r['video_id'] for r in recs[::2] if r['gt_status']=='empty_gt']]
 (ROOT/'REPORT_ko.txt').write_text('\n'.join(text)+'\n')
 chunks=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>59-video 평가</title><style>body{font:15px system-ui;margin:30px;max-width:1400px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{padding:12px;border:1px solid #aaa;margin:12px 0}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;text-align:left}</style><h1>59-video 8b-fp16 의미 평가</h1><p><a href="review.html">영상별 검토 체크리스트</a></p><pre>'+html.escape('\n'.join(text))+'</pre>']
 for r in expanded:
  chunks.append('<details><summary>'+html.escape(f"{r['record_id']} | {r['video_id']} | {r['condition']} | {r['gt_status']}")+'</summary><p>'+html.escape(r['description'])+'</p><pre>'+html.escape(json.dumps({k:r[k] for k in ['action_judgments','direction_judgments','predicted_events','notes_ko']},ensure_ascii=False,indent=2))+'</pre></details>')
 (ROOT/'report.html').write_text('\n'.join(chunks)+'</html>')
 print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
