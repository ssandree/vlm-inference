#!/usr/bin/env python3
"""Aggregate explicit session-authored visual observations; never classify captions by keywords."""
import csv, hashlib, html, json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCE = PROJECT/'evaluation/experiments/virat/caption_event_gt_191'
OLD = SOURCE/'llm_judge_eval_59_8b_fp16'

def read(p): return json.loads(p.read_text())
def dump(name, obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def family(a): return 'locomotion' if a in ('walking','running') else a
# Authored after viewing every V01--V59 contact sheet. A positive is an action-family
# observation, never verification of all properties in the source caption.
# Codes below serialize observations; they are not text-search classification rules.
CODES={'w':'locomotion','d':'driving','p':'parking','c':'carrying_object','b':'cycling','x':'exiting_facility'}
OBS = {
1:('wd',[129.23,301.50,358.93,531.23],'보행자 위치 변화와 은색 차량 이동이 보임. 하차 순간·차량 동일성·대화는 추가 확인.'),
2:('wd',[14.53,37.73,49.37,119.03],'도로 차량 이동과 밝은 옷 보행자가 보임. 은색 세단 주차·승하차와 나무 뒤 건물 출입은 미확정.'),
3:('wd',[42.33,55.37,68.37,133.50],'도로 차량 이동과 횡단 보행이 보임. 검은 픽업의 특정 시각·차종은 별도 확인.'),
4:('d',[3.33,13.97,19.27,29.90],'도로 차량 이동은 확인. 원거리 인물의 보행·두 명 동행은 판독 보류.'),
5:('w',[6.93,15.48,24.03],'계단의 보행자 위치 변화. 아래 patio에서 물건을 든 사람은 판독 보류.'),
6:('w',[81.18,131.16,181.14,256.11],'계단과 전경 광장 보행 확인. 문 경계 통과·읽기·ROI 화면 전환 서사는 추가 확인.'),
7:('w',[0.42,5.59,12.47],'두 사람이 계단에서 아래쪽으로 이동. 잔디 위 여성 경로는 확인되지 않음.'),
8:('w',[2.80,9.55,18.48],'계단과 상단 통로 보행 확인. 문 출입 인원 및 대화는 추가 확인.'),
9:('w',[1.21,10.97,20.78,35.42],'계단 보행이 있어 full-frame 캡션은 사람 행동을 빠뜨림. 문 퇴장은 추가 확인.'),
10:('w',[0.50,8.43,18.31,22.24],'계단 위아래 보행 확인. 분홍 옷·문 퇴장·카메라 초점 변경은 별도 확인.'),
11:('w',[0.71,6.55,26.99,32.83],'푸른 옷 인물과 마지막 전경 인물의 이동 확인. 걷기/달리기 속도는 통합.'),
12:('w',[11.68,16.10,20.57],'전경 두 사람의 보행 확인. 문 진입 및 퇴장은 프레임 사이 사건일 수 있음.'),
13:('w',[3.96,16.27,18.02],'계단 보행 확인. 문 경계 통과와 접근을 구분해야 함.'),
14:('w',[6.55,11.18,17.31],'계단에서 내려와 전경으로 이동하는 사람 확인.'),
15:('w',[0.33,6.59,14.10],'상단 우측 통로 인물 위치 변화. 작은 운반물의 색과 주체는 보류.'),
16:('w',[0.75,6.59,15.35,29.95],'계단의 보행 확인. 잔디 횡단과 아래 patio 서사는 별도 확인.'),
17:('w',[0.83,7.51,24.16,34.13],'계단 보행과 상단 우측 인물 이동 확인. doorway 출입·대화는 추가 확인.'),
18:('w',[0.58,7.59,12.27,26.32],'푸른 옷 인물의 계단 하강과 아래쪽 이동 확인. 대화는 정지 자세만으로 확정 불가.'),
19:('w',[2.55,10.72,15.89,19.32],'계단 하강과 전경 보행 확인. 출입·대화는 추가 확인.'),
20:('w',[0.54,4.96,16.06,24.91],'여러 계단/전경 보행 확인. 같은 사람의 왕복 순서는 별도 확인.'),
21:('w',[1.04,18.02,39.26,47.77],'상단 통로와 계단 보행 확인.'),
22:('w',[0.50,6.30,13.98,21.78],'계단을 내려오는 두 사람의 위치 변화 확인.'),
23:('wdp',[4.38,21.94,57.07,144.85],'어두운 세단이 전경 위치를 바꾸고 정차, 우측 두 사람과 상단 도로 이동 확인. 세단/흰 밴 주체 대응은 별도 확인.'),
24:('w',[3.63,12.35,23.99],'우측 인물이 차량 사이로 이동. 자전거는 이 표본에서 확정 못함.'),
25:('w',[0.58,5.13,9.68,11.97],'어두운 세단 옆 인물이 흰 컨테이너 쪽으로 이동. 트렁크 꺼내기·바닥 내려놓기·자전거는 추가 확인.'),
26:('w',[0.75,6.76,30.87],'전경 인물 이동 확인. 초반 이동수단의 자전거/카트 구분은 추가 확인.'),
27:('wb',[2.09,7.05,12.06,17.02],'두 보행자와 우측에서 전경으로 이동하는 자전거 확인. 카트 동반 여부는 추가 확인.'),
28:('d',[0.71,9.18,23.28],'좌측 흰 차량 위치 변화 확인. 아주 작은 보행자/자전거는 보류.'),
29:('wc',[2.17,5.67,7.43],'우측 전경 사람이 물건을 든 채 이동. 회색 SUV의 이동은 보류.'),
30:('w',[15.77,23.20,30.62],'전경 사람이 흰 컨테이너 옆으로 이동. full-frame 캡션은 사람 행동을 누락. 자전거·상호작용은 추가 확인.'),
31:('w',[12.06,16.65,23.57],'그늘 전경에서 인물 이동. 스쿠터인지 보행인지 이동수단은 추가 확인.'),
32:('w',[4.21,10.93,17.65],'우측 인물의 전경 방향 이동 확인. 자전거는 판독 보류.'),
33:('wc',[9.64,14.18,16.44],'노란색 물건을 든 우측 전경 인물의 이동 확인.'),
34:('w',[0.75,9.76,21.78,33.79],'우측 및 원거리 인물 이동 확인. 어두운 운반물과 출발 위치는 별도 확인.'),
35:('wd',[6.17,17.10,22.61,30.83],'상단 도로 흰 밴 이동과 우측 원거리 사람 이동 확인. 주차장 진입/전경이라는 구체 위치는 별도 확인.'),
36:('w',[0.54,11.26,17.69,24.11],'우측에서 중앙 쪽으로 보행자 이동. 차량 내 상대방과 상호작용은 별도 확인.'),
37:('dp',[0.29,4.80,8.22,12.77],'상단 우측 은색 차량이 위치를 바꾼 뒤 정차. 흰 밴 출발·추가 차량 서사는 별도 확인.'),
38:('d',[0.83,14.48,24.74,34.96],'상단 붉은 차량의 이동 확인. 보행자 및 밝기 전환은 보류.'),
39:('b',[4.26,8.05,11.85],'우측에서 전경으로 이동하는 자전거 확인. 두 보행자 동행은 확정 못함.'),
40:('wd',[13.43,22.95,32.42,35.59],'우측 전경 사람 이동과 상단 도로 차량 이동 확인. 하차·카트·상자 운반/내려놓기는 추가 확인.'),
41:('wd',[0.63,10.68,20.73,28.28],'우측 전경 사람과 상단 좌측 차량 위치 변화 확인. 흰 밴·자전거의 특정 주체는 별도 확인.'),
42:('b',[16,17,18],'도로 확대 표본에서 두 바퀴 이동수단을 타는 인물의 우→좌 이동 확인. 자전거 행동 종류로 판정하되 보행·인원·색·경로 세부는 추가 확인.'),
43:('w',[0.92,4.51],'초록 옷 전경 사람이 화면 아래로 나감. 다른 사람의 운반물은 별도 확인.'),
44:('w',[0.42,8.59,15.19,18.44],'좌측 통로 여러 보행자 이동 확인.'),
45:('w',[0.50,8.59,14.69,20.73],'좌측 통로 사람과 건물 옆 사람 이동 확인. 문 진입과 새 비행은 추가 확인.'),
46:('w',[0.33,4.51,8.68],'전경 두 사람과 계단 인물의 이동 확인. 문 퇴장과 조깅은 별도 확인.'),
47:('w',[0.38,4.96,11.06],'건물 좌측 원거리 통로 보행 확인. 카메라가 아래로 이동한다는 서사는 보류.'),
48:('w',[0.42,6.93,11.81,18.31],'상단 통로에서 입구 쪽 및 전경으로 인물 이동. 계단의 주체·방향·CHICAGO 표기는 추가 확인.'),
49:('wc',[2.42,6.30,10.14],'우측 전경 인물이 물건을 든 채 이동. 가방 종류·좌우 방향·달리기 속도는 별도 확인.'),
50:('wc',[8.26,17.98,19.90,21.86],'좌측 원거리 두 사람 및 우측 전경 배낭 인물의 이동 확인. 동행 분리와 서로 다른 방향은 추가 확인.'),
51:('w',[0.38,8.09,15.81,17.36],'좌측 통로와 우측 계단을 올라가는 인물 확인.'),
52:('w',[0.88,7.88,28.87,39.34],'입구 옆 사람 이동과 마지막 우측 계단 하강 확인. 서 있는 사람들의 대화는 확정 불가.'),
53:('wx',[0.42,3.80,5.51,7.18,17.36],'문이 열린 뒤 문 쪽에서 사람이 나오고 좌측 통로로 이동. 처음 사람과 나중 사람 동일성·가방은 추가 확인.'),
54:('d',[5.87,9.48,18.49],'좌측 붉은 차량 위치 변화 확인. 두 붉은 차량 및 나뭇가지 움직임은 별도 확인.'),
55:('d',[1.20,15.42,34.37,53.32],'붉은 차량이 좌측 도로 구간에서 위치를 바꿈. 컨버터블 형태·작은 보행자는 추가 확인.'),
56:('w',[6.37,12.05,20.52],'좌측 주차 차량 옆 사람의 이동 확인. full-frame은 이 사람 행동을 누락. 빨간 옷/녹색 차·나무 정지는 별도 확인.'),
57:('w',[46.75,69.20,76.68],'좌측에서 흰 옷 보행자가 도로를 건너 전경으로 이동. 운반 가방·두 명·좌우 방향은 추가 확인.'),
58:('d',[23.92,33.03,37.60,42.14],'파란 픽업이 좌측에서 중앙으로 돌아 나가는 위치 변화 확인. 두 사람 하차와 다시 출발하는 인과 서사는 추가 확인.'),
59:('d',[22.16,26.39,30.60,34.83],'좌측 녹색 계열 차량이 중앙 도로로 이동. 사람 접근·물러남·그림자/나무 움직임은 추가 확인.'),
}
assert set(OBS)==set(range(1,60))
inputs=read(OLD/'inputs.json'); records=inputs['records']; judges=read(OLD/'judge_results.json'); gt=inputs['ground_truth']; manifest={v['video_id']:v for v in read(SOURCE/'manifest.json')['videos']}
frames=read(ROOT/'frame_manifest.json'); observations=[]; per=[]; claims=[]
for vi in range(59):
 vid=records[vi*2]['video_id']; codes,times,note=OBS[vi+1]; visual=set(CODES[c] for c in codes)
 events=gt.get(vid,{}).get('caption_events',[]); annotations=set(family(e['action']) for e in events)
 refs=visual|annotations
 assert refs, f"Reference unresolved: {vid}; put into checklist rather than inventing a score"
 observations.append({'video_number':vi+1,'video_id':vid,'visual_action_families':sorted(visual),'annotated_action_families':sorted(annotations),'reference_action_families':sorted(refs),'evidence_times_sec':times,'notes_ko':note,'annotation_status':manifest.get(vid,{}).get('annotation_status','unknown'),'gt_status':records[vi*2]['gt_status'],'contact_sheet':frames[vi]['sheet']})
 for r,j in zip(records[vi*2:vi*2+2],judges[vi*2:vi*2+2]):
  grouped={}
  for e in j['predicted_events']:
   a=family(e['action']); grouped.setdefault(a,[]).append(e)
  pred=[]
  for a,es in grouped.items():
   semantic=all(e.get('semantic_uncertain',False) for e in es)
   status='yes' if a in refs and not semantic else 'unsure'
   evidence_source='visual_samples' if a in visual else 'original_gt_annotation' if a in annotations else 'unresolved'
   row={'claim_id':r['record_id']+'-'+a,'record_id':r['record_id'],'video_id':vid,'condition':r['condition'],'action_family':a,'answer':status,'evidence_source':evidence_source,'caption_evidence':[e['evidence'] for e in es],'visual_times_sec':times if a in visual else [],'gt_event_ids':[e['caption_event_id'] for e in events if family(e['action'])==a],'scope':'행동 종류의 존재. 인원·색·방향·장소·동일 주체·순서는 미확정.'}
   pred.append(row);claims.append(row)
  def metrics(u):
   p=(sum(1 if x['answer']=='yes' else u for x in pred)/len(pred)) if pred else None
   recalled=sum((u if all(e.get('semantic_uncertain',False) for e in grouped[a]) else 1) for a in refs if a in grouped)
   recall=recalled/len(refs) if refs else None
   score=0 if not pred else 100*(2*p*recall/(p+recall) if p+recall else 0)
   return {'precision':p,'recall':recall,'score':score,'recall_numerator':recalled,'precision_numerator':sum(1 if x['answer']=='yes' else u for x in pred)}
  mid,low,high=metrics(.5),metrics(0),metrics(1)
  per.append({'record_id':r['record_id'],'video_id':vid,'condition':r['condition'],'gt_status':r['gt_status'],'annotation_status':observations[-1]['annotation_status'],'score':mid['score'],'score_lower':low['score'],'score_upper':high['score'],'score_status':'reference_pending' if not refs else 'provisional','n_reference_families':len(refs),'n_predicted_families':len(pred),'n_yes':sum(x['answer']=='yes' for x in pred),'n_unsure':sum(x['answer']=='unsure' for x in pred),'n_no':0,'action_precision':mid['precision'],'action_recall':mid['recall'],'precision_numerator':mid['precision_numerator'],'recall_numerator':mid['recall_numerator'],'description':r['description'],'reference_families':sorted(refs),'predicted_families':list(grouped),'observations_ko':note,'claims':pred})
# Every original review question is retained with its ID. It asks for actor/location/
# attributes beyond family existence, so action-family positives do not auto-answer it.
review=read(OLD/'review_candidates.json')
for q in review:
 q['source']='original_291';q['assistant_note_ko']='보고서의 행동 종류 판정과 별개로 이 구체 명제는 사용자 확인 필요.'
for c in claims:
 if c['answer']!='unsure': continue
 existing=next((q for q in review if q['video_id']==c['video_id'] and q['kind']=='predicted_claim' and any(s['record_id']==c['record_id'] and s['evidence'] in c['caption_evidence'] for s in q['sources'])),None)
 if existing:
  existing.setdefault('score_claim_ids',[]).append(c['claim_id'])
 else:
  review.append({'review_id':'A-'+c['claim_id'],'video_id':c['video_id'],'kind':'score_claim','priority':'high','question_ko':f"‘{c['action_family']}’ 행동이 실제로 있는가?",'check_point_ko':'프레임 표본만으로 확정 못한 행동. GT 없음/빈 GT를 오답으로 간주하지 않음.','sources':[{'record_id':c['record_id'],'condition':c['condition'],'evidence':e} for e in c['caption_evidence']],'answer':'unsure','score_claim_ids':[c['claim_id']]})
# Explicitly include full descriptions so omitted attributes, posture, camera movement,
# birds/trees, repeated events, and narratives all remain reviewable.
for r in records:
 review.append({'review_id':'D-'+r['record_id'],'video_id':r['video_id'],'kind':'full_caption_details','priority':'normal','question_ko':'아래 설명에서 행동 종류 외의 인원·색·방향·장소·출입 극성·동일 주체·순서·배경/카메라 주장 중 틀리거나 확인 불가한 부분을 모두 정정해 주세요.','check_point_ko':'yes=원문 세부 주장 모두 확인; no=틀린 구절과 실제 장면 기록; unsure=판독 불가 구절 기록. 앉기/서기·새/나무·카메라 서사도 여기 포함. 행동 종류 점수가 100이어도 이 명제의 정답은 보장되지 않음.','sources':[{'record_id':r['record_id'],'condition':r['condition'],'evidence':r['description']}],'answer':'unsure','correction_ko':''})
for q in review:
 for k in ['start_sec','end_sec','actor_id','existing_gt_id','same_event_as','correction_ko']:q.setdefault(k,'')
summary=[]
for cond in ['full_frame','motion_roi']:
 rs=[r for r in per if r['condition']==cond]
 summary.append({'condition':cond,'n_videos':len(rs),'mean_provisional_score':sum(r['score'] for r in rs)/len(rs),'mean_lower':sum(r['score_lower'] for r in rs)/len(rs),'mean_upper':sum(r['score_upper'] for r in rs)/len(rs),'reference_pending':sum(r['score_status']=='reference_pending' for r in rs),'n_yes':sum(r['n_yes'] for r in rs),'n_unsure':sum(r['n_unsure'] for r in rs),'n_predicted_families':sum(r['n_predicted_families'] for r in rs)})
METHOD='''59개 전체 × full_frame/motion_roi = 118개 캡션의 행동 종류 잠정 채점.
원본 영상별 12개 시점(총 708프레임)의 contact sheet를 세션 assistant가 직접 봄. V42는 도로·이동수단 확대 표본 27개를 추가 검토. 연속 재생/전 프레임 검증은 아님.
행동 종류 기준: 기존 수동 의미 추출 predicted_events를 사용. walking/running은 locomotion으로 합침. 인원·색·방향·세부 장소·동일 주체·순서는 별도 체크리스트.
영상별 기준 행동 = 원본 GT에 기록된 종류 ∪ 이번 프레임에서 확인한 종류. 원본 GT는 annotation 근거로 인정하며 사건 자체 검증은 기존 체크리스트에 유지. 빈/누락 GT는 보이는 행동으로 보완하며 제외/자동 오답 처리하지 않음.
P=(확인된 예측 종류 + 0.5×미확정 예측 종류)/예측 종류 수. R=기준 종류 중 캡션이 명시한 종류/기준 종류 수. 캡션 의미 자체 모호하면 해당 R 기여도도 0.5.
잠정 점수=100×2PR/(P+R). 행동 서술이 없고 기준 행동은 있으면 0.
범위는 미확정 예측을 0/1로 바꾸는 민감도. 이미 인정한 GT와 표본 관측의 오류, 누락 기준 행동, 미확정 세부 속성까지 반영하는 신뢰구간은 아님.
이 점수는 행동 종류의 존재와 누락을 평가하며 문장 전체 사실성 점수가 아님. 옳은 행동 종류를 잘못된 주체·색·위치로 서술한 문제는 checklist.txt에서 추가 판단 필요.
예측이 없는 P는 N/A로 보존. 앉기/서기·새/나무·카메라 움직임·표지판 읽기는 행동 점수 taxonomy 바깥이며 전체 원문 검토 항목에 보존.
이전 33영상 50R+30P+20D 점수와 직접 비교 불가. 59개를 같은 규칙으로 점수화하려고 방향 GT에 의존하지 않는 행동 F1로 명시적으로 변경.
'''
# Correct the pending-reference range: no invented known recall for V42.
for r in per:
 if r['score_status']=='reference_pending':r['score_lower']=0;r['score_upper']=100
for s in summary:
 rs=[r for r in per if r['condition']==s['condition']];s['mean_lower']=sum(r['score_lower'] for r in rs)/59;s['mean_upper']=sum(r['score_upper'] for r in rs)/59
# Save reproducible inputs; reaggregation uses these snapshots after first run.
dump('observations.json',observations);dump('per_caption_scores.json',per);dump('action_claims.json',claims);dump('review_candidates.json',review);dump('summary.json',summary)
columns=['record_id','video_id','condition','gt_status','annotation_status','score','score_lower','score_upper','score_status','n_reference_families','n_predicted_families','n_yes','n_unsure','n_no','action_precision','action_recall']
with (ROOT/'per_caption_scores.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=columns,delimiter='\t');w.writeheader();w.writerows({k:r[k] for k in columns} for r in per)
with (ROOT/'per_video_scores.tsv').open('w') as f:
 fields=['video_id','gt_status','annotation_status','full_frame_score','full_frame_range','motion_roi_score','motion_roi_range','observations_ko'];w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
 for i in range(59):
  a,b=per[2*i:2*i+2];w.writerow({'video_id':a['video_id'],'gt_status':a['gt_status'],'annotation_status':a['annotation_status'],'full_frame_score':round(a['score'],2),'full_frame_range':f"{a['score_lower']:.2f}–{a['score_upper']:.2f}",'motion_roi_score':round(b['score'],2),'motion_roi_range':f"{b['score_lower']:.2f}–{b['score_upper']:.2f}",'observations_ko':a['observations_ko']})
lines=['59영상 전체 평가: 사용자 확인 체크리스트','',METHOD,f'전체 {len(review)}개 질문. 기존 291개 ID 보존 + 전체 118개 캡션 세부 검토 + 빠진 미확정 채점 행동.','[unsure]를 [yes] / [no]로 바꾸고 수정/추가 메모에 실제 행동·주체·방향을 적어 주세요.','예: A-R003-entering_facility = no / 실제: 문 앞 보행만 있음.','기존 291개는 세부 명제의 미해결 질문이며, 행동 종류 자체의 assistant 판정은 각 영상 머리말 및 보고서에서 확인.','']
for i,obs in enumerate(observations):
 a,b=per[2*i:2*i+2];lines += [f"V{i+1:02d} {obs['video_id']} | GT={obs['gt_status']} | annotation={obs['annotation_status']}",f"잠정 점수: Full-frame {a['score']:.2f} ({a['score_lower']:.2f}–{a['score_upper']:.2f}), Motion ROI {b['score']:.2f} ({b['score_lower']:.2f}–{b['score_upper']:.2f})",'영상 관측: '+obs['notes_ko'],'확인 시각(클립 초): '+str(obs['evidence_times_sec']),'기준 행동 종류: '+', '.join(obs['reference_action_families'])]
 for q in [q for q in review if q['video_id']==obs['video_id']]:
  lines += [f"[{q['answer']}] {q['review_id']} [{q['priority']}/{q['kind']}] {q['question_ko']}",'  확인 포인트: '+q['check_point_ko']]
  for s in q['sources']:lines += [f"  출처: {s['record_id']} ({s['condition']})",'  원문: '+s['evidence']]
  if q.get('gt_refs'):lines += ['  GT 참조: '+json.dumps(q['gt_refs'],ensure_ascii=False)]
  if q.get('score_claim_ids'):lines += ['  연결 채점 항목: '+', '.join(q['score_claim_ids'])]
  lines += ['  시간(초): ____ ~ ____ / 주체: ____ / same_event_as: ____','  수정/추가 메모: '+q.get('correction_ko',''),'']
 lines += ['']
(ROOT/'checklist.txt').write_text('\n'.join(lines)+'\n')
report=['59개 영상 전체 행동 평가 (잠정)','',METHOD,'결과: 영상별 점수의 산술평균']
for s in summary:report.append(f"{s['condition']}: {s['mean_provisional_score']:.2f}/100, 민감도 {s['mean_lower']:.2f}–{s['mean_upper']:.2f}, 확인 행동 {s['n_yes']}, 미확정 행동 {s['n_unsure']}, 기준 미확정 영상 {s['reference_pending']}")
report += ['',f'checklist.txt: {len(review)}개 질문 (기존 291개 보존).','per_video_scores.tsv: 59행, per_caption_scores.tsv: 118행.','observations.json 및 frame_manifest.json: 원본 프레임 관측·시각과 GT/annotation 상태.','답변 반영 전까지 모든 점수 잠정. 재실행은 집계이며 새로운 영상 검토를 하지 않음.']
(ROOT/'REPORT_ko.txt').write_text('\n'.join(report)+'\n')
chunks=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 59영상 잠정 평가</title><style>body{font:15px system-ui;margin:24px;max-width:1500px}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px}img{max-width:100%}details{border:1px solid #ccc;margin:12px 0;padding:10px}</style><h1>59영상 전체 행동 평가 · 잠정</h1><p><a href="checklist.txt">전체 확인 checklist.txt</a> · <a href="per_video_scores.tsv">영상별 점수</a></p><pre>'+html.escape('\n'.join(report))+'</pre><table><tr><th>영상</th><th>GT / annotation</th><th>Full-frame 점수 [범위]</th><th>Motion ROI 점수 [범위]</th></tr>']
for i,obs in enumerate(observations):
 a,b=per[2*i:2*i+2];chunks.append(f'<tr><td><a href="#V{i+1:02d}">V{i+1:02d} {obs["video_id"]}</a></td><td>{obs["gt_status"]} / {obs["annotation_status"]}</td><td>{a["score"]:.2f} [{a["score_lower"]:.2f}–{a["score_upper"]:.2f}]</td><td>{b["score"]:.2f} [{b["score_lower"]:.2f}–{b["score_upper"]:.2f}]</td></tr>')
chunks.append('</table>')
for i,obs in enumerate(observations):
 chunks.append(f'<details id="V{i+1:02d}"><summary>V{i+1:02d} {obs["video_id"]}</summary><p>{html.escape(obs["notes_ko"])}</p><p>기준 행동: {html.escape(", ".join(obs["reference_action_families"]))}</p><img loading="lazy" src="{obs["contact_sheet"]}" alt="12개 표본 프레임">')
 if i==41:chunks.append('<p>추가 확대 검토 (16–18초 이동수단)</p><img src="V42_vehicle.jpg" alt="추가 이동수단 확대 표본">')
 for r in per[2*i:2*i+2]:chunks.append('<h3>'+r['condition']+f' · {r["score"]:.2f}</h3><p>'+html.escape(r['description'])+'</p><pre>'+html.escape(json.dumps(r['claims'],ensure_ascii=False,indent=2))+'</pre>')
 chunks.append('</details>')
(ROOT/'report.html').write_text('\n'.join(chunks)+'</html>')
validation={'n_videos':len(observations),'n_captions':len(per),'n_contact_sheets':len(frames),'n_samples':sum(len(f['sample_times_sec']) for f in frames),'n_supplemental_samples':27,'no_excluded_videos':len(per)==118,'all_scores_numeric':all(isinstance(r['score'],(int,float)) for r in per),'all_uncertain_claims_in_checklist':all(any(c['claim_id'] in q.get('score_claim_ids',[]) for q in review) for c in claims if c['answer']=='unsure'),'all_original_review_ids_preserved':set(q['review_id'] for q in read(OLD/'review_candidates.json'))<=set(q['review_id'] for q in review),'all_full_captions_in_checklist':all('D-'+r['record_id'] in {q['review_id'] for q in review} for r in records),'annotation_status_counts':dict(Counter(o['annotation_status'] for o in observations)),'gt_status_counts':dict(Counter(o['gt_status'] for o in observations)),'n_review_questions':len(review),'continuous_video_verification':False}
assert all(validation[k] for k in ['no_excluded_videos','all_scores_numeric','all_uncertain_claims_in_checklist','all_original_review_ids_preserved','all_full_captions_in_checklist'])
dump('validation.json',validation)
dump('run_metadata.json',{'date_kst':'2026-10-06','judge':'current assistant, explicit visual contact-sheet reasoning','sample_only':True,'full_caption_factual_accuracy_scored':False,'score':'provisional_action_family_F1','original_gt_unchanged':True,'prior_human_gt_not_auto_merged':True,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'inputs.json',OLD/'judge_results.json',OLD/'review_candidates.json',SOURCE/'manifest.json',ROOT/'frame_manifest.json',Path(__file__)]}})
print(json.dumps({'summary':summary,'validation':validation},ensure_ascii=False,indent=2))
