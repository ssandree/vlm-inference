#!/usr/bin/env python3
"""Validate and aggregate saved LLM semantic judgments against reviewed GT v2.
No caption word matching, no external LLM invocation. Replays are deterministic.
"""
from pathlib import Path
import csv,json,html,hashlib,collections
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
GTROOT=BASE/'gt_human_review_v2'
MODELS=['8b','8b-int8','8b-fp8','30b','30b-fp8']
CONDS=['full_frame','motion_roi']
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def tsv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def ratio(a,b):return a/b if b else None
def count(labels,good,w):return labels.count(good)+w*labels.count('uncertain')
def metrics(rows,w=.5):
    a=[x['label'] for r in rows for x in r['action_judgments']]
    d=[x['label'] for r in rows for x in r['direction_judgments']]
    q=[x['label'] for r in rows for x in r['polarity_judgments']]
    pred=[p for r in rows for p in r['predicted_claims'] if p['kind']!='context']
    p=[x['label'] for x in pred];ev=[x['label'] for x in pred if x['kind']=='event']
    R=ratio(count(a,'matched',w),len(a));P=ratio(count(p,'supported',w),len(p));D=ratio(count(d,'correct',w),len(d));Q=ratio(count(q,'correct',w),len(q))
    hden=ev.count('supported')+ev.count('human_rejected_event')
    h=ratio(ev.count('human_rejected_event'),hden)
    score=100*(.5*R+.3*P+.1*D+.1*Q) if all(x is not None for x in [R,P,D,Q]) else None
    return {'n_captions':len(rows),'action_targets':len(a),'action_matched':a.count('matched'),'action_uncertain':a.count('uncertain'),'action_recall':R,'reviewed_factual_precision':P,'reviewed_event_precision':ratio(count(ev,'supported',w),len(ev)),'direction_targets':len(d),'direction_correct':d.count('correct'),'direction_wrong':d.count('contradictory'),'direction_missing':d.count('missing'),'direction_recall':D,'polarity_targets':len(q),'polarity_correct':q.count('correct'),'polarity_wrong':q.count('contradictory'),'polarity_uncertain':q.count('uncertain'),'polarity_recall':Q,'reviewed_assertions':len(p),'supported_assertions':p.count('supported'),'uncertain_assertions':p.count('uncertain'),'rejected_event_assertions':p.count('human_rejected_event'),'attribute_errors':p.count('attribute_error'),'attribute_error_rate_all_assertions':ratio(p.count('attribute_error'),len(p)),'reviewed_event_assertions':len(ev),'uncertain_event_assertions':ev.count('uncertain'),'resolved_event_denominator':hden,'verified_event_hallucination_rate':h,'all_reviewed_false_claim_rate':ratio(p.count('human_rejected_event')+p.count('attribute_error'),len(p)-p.count('uncertain')),'score':score}
def pct(n):return 'N/A' if n is None else f'{100*n:.2f}%'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in r)+' |' for r in rows])
def main():
    rows=json.loads((ROOT/'judge_results.json').read_text()); gt=json.loads((GTROOT/'evaluation_gt_v2.json').read_text()); targets=json.loads((GTROOT/'scoring_targets.json').read_text());ledger={r['review_id']:r for r in json.loads((GTROOT/'review_ledger.json').read_text())}
    assert len(rows)==90 and len({r['record_id'] for r in rows})==90
    assert {(r['model'],r['condition'],r['video_id']) for r in rows}=={(m,c,v) for m in MODELS for c in CONDS for v in gt}
    family_rows=[];dir_rows=[];pol_rows=[];claims=[];event_rows=[];details=[];unknown=[]
    for r in rows:
        hdr={k:r[k] for k in ['record_id','model','condition','video_id']};t=targets[r['video_id']]
        assert {x['action_family'] for x in r['action_judgments']}==set(t['action_families'])
        assert {x['review_id'] for x in r['direction_judgments']}=={x['review_id'] for x in t['direction_targets']}
        assert {x['review_id'] for x in r['polarity_judgments']}=={x['review_id'] for x in t['polarity_targets']}
        for kind,valid in [('action_judgments',{'matched','missed','uncertain'}),('direction_judgments',{'correct','missing','contradictory','uncertain'}),('polarity_judgments',{'correct','missing','contradictory','uncertain'}),('predicted_claims',{'supported','human_rejected_event','attribute_error','context_error','uncertain'})]:
            for x in r[kind]:
                assert x['label'] in valid
                assert not x['evidence'] or x['evidence'] in r['description']
                if x['label']=='uncertain':unknown.append({**hdr,'kind':kind,**x})
        for p in r['predicted_claims']:
            rev=ledger[p['review_id']];assert rev['video_id']==r['video_id']
            if p['label']=='supported':assert rev['resolved_status']=='yes'
            if p['label']=='human_rejected_event':assert rev['resolved_status']=='no' and rev['semantic_kind']=='event'
            claims.append({**hdr,**p})
        family_rows.extend({**hdr,**x} for x in r['action_judgments']);dir_rows.extend({**hdr,**x} for x in r['direction_judgments']);pol_rows.extend({**hdr,**x} for x in r['polarity_judgments'])
        am={x['action_family']:x['label'] for x in r['action_judgments']};pm={p['review_id']:p for p in r['predicted_claims']}
        projections=[]
        for e in gt[r['video_id']]['caption_events']:
            links=e.get('review_links',[]);hits=[pm[k] for k in links if k in pm]
            if e.get('review_status')=='conflict_requires_actor_alignment':label='uncertain_original_group_conflict'
            elif any(p['label']=='supported' for p in hits):label='matched_semantic_context'
            elif any(p['label']=='uncertain' for p in hits):label='uncertain'
            elif am.get(e['action_family']) in ['matched','uncertain'] and e['provenance_layer']=='original_gt':label='family_mentioned_instance_unlocalized'
            else:label='missed_semantic_context'
            x={**hdr,'gt_event_id':e['caption_event_id'],'action':e['action'],'action_family':e['action_family'],'reference_sentence':e.get('reference_sentence'),'original_start_time':e.get('start_time'),'original_end_time':e.get('end_time'),'direction':e.get('direction'),'review_links':links,'semantic_context_label':label,'action_family_label':am[e['action_family']],'provenance_layer':e['provenance_layer'],'scope':'semantic context/family; not proven per-track instance recall'}
            event_rows.append(x);projections.append(x)
        r['gt_event_details']=projections;r['metrics']=metrics([r]);details.append({**hdr,**r['metrics'],'matched_families':[x['action_family'] for x in r['action_judgments'] if x['label']=='matched'],'missed_families':[x['action_family'] for x in r['action_judgments'] if x['label']=='missed'],'false_event_review_ids':[p['review_id'] for p in r['predicted_claims'] if p['label']=='human_rejected_event'],'attribute_error_review_ids':[p['review_id'] for p in r['predicted_claims'] if p['label']=='attribute_error']})
    summaries=[]
    for model in MODELS:
        for cond in CONDS:
            rr=[r for r in rows if r['model']==model and r['condition']==cond];m=metrics(rr);lo=metrics(rr,0);hi=metrics(rr,1)
            summaries.append({'model':model,'condition':cond,**m,'score_uncertain_0':lo['score'],'score_uncertain_1':hi['score'],'precision_uncertain_0':lo['reviewed_factual_precision'],'precision_uncertain_1':hi['reviewed_factual_precision']})
    ranking=[]
    for model in MODELS:
        ss=[s for s in summaries if s['model']==model]
        ranking.append({'model':model,'mean_score':sum(s['score'] for s in ss)/2,'score_uncertain_0':sum(s['score_uncertain_0'] for s in ss)/2,'score_uncertain_1':sum(s['score_uncertain_1'] for s in ss)/2,'roi_minus_full_frame':ss[1]['score']-ss[0]['score']})
    ranking.sort(key=lambda r:-r['mean_score'])
    for i,r in enumerate(ranking,1):r['rank']=i
    dump('summary.json',summaries);tsv('summary.tsv',summaries);dump('ranking.json',ranking);tsv('ranking.tsv',ranking)
    dump('per_video_details.json',rows);tsv('per_video_details.tsv',details);tsv('action_judgments.tsv',family_rows);tsv('direction_judgments.tsv',dir_rows);tsv('polarity_judgments.tsv',pol_rows);tsv('predicted_claims.tsv',claims);tsv('gt_event_details.tsv',event_rows);dump('uncertain_cases.json',unknown)
    report(rows,summaries,ranking,ledger,gt)
    meta={'judge':'current conversation GPT-6 LLM','mode':'fresh direct semantic judgment, serialized then replayed; no external judge API invocation','temperature':None,'seed':None,'replay_deterministic':True,'fresh_inference_determinism_guaranteed':False,'checklist_sha256':hashlib.sha256((ROOT/'CHECKLIST_ko.reviewed_snapshot.txt').read_bytes()).hexdigest(),'gt_manifest':json.loads((GTROOT/'manifest.json').read_text()),'weights':{'action_family_recall':.50,'reviewed_factual_precision':.30,'direction_recall':.10,'facility_polarity_recall':.10},'uncertain_weight':.5,'n_captions':90,'reviewed_assertion_count':len(claims),'event_detail_rows':len(event_rows),'validation':'all 90 model/condition/video combinations, GT target coverage, label consistency and exact full-caption evidence checked','limitations':['Single nonblind LLM judge; human review informed by these same captions, not held-out.','No source video independently inspected by assistant.','New event timestamps and track IDs unprovided; family recall is not localized instance recall.','Precision/hallucination apply only to reviewed propositions. Unreviewed scene detail is not silently marked correct.','Old and new scores not numerically comparable: GT, action-family policy, precision units, direction scope and weights changed.'],'sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['judge_prompt.txt','author_judge_v2.py','build_gt_v2.py','evaluate_v2.py','judge_results.json']}}
    dump('run_metadata.json',meta)
    current={'version':'human_review_v2','ground_truth':str(GTROOT/'evaluation_gt_v2.json'),'per_video_gt_root':str(GTROOT/'gt'),'scoring_targets':str(GTROOT/'scoring_targets.json'),'review_ledger':str(GTROOT/'review_ledger.json'),'evaluation_report':str(ROOT/'report.html'),'note':'Original gt/ preserved. New evaluation explicitly uses this version.'}
    (BASE/'CURRENT_REVIEWED_GT.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    for s in summaries:print(s['model'],s['condition'],'R/P/H/D/Q:',*[round(s[k]*100,2) if s[k] is not None else None for k in ['action_recall','reviewed_factual_precision','verified_event_hallucination_rate','direction_recall','polarity_recall']],'score',round(s['score'],2))
    print('RANKING',json.dumps(ranking,ensure_ascii=False))

def report(rows,summaries,ranking,ledger,gt):
    manifest=json.loads((GTROOT/'manifest.json').read_text())
    intro=['사용자 검토 GT v2 반영 및 LLM 의미 재평가','',
    f"사용자 체크리스트 140건(원문 yes 82 / no 58)을 반영했다. 메모와 충돌을 해석한 결과는 {manifest['resolved_statuses']}이다. 원본 GT 36건을 보존하고 시간/트랙 미지정의 사람 확인 의미 이벤트 57건을 추가했다. 이 93개를 독립 actor-instance 93건이라고 간주하지 않는다.",
    'GT 실제 파일: gt_human_review_v2/evaluation_gt_v2.json 및 gt_human_review_v2/gt/<video>/caption_events.json. 현재 반영본 경로는 CURRENT_REVIEWED_GT.json에 기록했다. 원본 GT와 사용자 체크리스트는 수정하지 않았다.',
    '',
    '반영한 중요 정정',
    '• V07-02: 흰 세단 주차는 최신 no를 채택. 이전 대화의 yes를 취소했다.',
    '• V04-09: 체크는 no지만 "들어가긴 하는데 작다"는 구체 메모를 우선하여 진입 GT 유지, 작은 가시성 기록.',
    '• V05-06: 접근 후 돌아섬이 아니라 건물에서 나와 왼쪽 이동. 방향은 맞아도 서사 연결은 틀릴 수 있다.',
    '• V06-13: 하차 증거가 없다는 메모는 uncertain. 확정 환각으로 처리하지 않았다.',
    '• V07-04/05: 빨간 차량 이동 no와 그 차량의 좌→우 이동 yes가 충돌하므로 보류.',
    '• V03-07: 설명의 2인 그룹 보행 no는 반영하되 원본 그룹 2개 중 어느 것인지 불명확하므로 충돌 표시. 개인 보행 GT까지 삭제하지 않았다.',
    '• V01-25/28: 같은 흰 픽업트럭이라는 확인을 반영하여 병합. walking/running은 전체 평가에서 locomotion으로 통합.',
    '',
    '새 채점 기준',
    'R: 영상별 고유 행동 계열 39개 기준 recall. 여러 GT 트랙에 같은 문장을 복제해 instance recall로 주장하지 않는다.',
    'P: 사용자가 검토한 예측 명제의 사실 precision. 행동과 방향/회전/주체 연결 등 속성을 포함한다. 고전적인 instance action precision과 다르며, event-only precision도 summary.tsv에 별도 제공한다.',
    'H: 검토가 확정된 이벤트 주장 중 사용자가 부정한 사건 비율. uncertain은 분모에서 제외하고 개수를 공개한다. 방향·극성·인원·정체성 오류는 별도 attribute_error로 집계하며 H에 섞지 않는다. GT에 없다는 이유로 환각 처리하지 않는다.',
    'D: 사람 검토 방향 12개 + 원본 개인 GT 방향 3개 = 15개 기준 recall. Q: 건물 진입/퇴장 극성 7개 기준 recall. 사람 확인의 오른쪽/아래쪽은 필요한 경우 반평면 수준으로 유지하고 정확한 대각선이나 시간 구간을 만들어내지 않는다.',
    '종합점수 = 100 × (0.50R + 0.30P + 0.10D + 0.10Q). uncertain은 0.5, missed/missing/contradictory는 0. uncertain=0/1 민감도 범위도 공개한다. H를 다시 벌점으로 넣지 않는다.',
    '모델 순위는 full_frame과 motion_roi 점수의 동일 가중 평균. 이번 GT·측정 단위·가중치가 달라 이전 점수와 단순 상승폭 비교를 하면 안 된다.',
    '',
    '최종 요약 (R/P/H/D/Q는 %) ',
    table(['모델','조건','행동 R','검토명제 P','사건 H','방향 D','출입극성 Q','점수','uncertain 범위'],[[s['model'],s['condition'],pct(s['action_recall']),pct(s['reviewed_factual_precision']),pct(s['verified_event_hallucination_rate']),pct(s['direction_recall']),pct(s['polarity_recall']),f"{s['score']:.2f}",f"{s['score_uncertain_0']:.2f}–{s['score_uncertain_1']:.2f}"] for s in summaries]),'',
    table(['순위','모델','평균 점수','uncertain 범위','ROI−full_frame'],[[r['rank'],r['model'],f"{r['mean_score']:.2f}",f"{r['score_uncertain_0']:.2f}–{r['score_uncertain_1']:.2f}",f"{r['roi_minus_full_frame']:+.2f}"] for r in ranking]),'',
    '결과 해석',
    '1. 8b-fp8(49.45)과 30b(49.36)의 평균 점수는 약 0.09점 차이며 uncertain 범위가 겹쳐 사실상 근접한 결과다.',
    '2. 8b-fp8 ROI는 하차·문 열기/닫기·물건 꺼내기·대화·출입을 포착해 가장 높은 조건별 점수를 얻었다.',
    '3. 30b full_frame은 짧은 서술의 검토 사실 precision이 높지만, 일부 인원·계단 이동·동일인 연결은 틀렸다. 사건 H가 낮다고 모든 세부 서술이 정확한 것은 아니다.',
    '4. GT 누락 때문에 벌점받던 자전거·차량 이동은 정답으로 복구됐다. 대신 실제로 부정된 주차 후 출발·추가 인물·잔디밭 달리기와 방향/극성 오류가 분리되어 반영됐다.',
    '',
    '재현성 및 한계',
    '이 세션 LLM이 90개 설명 전체를 다시 읽고 의미 판정을 작성했다. keyword/regex 채점을 하지 않았으며 체크리스트 출처를 기계적으로 정답 연결하지 않았다. 예: R028에는 계단 하강이 없어 예전 후보 연결에서 잘못 붙은 no를 제거했다.',
    '사용자 확인은 GT의 사실 근거이며, 설명문의 의미 일치는 LLM 판정이다. 원문 전체를 evidence로 저장했으므로 국소 인용인 것처럼 제시하지 않는다. 전용 API 또는 temperature=0 실행은 하지 않았고 세션 temperature/seed는 null이다. 저장 판정의 재집계는 결정적이다.',
    '원본 영상을 assistant가 다시 검증한 것은 아니다. 검토되지 않은 날씨/OCR/세부 장면은 정답으로 간주하지 않는다. 이 모델 설명으로 GT를 보완했으므로 독립 hold-out 성능이라고 해석할 수 없다. 시간/주체가 없는 새 GT는 추후 instance 평가를 위해 보완이 필요하다.',
    '재실행: python build_gt_v2.py → python author_judge_v2.py → python evaluate_v2.py. 두 번째 스크립트는 이번 LLM 판정을 직렬화하며 새 API 추론을 수행하지 않는다.',
    '상세 파일: review_ledger.json(원답·해석), resolutions.json, GT_UPDATE_ko.txt, per_video_details.json/tsv, predicted_claims.tsv, gt_event_details.tsv, direction_judgments.tsv, polarity_judgments.tsv, uncertain_cases.json.',
    '']
    text=list(intro); text.append('영상별 GT 업데이트와 부정/보류 목록')
    update=['GT v2 업데이트 기록','추가 이벤트 시간/주체 ID는 제공되지 않아 생성하지 않았습니다. 원본 GT는 보존됩니다.','']
    for v in gt:
        update += ['',v,'원본 + 추가 의미 이벤트: '+str(len(gt[v]['caption_events']))]
        for r in ledger.values():
            if r['video_id']==v:update.append(f"{r['review_id']} [{r['raw_answer']} → {r['resolved_status']}] {r.get('corrected_fact_ko',r['statement_ko'])}\n  {r['resolution_ko']}")
    (GTROOT/'GT_UPDATE_ko.txt').write_text('\n'.join(update)+'\n');text+=update
    text.append('\n90개 설명 상세')
    for r in rows:
        text += ['',f"{r['record_id']} {r['model']} {r['condition']} {r['video_id']}",r['description'],r['notes_ko'],table(['행동 계열','판정'],[[a['action_family'],a['label']] for a in r['action_judgments']]),table(['검토 ID','설명에 있는 주장','판정','근거/정정'],[[p['review_id'],p['assertion_ko'],p['label'],p['reason_ko']] for p in r['predicted_claims']])]
    text+=['','Judge prompt 전문',(ROOT/'judge_prompt.txt').read_text()]
    (ROOT/'REPORT_ko.txt').write_text('\n'.join(text)+'\n')
    def ht(headers,data):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</table></div>'
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>사용자 검토 GT v2 재평가</title><style>body{font-family:system-ui,sans-serif;margin:25px;max-width:1500px;color:#182b3c}pre{white-space:pre-wrap;line-height:1.6}details{border:1px solid #bbcbd8;margin:15px 0;padding:12px;border-radius:6px}summary{cursor:pointer;font-weight:bold}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #bdcad3;padding:6px;text-align:left}.scroll{overflow:auto}</style><h1>사용자 검토 GT v2 · 의미 재평가</h1><pre>'+html.escape('\n'.join(intro))+'</pre>']
    page+=['<details><summary>GT 업데이트 · 전체 140개 검토 반영 내역</summary><pre>'+html.escape('\n'.join(update))+'</pre></details>']
    for r in rows:
        page+=['<details><summary>'+html.escape(f"{r['record_id']} · {r['model']} · {r['condition']} · {r['video_id']}")+'</summary><p>'+html.escape(r['description'])+'</p><p>'+html.escape(r['notes_ko'])+'</p>',ht(['행동 계열','판정'],[[a['action_family'],a['label']] for a in r['action_judgments']]),ht(['검토 ID','설명 주장','판정','근거/정정'],[[p['review_id'],p['assertion_ko'],p['label'],p['reason_ko']] for p in r['predicted_claims']]),ht(['방향/극성 GT','값','판정'],[[d['review_id'],d.get('direction',d.get('polarity')),d['label']] for d in r['direction_judgments']+r['polarity_judgments']]),'<details><summary>원본·추가 GT 이벤트별 의미 대응</summary>',ht(['GT ID','행동','설명','판정'],[[e['gt_event_id'],e['action'],e.get('reference_sentence') or '',e['semantic_context_label']] for e in r['gt_event_details']]),'</details></details>']
    page+=['<details><summary>Judge prompt 전문</summary><pre>'+html.escape((ROOT/'judge_prompt.txt').read_text())+'</pre></details></html>']
    (ROOT/'report.html').write_text('\n'.join(page))
if __name__=='__main__':main()
