#!/usr/bin/env python3
"""Validate and replay saved LLM decisions; never interpret caption words.
Run: python evaluate.py
Fresh judging requires an LLM reading judge_prompt.txt and inputs.json and writing
judge_results.json using the same schema. This script makes NO model/API calls.
"""
import csv, hashlib, html, json, platform
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODELS=['8b','8b-int8','8b-fp8','30b','30b-fp8']
CONDS=['full_frame','motion_roi']
def dump(name,data): (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def tsv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); w.writeheader(); w.writerows(rows)
def ratio(a,b): return a/b if b else None
def weighted(labels,yes,w): return labels.count(yes)+w*labels.count('uncertain')
def metrics(rows,w=0.5):
    a=[x['label'] for r in rows for x in r['action_judgments']]
    d=[x['label'] for r in rows for x in r['direction_judgments']]
    p=[x['label'] for r in rows for x in r['predicted_events']]
    R=ratio(weighted(a,'matched',w),len(a)); P=ratio(weighted(p,'supported',w),len(p)); D=ratio(weighted(d,'correct',w),len(d))
    H=ratio(p.count('hallucinated_gt')+(1-w)*p.count('uncertain'),len(p))
    result={'n_captions':len(rows),'gt_action_units':len(a),'gt_direction_units':len(d),'predicted_action_units':len(p),'action_matched':a.count('matched'),'action_missed':a.count('missed'),'action_uncertain':a.count('uncertain'),'pred_supported':p.count('supported'),'pred_hallucinated_gt':p.count('hallucinated_gt'),'pred_uncertain':p.count('uncertain'),'direction_correct':d.count('correct'),'direction_missing':d.count('missing'),'direction_contradictory':d.count('contradictory'),'direction_uncertain':d.count('uncertain'),'action_recall':R,'action_precision_gt_only':P,'direction_recall':D,'hallucination_rate_gt_unsupported':H,'confident_unsupported_rate':ratio(p.count('hallucinated_gt'),len(p)),'caption_unsupported_incidence':ratio(sum(any(e['label']=='hallucinated_gt' for e in r['predicted_events']) for r in rows),len(rows)),'score':100*(.5*R+.2*P+.2*D+.1*(1-H)) if None not in (R,P,D,H) else None}
    return result

def main():
    data=json.loads((ROOT/'inputs.json').read_text()); gt=data['ground_truth']; inp={r['record_id']:r for r in data['records']}
    results=json.loads((ROOT/'judge_results.json').read_text())
    assert len(results)==90 and len({r['record_id'] for r in results})==90
    assert {r['record_id'] for r in results}==set(inp)
    expanded=[]; event_rows=[]; pred_rows=[]; detail=[]
    for r in results:
        row=inp[r['record_id']]; events=gt[row['video_id']]['caption_events']; caption=row['description']
        actions={x['action']:x for x in r['action_judgments']}; dirs={(x['action'],x['gt_direction']):x for x in r['direction_judgments']}
        assert set(actions)=={e['action'] for e in events}
        assert set(dirs)=={(e['action'],e['direction']) for e in events if e['direction']}
        assert len(actions)==len(r['action_judgments']) and len(dirs)==len(r['direction_judgments'])
        assert len({p['action'] for p in r['predicted_events']})==len(r['predicted_events'])
        for kind,labels in [('action_judgments',{'matched','missed','uncertain'}),('direction_judgments',{'correct','missing','contradictory','uncertain'}),('predicted_events',{'supported','hallucinated_gt','uncertain'})]:
            for j in r[kind]:
                assert j['label'] in labels
                assert not j['evidence'] or j['evidence'] in caption, (r['record_id'],j)
                if j['label'] in {'matched','correct','supported','uncertain','hallucinated_gt','contradictory'}: assert j['evidence']
        for p in r['predicted_events']:
            if p['label']=='supported': assert actions[p['action']]['label']=='matched'
            if p['label']=='hallucinated_gt': assert p['action'] not in actions
        for d in r['direction_judgments']:
            if d['label']=='correct': assert actions[d['action']]['label']=='matched'
        header={k:row[k] for k in ['record_id','model','condition','video_id']}
        projection=[]
        for e in events:
            a=actions[e['action']]; d=dirs.get((e['action'],e['direction']))
            joint=a['label']
            if d and a['label']!='missed':
                if d['label'] in {'missing','contradictory'}: joint='missed'
                elif a['label']=='uncertain' or d['label']=='uncertain': joint='uncertain'
            x={**header,'gt_event_id':e['caption_event_id'],'gt_action':e['action'],'gt_direction':e['direction'],'event_form':e['event_form'],'source_track_ids':','.join(e['source_track_ids']),'start_time':e['start_time'],'end_time':e['end_time'],'action_label':a['label'],'action_evidence':a['evidence'],'action_reason_ko':a['reason_ko'],'direction_label':d['label'] if d else 'not_applicable','direction_evidence':d['evidence'] if d else '', 'direction_reason_ko':d['reason_ko'] if d else 'GT 방향 없음: 채점 제외','joint_label':joint,'evaluation_scope':'semantic_coverage_projection_not_instance_localization'}
            event_rows.append(x); projection.append(x)
        for p in r['predicted_events']: pred_rows.append({**header,**p})
        m=metrics([r]); details={**row,**r,'metrics':m,'gt_event_judgments':projection}
        expanded.append(details)
        detail.append({**header,**m,'matched_gt_ids':','.join(x['gt_event_id'] for x in projection if x['action_label']=='matched'),'missed_gt_ids':','.join(x['gt_event_id'] for x in projection if x['action_label']=='missed'),'uncertain_gt_ids':','.join(x['gt_event_id'] for x in projection if x['action_label']=='uncertain'),'hallucinated_predicted_events':'; '.join(p['action'] for p in r['predicted_events'] if p['label']=='hallucinated_gt'),'uncertain_predicted_events':'; '.join(p['action'] for p in r['predicted_events'] if p['label']=='uncertain')})
    summaries=[]
    for model in MODELS:
        for cond in CONDS:
            group=[r for r in expanded if r['model']==model and r['condition']==cond]
            m=metrics(group); lo=metrics(group,0); hi=metrics(group,1)
            raw=[e for r in group for e in r['gt_event_judgments']]; raw_d=[e for e in raw if e['gt_direction']]
            m.update({'raw_gt_event_count':len(raw),'raw_gt_direction_count':len(raw_d),'raw_event_action_coverage':ratio(weighted([e['action_label'] for e in raw],'matched',.5),len(raw)),'raw_event_direction_coverage':ratio(weighted([e['direction_label'] for e in raw_d],'correct',.5),len(raw_d)),'raw_event_joint_coverage':ratio(weighted([e['joint_label'] for e in raw],'matched',.5),len(raw)),'score_uncertain_0':lo['score'],'score_uncertain_1':hi['score'],'action_recall_uncertain_0':lo['action_recall'],'action_recall_uncertain_1':hi['action_recall'],'precision_uncertain_0':lo['action_precision_gt_only'],'precision_uncertain_1':hi['action_precision_gt_only'],'unsupported_rate_lower':hi['hallucination_rate_gt_unsupported'],'unsupported_rate_upper':lo['hallucination_rate_gt_unsupported']})
            m['raw_event_composite_score']=100*(.5*m['raw_event_action_coverage']+.3*m['action_precision_gt_only']+.2*m['raw_event_direction_coverage'])
            summaries.append({'model':model,'condition':cond,**m})
    rankings=[]
    for model in MODELS:
        rows=[r for r in summaries if r['model']==model]
        rankings.append({'model':model,'mean_score':sum(r['score'] for r in rows)/2,'raw_event_mean_score':sum(r['raw_event_composite_score'] for r in rows)/2,'score_uncertain_0':sum(r['score_uncertain_0'] for r in rows)/2,'score_uncertain_1':sum(r['score_uncertain_1'] for r in rows)/2,'motion_roi_minus_full_frame':rows[1]['score']-rows[0]['score']})
    rankings.sort(key=lambda r:-r['mean_score'])
    for i,r in enumerate(rankings,1): r['rank']=i
    pooled=[{'condition':c,**metrics([r for r in expanded if r['condition']==c])} for c in CONDS]
    dump('summary.json',summaries); tsv('summary.tsv',summaries)
    dump('raw_event_model_ranking.json', [{'rank':i,'model':r['model'],'raw_event_mean_score':r['raw_event_mean_score']} for i,r in enumerate(sorted(rankings,key=lambda r:-r['raw_event_mean_score']),1)]); dump('model_ranking.json',rankings); tsv('model_ranking.tsv',rankings)
    dump('condition_summary.json',pooled)
    dump('per_video_details.json',expanded); tsv('per_video_details.tsv',detail)
    tsv('gt_event_details.tsv',event_rows); tsv('predicted_event_details.tsv',pred_rows)
    uncertain=[{'record_id':r['record_id'],'model':r['model'],'condition':r['condition'],'video_id':r['video_id'],'kind':kind,**j} for r in expanded for kind in ['action_judgments','direction_judgments','predicted_events'] for j in r[kind] if j['label']=='uncertain']
    dump('uncertain_cases.json',uncertain)
    metadata={'judge':'GPT-6 current Codex conversation LLM','judge_execution':'direct semantic reasoning in this conversation; judgments serialized by author_judgments.py','external_judge_api_calls':0,'fresh_judge_rerun_available_in_script':False,'temperature':None,'seed':None,'temperature_note':'Session temperature/seed are not exposed or controlled. Determinism applies only to replayed judgments and aggregation.','blinded':False,'independent_review':False,'evaluation_date':'2026-10-01','python_version':platform.python_version(),'n_models':5,'n_conditions':2,'n_videos':9,'n_captions':90,'gt_raw_events_per_caption_set':36,'gt_unique_actions_per_caption_set':17,'gt_unique_direction_units_per_caption_set':7,'primary_metric_scope':'video-level semantic action-type coverage, not event-instance recall','uncertain_weight':0.5,'score_weights':{'action_recall':.5,'action_precision_gt_only':.2,'direction_recall':.2,'one_minus_gt_unsupported_rate':.1},'actual_video_hallucination_rate':None,'factual_hallucination_note':'Not measurable from non-exhaustive GT alone. No source-video verification was performed.','validation':{'all_90_keys_unique':True,'all_expected_gt_units_covered':True,'all_evidence_verbatim':True,'label_consistency_checked':True,'original_gt_event_projections':len(event_rows)},'artifacts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'judge_prompt.txt',ROOT/'inputs.json',ROOT/'judge_results.json',ROOT/'author_judgments.py',ROOT/'evaluate.py']}}
    dump('run_metadata.json',metadata)
    report(summaries,rankings,expanded,pooled,metadata)
    for s in summaries: print(s['model'],s['condition'],'R/P/D/H',*[round(100*s[k],2) for k in ['action_recall','action_precision_gt_only','direction_recall','hallucination_rate_gt_unsupported']],'score',round(s['score'],2),'range',round(s['score_uncertain_0'],2),round(s['score_uncertain_1'],2))
    print('Ranking:',rankings)
    print('Validated:',len(event_rows),'GT projections;',len(pred_rows),'predicted semantic actions;',len(uncertain),'uncertain entries (GT/prediction views overlap).')

def pct(v): return 'N/A' if v is None else f'{100*v:.2f}%'
def table(headers,rows): return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])
def report(summaries,rankings,expanded,pooled,metadata):
    text=['LLM semantic judge 평가 결과','',
    '대상: 5개 모델 × 2개 조건 × annotation available 9개 영상 = 90개 설명. 원본 GT는 영상 9개에 총 36개 이벤트. 이 세션의 GPT-6 LLM이 설명 전체를 읽고 의미 판정했으며 keyword/regex 채점은 사용하지 않았다. 구조 파싱과 판정 라벨 집계만 Python으로 수행했다.',
    '',
    '평가 단위와 해석',
    '주요 action recall은 영상별 고유 행동 유형 17개 기준 의미 충족률이다. 동일 행동의 여러 트랙·구간과 개인/그룹 중복을 합친다. 방향 recall은 고유 (영상, 행동, GT 방향) 7개 기준이다. 원본 GT 36개별 판정은 별도로 모두 제공하되 한 행동 언급이 여러 GT ID에 투영될 수 있으므로 개별 이벤트/주체를 식별한 instance recall이 아니다. 이 차이를 숨기지 않기 위해 아래 원본 GT 기준 보조표를 함께 제공한다.',
    'Action precision은 추출한 고유 핵심 예측 행동 중 GT가 지지하는 비율이다. 환각률 H는 요청에 따른 closed-world GT 비지원 이벤트율이다. 이 GT는 차량 이동·배경 행동 등이 누락될 수 있어 H를 실제 영상의 확정 환각률로 해석하면 안 된다. 실제 영상 검증 환각률은 산출하지 않았다. 행동-방향 결합 충족 여부도 GT 상세표의 joint_label로 제공한다.',
    '정적인 자세·배경·OCR·색상·인원수·시각은 채점 범위 밖이다. 차량 이동·주차·보행·대화·물체 조작 등 확신적으로 서술된 핵심 동작은 예측 행동에 포함한다. 따라서 GT에 없는 보행이나 자동차 이동 역시 GT 비지원 항목이 될 수 있다.',
    '',
    'Uncertain 및 점수 규칙',
    'matched/correct/supported=1, missed/missing/contradictory=0, uncertain=0.5. Uncertain 예측은 precision에 0.5, 비지원율에 0.5 반영한다. 0점·1점 민감도 범위는 신뢰구간이 아니다. 방향 누락은 방향 recall에서 0점이며 GT 방향이 null이면 분모에서 제외한다.',
    '종합점수 = 100 × [0.50 × action recall + 0.20 × action precision + 0.20 × direction recall + 0.10 × (1 − H)]. 본 정의에서는 P=1−H이므로 실효 가중치는 행동 50%, GT 지지 precision 30%, 방향 20%다. 9개 영상의 분자/분모를 합산한다. 모델 순위는 두 조건 점수의 동일 가중 평균이다.',
    '',
    '최종 요약표',
    table(['모델','조건','Action recall (17)','Action precision','Direction recall (7)','GT 비지원율 H','점수 /100','uncertain 0~1 점수'],[[s['model'],s['condition'],pct(s['action_recall']),pct(s['action_precision_gt_only']),pct(s['direction_recall']),pct(s['hallucination_rate_gt_unsupported']),f"{s['score']:.2f}",f"{s['score_uncertain_0']:.2f}–{s['score_uncertain_1']:.2f}"] for s in summaries]),
    '',
    '원본 GT 36개 기준 보조표: 의미 포함 범위이며 actor-instance 검출률이 아님',
    table(['모델','조건','Action coverage','Direction coverage','Action+direction coverage','원본 event 가중 종합점수'],[[s['model'],s['condition'],pct(s['raw_event_action_coverage']),pct(s['raw_event_direction_coverage']),pct(s['raw_event_joint_coverage']),f"{s['raw_event_composite_score']:.2f}"] for s in summaries]),
    '',
    '모델 랭킹',
    table(['순위','모델','두 조건 평균','uncertain 범위','ROI − full_frame'],[[r['rank'],r['model'],f"{r['mean_score']:.2f}",f"{r['score_uncertain_0']:.2f}–{r['score_uncertain_1']:.2f}",f"{r['motion_roi_minus_full_frame']:+.2f}"] for r in rankings]),
    '',
    '원본 event 가중 순위(보조): '+ ' > '.join(r['model'] for r in sorted(rankings,key=lambda x:-x['raw_event_mean_score'])),
    '원본 36개를 각각 집계하면 30b가 1위, 중복을 제거한 의미 유형에서는 30b-fp8가 1위다. 30b의 한 right-to-left 보행 구절이 여러 GT 트랙·그룹·시간 구간으로 투영되어 방향 점수가 커지기 때문이다. 평가 단위에 따라 1·2위가 바뀌므로 단일 우승 모델로 단정하지 않는다.',
    '',
    '해석 (5줄)',
    '1. 30b 계열은 full_frame에서 차량 하차와 건물 출입을 명시하여 8b 계열보다 높은 행동 커버리지를 보였다.',
    '2. motion_roi는 여러 모델에서 차량 하차·탑승 및 출입문 통과를 더 구체적으로 기술했다. 다만 30b는 full_frame에 있던 일부 건물 출입과 방향 단서를 ROI에서 잃었다.',
    '3. 010114 영상의 트렁크 열기·닫기, 차량 탑승, 달리기는 모든 모델·조건이 놓쳤다. 보행/자전거/자동차 이동 설명으로 이를 대체할 수 없다.',
    '4. 방향은 전반적으로 누락되며 계단 상하 이동·화면 내 위치·차량 방향을 사람의 이동 방향으로 인정하지 않았다. 특히 30b-fp8 full_frame의 차량 right-to-left는 보행 방향 점수가 아니다.',
    '5. 길고 다양한 설명은 GT에 없는 행동도 추가하므로 GT precision이 낮아질 수 있다. 상위 순위의 uncertain 범위가 겹치며, GT 누락·판정 단위·단일 judge 때문에 점수 차이를 실제 영상 정확도의 확정적 우열로 해석하지 않는다.',
    '',
    '재현성',
    '온도와 seed는 현재 세션에서 설정·확인할 수 없어 null로 기록했다. temperature=0 API 실행을 했다고 주장하지 않는다. 고정 루브릭으로 이 세션 LLM이 직접 판정했으며 블라인드 평가나 독립 judge 검토는 수행하지 않았다. 저장된 판단에 대한 수치 재계산은 결정적이다. 새 LLM 판정의 완전 동일 재현을 보장하지 않는다.',
    '실행: python prepare_inputs.py → python author_judgments.py → python evaluate.py',
    'author_judgments.py는 이번 LLM이 내린 판정을 직렬화하는 스크립트이며 새 모델 추론을 수행하는 자동 judge가 아니다. 판정 변경 없이 재계산할 때는 python evaluate.py만 실행한다. 새 판정은 judge_prompt.txt와 inputs.json을 LLM에 제공하고 동일 스키마의 judge_results.json으로 저장한 뒤 evaluate.py로 집계해야 한다.',
    '입력 파일 SHA-256은 input_manifest.json, 실행 및 판정 파일 정보는 run_metadata.json에 저장했다. per_video_details.json/tsv, gt_event_details.tsv, predicted_event_details.tsv, uncertain_cases.json에서 원문·판정·근거를 검토할 수 있다.',
    '',
    '영상별 상세표 (90개 설명; matched/missed는 행동 의미 판정이며 방향은 별도)',
    ]
    for r in expanded:
        text.extend(['',f"{r['record_id']} | {r['model']} | {r['condition']} | {r['video_id']}",'설명: '+r['description'],
        table(['GT ID','행동','GT 방향','행동','방향','결합','행동 근거'],[[e['gt_event_id'],e['gt_action'],e['gt_direction'] or '—',e['action_label'],e['direction_label'],e['joint_label'],e['action_evidence'] or '—'] for e in r['gt_event_judgments']]),
        table(['예측 핵심 행동','판정','원문 근거','이유'],[[p['action'],p['label'],p['evidence'],p['reason_ko']] for p in r['predicted_events']]),'검토 메모: '+(r['notes_ko'] or '—')])
    text.extend(['','Judge prompt 전문', (ROOT/'judge_prompt.txt').read_text()])
    full='\n'.join(text)+'\n'; (ROOT/'REPORT_ko.txt').write_text(full)
    intro='\n'.join(text[:text.index('영상별 상세표 (90개 설명; matched/missed는 행동 의미 판정이며 방향은 별도)')])
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>VIRAT LLM judge 평가</title><style>body{font-family:system-ui,sans-serif;margin:28px;max-width:1500px;color:#17202a}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd;padding:6px;text-align:left}details{border:1px solid #bbc;padding:12px;margin:14px 0}summary{cursor:pointer;font-weight:700}.missed,.hallucinated_gt,.contradictory{background:#ffeded}.uncertain{background:#fff3cf}.matched,.supported,.correct{background:#e1f5e7}</style><h1>VIRAT LLM judge 평가</h1><pre>'+html.escape(intro)+'</pre><h2>영상별 상세 (클릭해서 펼치기)</h2>']
    def ht(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td class="'+(str(x) if x in ['matched','supported','correct','uncertain','missed','hallucinated_gt','contradictory'] else '')+'">'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    for r in expanded:
        parts.append('<details><summary>'+html.escape(f"{r['model']} / {r['condition']} / {r['video_id']}")+'</summary><p>'+html.escape(r['description'])+'</p>')
        parts.append(ht(['GT ID','행동','GT 방향','행동 판정','방향 판정','결합','근거'],[[e['gt_event_id'],e['gt_action'],e['gt_direction'] or '—',e['action_label'],e['direction_label'],e['joint_label'],e['action_evidence']] for e in r['gt_event_judgments']]))
        parts.append('<h3>예측 핵심 이벤트</h3>'+ht(['행동','판정','근거','이유'],[[p['action'],p['label'],p['evidence'],p['reason_ko']] for p in r['predicted_events']]))
        parts.append('<p>'+html.escape(r['notes_ko'])+'</p></details>')
    parts.append('<details><summary>Judge prompt 전문</summary><pre>'+html.escape((ROOT/'judge_prompt.txt').read_text())+'</pre></details></html>')
    (ROOT/'report.html').write_text('\n'.join(parts))
if __name__=='__main__': main()
