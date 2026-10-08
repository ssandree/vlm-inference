#!/usr/bin/env python3
"""Serialize fresh semantic judgments authored by the current conversation LLM.
This is not a keyword evaluator and does not invoke an external model API.
"""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
DATA=json.loads((ROOT.parent/'inputs.json').read_text())
LEDGER={r['review_id']:r for r in json.loads((BASE/'gt_human_review_v2/review_ledger.json').read_text())}
TARGETS=json.loads((BASE/'gt_human_review_v2/scoring_targets.json').read_text())
VIDEOS=list(DATA['ground_truth'])
# Each line: record index | reviewed propositions actually asserted (local IDs)
# | supported GT action families. ~ means semantic/GT uncertainty.
# Old source links were not used to generate this manually authored decision table.
SPEC='''
0|1 2 3 14 18 19 16 20|D L
1|11 12 6 8 15|D K L
2|1 2 9|L
3|1 2 3|L F
4|1 2 24|D P L
5|1 3 2 6 8 10 11 12 14 15 7|D P O L V R C B
6|1 2 3 10|L
7|1 2 4 5 10|L
8|1 2 8|L F
9|1 2 3 4 5 6 9 10 13|L B A J
10|1 2 3|L
11|2 4 5 6|L
12|1 2 3 13|L
13|4 6 8 12 13|L A
14|1 3 4 14|L
15|6 10 11|L K
16|1|D
17|3 5 6|L D
18|1 2 20 21 22 17 9|D L
19|11 12 15 21 6 10 17|D K L
20|1 2 9|L
21|1 2 3|L F
22|1 2 6 25 26|D P
23|1 3 2 6 8 10 11 15 14 16 21|D P O L V B C
24|1 2 3 10|L
25|1 2 4 6 10 11|L
26|1 3|L
27|1 2 3 9 10 11 13|L A J M
28|2|L
29|2 1 11 4 5 7 ~8 ~9|L F
30|1 2 13|L
31|4 5 6 7 12 13|L A
32|1 3 4|L
33|6 7 10 11 12|L Q K
34|1|D
35|3 8 1|L D
36|1 2 3 14 16 6|D L
37|11 12 6 7 8 15|K D L
38|1 2 9|L
39|1 2 5|L B
40|1 6 25 ~8 ~9 24|D L ~O
41|1 4 2 7 8 10 11 12 13 14 17|D P O L V R C
42|1 2 3 10|L
43|1 4 7 10 11|L
44|1 2 3|L
45|1 2 3 4 9 12 13|L B A G
46|1 2|L
47|2 4 5 7 8 10 12 13 14|L F
48|1 2 3 13|L
49|1 4 5 9 10 11 12 13|L F A
50|1 3 4 5 13|L
51|6 10 11|L K
52|1|D
53|3 4 6 9|L I D
54|4 5 14|~D L
55|4 14 16 11 13 10|~D L K
56|1 ~4 7 8|L ~F D
57|1 2 3|L F
58|1 2 8 9 18 24 28|D P O L
59|1 4 2 5 27 8 18 25|D P O L
60|1 3 10|L
61|1 10 11 8|L F
62|1 2 7|L
63|3 9 13|L A
64|1 7 8|L F
65|1 2 4|L
66|1 13|L
67|1 4 13|L
68|2 8|L
69|6 9 10|L K
70|1 2|D
71|3 4 6 7 10|L I D
72|24|L
73|4 5 17 23 10|~D L
74|1 ~4|L ~F
75|1 5 6|L B
76|1 2 8 9 18 24 28 29 30|D P O L
77|1 4 2 5 27 8 18 19 20 22 23 7|D P O L
78|1 3 10|L
79|1 10 11 8 9 3|L F
80|1 13|L
81|3 9 13|L A
82|1 7 8|L F
83|1 2 4|L
84|1 13|L
85|1 4 9 10 13|L F
86|14 15 ~6|L
87|6 16 10 11|L G K
88|1 2|D
89|3 4 6 7 10|L I D
'''
FAMILIES={'D':'driving','P':'parking','L':'locomotion','F':'facility_transition','O':'getting_out_of_vehicle','I':'getting_into_vehicle','V':'opening_vehicle_door','C':'closing_vehicle_door','T':'opening_vehicle_trunk','Z':'closing_vehicle_trunk','B':'carrying_object','R':'retrieving_object','G':'gesturing','A':'talking','M':'manipulating_object','J':'posture_change','K':'cycling','Q':'looking_around'}
# LLM direction assignments, tied to actor/action; defaults are omissions.
DIR={41:{'V01-17':'correct'},58:{'V01-18':'correct'},59:{'V01-18':'correct'},76:{'V01-18':'correct'},77:{'V01-18':'correct'},23:{'V01-17':'contradictory'},9:{'V03-06':'correct'},12:{'V04-02':'correct'},30:{'V04-02':'correct'},48:{'V04-02':'correct'},10:{'V05-03':'correct'},11:{'V05-06':'correct'},68:{'V06-02':'correct'},50:{'V06-02':'contradictory','V06-05':'correct'},14:{'V06-02':'contradictory'},32:{'V06-02':'contradictory'},1:{'V07-12':'correct'},19:{'V07-12':'correct'},37:{'V07-12':'correct'},55:{'V07-12':'contradictory'},72:{'V07-24':'correct'},56:{'V08-08':'correct'},70:{'V09-02':'correct'},88:{'V09-02':'correct'}}
POL={61:{'V02-08':'correct'},79:{'V02-08':'correct'},8:{'V03-08':'uncertain'},49:{'V04-09':'correct','V04-10':'correct'},85:{'V04-09':'correct','V04-10':'correct'},29:{'V05-07':'correct','V05-08':'uncertain'},47:{'V05-07':'correct','V05-08':'correct'},64:{'V05-07':'correct','V05-08':'correct'},82:{'V05-07':'correct','V05-08':'correct'},3:{'V08-04':'contradictory'},21:{'V08-04':'contradictory'},57:{'V08-04':'contradictory'},56:{'V08-04':'uncertain'},74:{'V08-04':'uncertain'}}
NOTES={13:'사용자가 파티오 보행자의 문 통과 자체를 부정했다. 다른 GT 출입이 있다는 이유로 이 사건을 정답으로 바꾸지 않음.',28:'이전 검토표 출처 연결의 오류를 수정: 실제 문장은 계단 오르기만 서술하므로 내려가기 no 항목을 부과하지 않음.',11:'왼쪽 이동은 인정하되 접근 후 돌아섬이라는 서사는 사용자 정정에 따라 오류로 분리.',40:'parked cars 안쪽에서 나타남은 차 사이인지 하차인지 불명확. 2인 보행은 거부됐지만 하차 해석은 uncertain.',49:'매우 작은 진입도 실제라는 사용자 메모를 반영. 진입/퇴장은 맞고 동일인 연결은 틀림.',50:'하차 주장은 증거 부족으로 uncertain. 오른쪽 보행 쌍의 방향을 별도의 좌→우 보행자 GT로 대체하지 않음.',54:'빨간 차량에 관한 사용자 답변이 상충하여 해당 차량 주장을 uncertain 처리.',55:'빨간 차량은 uncertain, 자전거 이동은 실제이나 좌→우 방향은 오류.',73:'빨간 차량 존재/이동 충돌과 추가 차량 no를 구분.',72:'walking/running 엄격 구분을 원치 않는다는 사용자 지시에 따라 기존 running GT와 locomotion 계열로 일치.',87:'사용자가 확인한 머리 만지기를 gesturing 의미로 인정. 자전거도 실제 사건으로 인정.'}
def main():
    assert hashlib.sha256((ROOT/'CHECKLIST_ko.reviewed_snapshot.txt').read_bytes()).hexdigest()=='87d511f21dde2f02f27f9c2b5d540029e2185662abb5ea8d631d5ed96f037ed6', 'Different review snapshot: do not reuse authored judgments.'
    specs={int(line.split('|')[0]):line.split('|')[1:] for line in SPEC.strip().splitlines()}
    assert set(specs)==set(range(90))
    out=[]
    for i,row in enumerate(DATA['records']):
        vi=VIDEOS.index(row['video_id'])+1;claims,families=specs[i];pred=[]
        for token in claims.split():
            uncertain=token.startswith('~');n=int(token.lstrip('~'));rid=f'V{vi:02d}-{n:02d}';review=LEDGER[rid]
            status=review['resolved_status'];kind=review['semantic_kind']
            if uncertain or status=='uncertain':label='uncertain'
            elif status=='yes':label='supported'
            elif status=='corrected':label='attribute_error'
            elif status=='no':label='attribute_error' if kind=='attribute' else 'context_error' if kind=='context' else 'human_rejected_event'
            else:raise AssertionError(status)
            pred.append({'review_id':rid,'assertion_ko':review['statement_ko'],'kind':kind,'label':label,'semantic_assertion_uncertain':uncertain,'raw_human_answer':review['raw_answer'],'resolved_human_status':status,'reason_ko':review['resolution_ko'],'corrected_fact_ko':review.get('corrected_fact_ko'),'evidence':row['description'],'evidence_scope':'full_caption'})
        assert len({p['review_id'] for p in pred})==len(pred)
        fm={FAMILIES[t.lstrip('~')]:('uncertain' if t.startswith('~') else 'matched') for t in families.split()}
        target=TARGETS[row['video_id']]; assert set(fm)<=set(target['action_families']),(i,set(fm)-set(target['action_families']))
        judgments=[{'action_family':f,'label':fm.get(f,'missed'),'evidence':row['description'] if f in fm else '', 'evidence_scope':'full_caption','reason_ko':('설명 전체를 의미적으로 판정: 검토된 실제 행동 계열의 서술이 있음. 속성 오류는 별도로 평가.' if f in fm else '이 실제 행동 계열을 성립시키는 의미적 서술이 없음.')} for f in target['action_families']]
        ds=[{**t,'label':DIR.get(i,{}).get(t['review_id'],'missing'),'evidence':row['description'] if t['review_id'] in DIR.get(i,{}) else ''} for t in target['direction_targets']]
        ps=[{**t,'label':POL.get(i,{}).get(t['review_id'],'missing'),'evidence':row['description'] if t['review_id'] in POL.get(i,{}) else ''} for t in target['polarity_targets']]
        out.append({**row,'predicted_claims':pred,'action_judgments':judgments,'direction_judgments':ds,'polarity_judgments':ps,'notes_ko':NOTES.get(i,'행동 의미와 주체에 붙은 속성을 분리하여 판정. 원본 출처 목록을 기계적으로 정답 매칭에 사용하지 않음.')})
    (ROOT/'judge_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('Saved',len(out),'new LLM-authored caption judgments;',sum(len(r['predicted_claims']) for r in out),'reviewed propositions.')
if __name__=='__main__':main()
