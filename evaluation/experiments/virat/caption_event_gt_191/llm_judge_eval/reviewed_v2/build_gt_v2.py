#!/usr/bin/env python3
"""Ingest user answers; apply explicitly documented semantic resolutions.
Text parsing only locates review IDs/checkboxes. No lexical caption matching.
"""
from pathlib import Path
import json,hashlib,copy,collections
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
OUT=BASE/'gt_human_review_v2'
CHECK=ROOT.parent/'human_review/CHECKLIST_ko.txt'
def dump(p,obj): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def key(v,n): return f'V{v:02d}-{n:02d}'
# Explicit semantic type assignment, authored by the LLM, not inferred from words.
TYPES={
1:'driving parking @turn @turn @parking_geometry driving @stationary getting_out_of_vehicle @count locomotion opening_vehicle_door retrieving_object @object_type closing_vehicle_door carrying_object @direction @direction @direction locomotion @object_type @object_type actor_presence locomotion @count driving parking driving driving @repetition driving',
2:'locomotion @stair_direction locomotion @route @destination locomotion @destination facility_transition @count context context',
3:'locomotion locomotion locomotion carrying_object locomotion locomotion @group_association facility_transition talking posture_change manipulating_object gesturing context',
4:'locomotion locomotion locomotion locomotion @direction locomotion @direction facility_transition facility_transition facility_transition @identity talking context',
5:'@stair_direction locomotion locomotion locomotion locomotion @sequence facility_transition facility_transition @sequence @identity locomotion locomotion locomotion locomotion',
6:'@direction locomotion group_separation stopping locomotion locomotion looking_around locomotion locomotion cycling cycling cycling getting_out_of_vehicle driving @direction gesturing',
7:'driving parking driving driving @direction driving @turn driving driving driving cycling @direction @direction locomotion locomotion @group_association @group_association locomotion locomotion carrying_object facility_transition carrying_object locomotion locomotion',
8:'locomotion @destination @polarity facility_transition carrying_object @direction driving @direction context',
9:'driving @direction locomotion getting_into_vehicle locomotion driving @turn locomotion locomotion context'}
RESOLUTIONS={
'V04-09':('yes','[no]보다 후속 구체 메모 "들어가긴하는데 ... 작다"를 우선: 실제 진입을 확인한 것으로 반영. visibility=small/difficult; GT 제외하지 않음.'),
'V05-06':('corrected','yes 표시는 왼쪽 이동을 확인하지만 메모는 접근/되돌아섬을 부정. 올바른 사실은 건물에서 나온 뒤 왼쪽으로 이동. 원래 문장의 주체 연결은 오서술.'),
'V06-13':('uncertain','no와 함께 "뒤쪽에서 나오긴 했는데 내린다는 증거는 없음"이라고 명시. 하차 부재 확정이 아니라 관측 증거 부족으로 처리.'),
'V07-04':('uncertain','빨간 차량 이동 no와 V07-05의 동일 차량 좌→우 이동 yes가 논리적으로 충돌. 색상/주체/위치 중 무엇이 틀렸는지 불명확.'),
'V07-05':('uncertain','V07-04와 함께 충돌 묶음으로 보류. 좌→우 차량 GT를 확정 생성하지 않음.'),
'V07-02':('no','최신 체크리스트 no를 이전 대화의 주차 yes보다 우선 적용.'),
'V03-07':('no','이 설명의 함께 걷는 두 사람은 사용자가 부정. 기존 그룹 CE017/CE018 전체 삭제까지 의미한다고 확대 해석하지 않고 충돌 플래그를 남김.'),
'V01-28':('yes','사용자가 픽업트럭과 같은 차량이라고 확인: V01-25와 동일 의미 사건으로 병합.'),
'V02-06':('yes','화단으로 향함은 yes, 잔디 횡단은 no. 화단 방향 이동만 반영하고 잔디 경로는 부여하지 않음.'),
'V07-24':('yes','사용자의 walking/running을 강하게 구분하지 말라는 지시 반영. 모든 영상에서 두 행동을 locomotion 계열로 통합하되 원본 action은 보존.')}
# Direction facts and their action anchors, explicitly interpreted by LLM.
DIRECTIONS={
'V01-17':('locomotion','rightward_halfplane','V01-17'),
'V01-18':('locomotion','downward_halfplane','V01-17'),
'V03-06':('locomotion','left_to_right','V03-06'),
'V04-02':('locomotion','right_to_left','V04-02'),
'V05-03':('locomotion','left_to_right','V05-03'),
'V05-06':('locomotion','right_to_left','V05-06'),
'V06-02':('locomotion','right_to_left','V06-02'),
'V06-05':('locomotion','left_to_right','V06-05'),
'V07-12':('cycling','right_to_left','V07-11'),
'V07-24':('locomotion','left_to_right','V07-24'),
'V08-08':('driving','right_to_left','V08-07'),
'V09-02':('driving','left_to_right','V09-01')}
LINKS={'V01-08':'CE001','V02-08':'CE001','V04-09':'CE001','V04-10':'CE002','V05-07':'CE002','V05-08':'CE001','V06-16':'CE005','V07-24':'CE004','V08-04':'CE001','V09-04':'CE001'}
POLARITIES={'V02-08':'exit','V03-08':'exit','V04-09':'enter','V04-10':'exit','V05-07':'exit','V05-08':'enter','V08-04':'exit'}
def family(action): return 'locomotion' if action in ['walking','running'] else 'facility_transition' if action in ['entering_facility','exiting_facility'] else action

def main():
    text=CHECK.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()=='87d511f21dde2f02f27f9c2b5d540029e2185662abb5ea8d631d5ed96f037ed6', 'Checklist changed: requires a fresh semantic review of corrections before regenerating v2.'
    (ROOT/'CHECKLIST_ko.reviewed_snapshot.txt').write_text(text)
    originals=json.loads((ROOT.parent/'inputs.json').read_text())
    candidates=json.loads((ROOT.parent/'human_review/review_candidates.json').read_text())['claims']
    byid={c['review_id']:copy.deepcopy(c) for c in candidates}
    blocks={};curr=None
    for line in text.splitlines():
        if line.startswith('[') and '] V' in line:
            answer,rest=line[1:].split('] ',1);rid=rest.split()[0]
            assert rid in byid and rid not in blocks
            curr=rid;blocks[rid]={'raw_answer':answer.strip(),'lines':[line]}
        elif curr:
            if line.startswith('VIRAT_') or line.startswith('추가 발견:'): curr=None
            else: blocks[curr]['lines'].append(line)
    assert len(blocks)==len(byid)==140
    reviews=[]
    for rid,c in byid.items():
        b=blocks[rid];vi=int(rid[1:3]);n=int(rid[4:]);typ=TYPES[vi].split()[n-1]
        raw=b['raw_answer']; assert raw in ['yes','no']
        status,reason=RESOLUTIONS.get(rid,(raw,'사용자의 체크리스트 답변을 그대로 반영.'))
        c['candidate_priority_at_generation']=c.pop('priority')
        c.update({'answer':raw,'answer_source':'user_edited_CHECKLIST_ko.txt','correction_or_evidence':reason,'raw_answer':raw,'resolved_status':status,'resolution_ko':reason,'review_block':'\n'.join(b['lines']).strip(),'semantic_kind':'attribute' if typ.startswith('@') else 'context' if typ=='context' else 'event','action_family':typ if not typ.startswith('@') else None,'attribute_type':typ[1:] if typ.startswith('@') else None,'provenance':{'source_file':str(CHECK),'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'reviewer':'user','assistant_video_verification':False}})
        if rid=='V05-06': c['corrected_fact_ko']='사람이 건물에서 나온 뒤 화면 왼쪽으로 걸어간다.'
        reviews.append(c)
    dump(OUT/'review_ledger.json',reviews)
    dump(OUT/'resolutions.json',[{'review_id':rid,'status':s,'reason_ko':r} for rid,(s,r) in RESOLUTIONS.items()])
    byid={r['review_id']:r for r in reviews}; gtout={}
    for video,d in originals['ground_truth'].items():
        events=copy.deepcopy(d['caption_events']); rr=[r for r in reviews if r['video_id']==video]
        for e in events:
            e['action_family']=family(e['action']);e['provenance_layer']='original_gt';e['review_links']=[]
            if video=='VIRAT_S_010003_03_000219_000259' and e['caption_event_id'] in ['CE017','CE018']:
                e['review_status']='conflict_requires_actor_alignment';e['review_note']='V03-07 rejects described pair; cannot decide which original group. Excluded from group-specific scoring; original kept.'
        for r in rr:
            if r['resolved_status'] not in ['yes','corrected']: continue
            rid=r['review_id']
            if rid in LINKS:
                e=next(e for e in events if e['caption_event_id']==LINKS[rid]);e['review_links'].append(rid)
                if rid=='V04-09': e['visibility_note']='human_confirmed_but_very_small'
                continue
            if r['semantic_kind']!='event' and rid!='V05-06': continue
            if rid=='V01-28': continue
            act='locomotion' if rid=='V05-06' else r['action_family']
            actor='vehicle' if act in ['driving','parking'] else 'person'
            ev={'caption_event_id':'HE_'+rid,'event_form':'human_review_semantic_event','action':act,'action_family':act,'subject':{'category':actor,'track_id':None,'identity_verified':False},'source_track_ids':[],'start_time':None,'end_time':None,'start_frame':None,'end_frame':None,'direction':DIRECTIONS[rid][1] if rid in DIRECTIONS else None,'object':[],'reference_sentence':r.get('corrected_fact_ko',r['statement_ko']),'review_links':[rid]+(['V01-28'] if rid=='V01-25' else []),'provenance_layer':'user_video_review','review_status':'confirmed','temporal_localization':'not_provided','deduplication_status':'semantic_context_known_actor_time_unresolved','source_checklist_sha256':r['provenance']['source_sha256']}
            if rid in POLARITIES:ev['transition_polarity']=POLARITIES[rid]
            events.append(ev)
        if video==originals['records'][4]['video_id']:
            events.append({'caption_event_id':'HE_V01_departure_walk','event_form':'human_review_semantic_event','action':'locomotion','action_family':'locomotion','subject':{'category':'person','track_id':None,'identity_verified':False},'source_track_ids':[],'start_time':None,'end_time':None,'start_frame':None,'end_frame':None,'direction':None,'confirmed_direction_claims':['V01-17','V01-18'],'object':[],'reference_sentence':'하차자가 차량에서 멀어져 보도 쪽으로 걷는다. 오른쪽/아래쪽 이동은 각각 확인됐지만 구간 구분은 미제공.','review_links':['V01-17','V01-18'],'provenance_layer':'user_video_review','review_status':'confirmed','temporal_localization':'not_provided'})
        gtout[video]={'video_id':video,'version':'human_review_v2','caption_events':events,'confirmed_attributes':[r for r in rr if r['semantic_kind']=='attribute' and r['resolved_status']=='yes'],'confirmed_context':[r for r in rr if r['semantic_kind']=='context' and r['resolved_status']=='yes'],'rejected_claims':[r for r in rr if r['resolved_status']=='no'],'uncertain_claims':[r for r in rr if r['resolved_status']=='uncertain'],'corrected_claims':[r for r in rr if r['resolved_status']=='corrected'],'exhaustive_for_all_visible_events':False,'evaluation_scope':'human-reviewed semantic events, action families and reviewed claims; not temporally localized instance detection'}
        dump(OUT/'gt'/video/'caption_events.json',gtout[video])
    dump(OUT/'evaluation_gt_v2.json',gtout)
    # Nonduplicated action-family targets, and human-verified direction/polarity facts.
    targets={}
    for video,d in gtout.items():
        fs=sorted({e['action_family'] for e in d['caption_events']})
        dd=[{'review_id':rid,'action_family':a,'direction':di,'anchor':anchor,'note':'separate confirmed direction statements; do not synthesize diagonal'} for rid,(a,di,anchor) in DIRECTIONS.items() if byid[rid]['video_id']==video]
        if video=='VIRAT_S_010003_03_000219_000259':
            dd += [{'review_id':rid,'action_family':a,'direction':di,'anchor':'original_individual_GT','note':'기존 개인 GT의 방향 보존. 거부된 그룹 설명을 개인 GT로 대체 매칭하지 않음.'} for rid,a,di in [('GT03_carry_ltr','carrying_object','left_to_right'),('GT03_walk_diagonal','locomotion','lower_left_to_upper_right'),('GT03_walk_rtl','locomotion','right_to_left')]]
        pp=[{'review_id':rid,'action_family':'facility_transition','polarity':po} for rid,po in POLARITIES.items() if byid[rid]['video_id']==video]
        targets[video]={'action_families':fs,'direction_targets':dd,'polarity_targets':pp}
    dump(OUT/'scoring_targets.json',targets)
    summary={'version':'human_review_v2','checklist_sha256':hashlib.sha256(text.encode()).hexdigest(),'raw_answers':dict(collections.Counter(r['raw_answer'] for r in reviews)),'resolved_statuses':dict(collections.Counter(r['resolved_status'] for r in reviews)),'n_original_events':36,'n_added_semantic_events':sum(sum(e['provenance_layer']=='user_video_review' for e in d['caption_events']) for d in gtout.values()),'n_family_targets':sum(len(t['action_families']) for t in targets.values()),'n_direction_targets':sum(len(t['direction_targets']) for t in targets.values()),'n_polarity_targets':sum(len(t['polarity_targets']) for t in targets.values()),'original_gt_modified':False,'source_review_modified':False,'unknown_time_or_track_not_fabricated':True}
    dump(OUT/'manifest.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
