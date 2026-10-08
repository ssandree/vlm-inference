#!/usr/bin/env python3
"""Render questions from saved semantic judgments and explicit review concerns."""
import csv,html,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
LABEL={'locomotion':'보행 또는 달리기','carrying_object':'물건을 들고 운반','driving':'차량 이동','parking':'주차','talking':'대화','cycling':'자전거 주행','entering_facility':'건물 안으로 들어감','exiting_facility':'건물 밖으로 나옴','getting_into_vehicle':'차량 탑승','getting_out_of_vehicle':'차량 하차','pushing_cart':'카트 밀기','unloading_object_from_vehicle':'차량에서 물건 꺼내기','loading_object_to_vehicle':'차량에 물건 싣기','putting_object_down':'물건 내려놓기','riding_scooter':'스쿠터 주행','opening_vehicle_trunk':'트렁크 열기','closing_vehicle_trunk':'트렁크 닫기','wandering':'배회','gesturing':'몸짓/손짓'}
D={'left_to_right':'좌→우','right_to_left':'우→좌','top_to_bottom':'상→하','bottom_to_top':'하→상','lower_left_to_upper_right':'좌하→우상','lower_right_to_upper_left':'우하→좌상','upper_left_to_lower_right':'좌상→우하','upper_right_to_lower_left':'우상→좌하'}
# Hand-authored detailed factual propositions, rather than automatic keyword extraction.
CUSTOM=[
(0,'차량 근처로 온 한 남자가 실제로 차량 안에서 밖으로 내린다.','주변에서 나타난 사람과 하차자를 구분. GT 하차 트랙/구간과 연결.'),
(0,'하차한 남자의 차량이 이후 출발한다.','하차자와 차량의 동일성 및 순서를 확인. 문장 own vehicle/pulls up/then departs의 연결이 모호함.'),
(1,'두 사람이 대화하다가 함께 화면 오른쪽으로 걸어간다.','동행과 대화는 별도 근거가 필요. 대화 식별 불가면 unsure.'),
(3,'나무 뒤 건물 내부의 문가에서 사람이 움직인다.','ROI의 크롭 위치가 실제 건물 내부인지 확인.'),
(3,'그 건물로 들어간 사람이 곧 다시 밖으로 나온다.','입장/퇴장 및 동일인 연결을 각각 확인.'),
(11,'유리문에서 한 사람이 나오고, 잠시 뒤 다른 두 사람이 건물 안으로 들어간다.','행동과 인원/순서를 분리하여 정정 가능.'),
(11,'마지막 근접 장면의 사람은 테이블 위 읽을거리를 읽고 있다.','시선만으로 읽기를 확정하지 말 것. 영상에 근접 장면이 실제 존재하는지도 확인.'),
(13,'한 여성이 잔디밭을 가로질러 테이블이 있는 벽돌 patio 방향으로 걷는다.','잔디와 포장 보행로, 인물 식별 및 경로를 확인.'),
(16,'영상에 사람이 수행하는 동적 행동이 전혀 없다.','full_frame은 정적 장면으로만 서술하지만 원본 GT에 출입 이벤트가 있음.'),
(21,'한 사람이 테이블 사이의 열린 광장을 실제로 달려간다.','주 평가에서는 보행/달리기를 통합하지만 동작 세부 정확도는 별도 확인.'),
(32,'배경 사람들이 실제 시설 문 경계를 통과한다.','어느 방향으로 들어가거나 나오는지, 계단/복도 이동인지 확인.'),
(44,'상단 중앙에서 도착한 흰 밴들이 주차 차량 사이에 정차한 뒤 계속 머문다.','잠깐 정차인지 주차인지, 한 대/여러 대인지 구분.'),
(47,'보행자 외에 자전거를 타는 사람이 있으며 운동장 울타리 옆 길을 지나간다.','걷는 사람, 자전거를 미는 사람, 카트를 미는 사람과 구분. 이전 검토와 동일 사건인지 확인.'),
(49,'검은 세단 트렁크에서 물건을 꺼낸다.','기존 GT 이벤트 목록이 비어 있음. 손/물체와 차량 후면 경계를 확인.'),
(49,'꺼낸 물건을 차량 옆 바닥에 내려놓는다.','물건 꺼내기와 내려놓기는 별도 구간으로 기록.'),
(49,'사람이 떠난 뒤 차량 뒷문 또는 트렁크가 열린 상태로 남는다.','rear door와 trunk를 구분. 열린 상태만으로 열기 동작을 추가하지 말 것.'),
(50,'보행자가 자전거 이용자와 함께 좌→우로 움직인다.','GT 보행 방향 우→좌/우하→좌상과 비교. 서로 다른 시간·주체인지 확인.'),
(51,'왼쪽으로 움직이는 사람은 자전거를 타는 대신 카트를 밀고 있다.','full_frame 자전거 서술과 같은 사람인지 먼저 확인. 물건을 직접 들고 운반하는 것과 구분.'),
(53,'두 사람이 함께 물건이 든 카트를 좌하 방향으로 민다.','동행 인원, 카트 여부, 화면 방향을 각각 확인.'),
(57,'은색 세단 옆으로 지나가는 사람이 가방을 들고 있다.','작은 물체/신체 일부와 구분.'),
(58,'주차장의 차량뿐 아니라 사람도 전혀 움직이지 않는다.','full_frame 정적 설명과 ROI 보행/자전거 묘사 비교.'),
(59,'두 사람이 돌 옆의 흰 상자에 접근하여 잠시 상호작용한다.','상자 조작인지 서로 대화하는지 단순 접근인지 구분.'),
(60,'좌하 구역의 이동자는 스쿠터를 타고 좌→우로 이동한다.','자전거/카트/보행 오인 여부 및 ROI 보행자와 동일인인지 확인.'),
(65,'하단 오른쪽의 사람이 노란 물체를 들고 흰 수거함 옆을 걷는다.','물체 존재와 색은 분리하여 기록.'),
(71,'은색 SUV에 접근한 남자가 차 안의 사람과 상호작용한 뒤 걸어서 떠난다.','탑승/하차와 구분. 단순 가까이 서 있는 상태를 상호작용으로 오인했는지 확인.'),
(71,'운동장 울타리 옆 두 사람은 화면 왼쪽으로 걷는다.','GT 우→좌 보행자와 대응하는지 확인.'),
(79,'주차장 근처 사람이 실제 차량 안에서 하차한다.','빈 GT이므로 추가 annotation 필요 여부 확인. full_frame에도 하차 서술 있음.'),
(79,'수거함에 접근한 사람이 상자를 들고 와 내려놓은 후 돌아선다.','carrying→placing→turning 동일인 연결과 작은 물체를 확인.'),
(79,'좌하의 나무 옆 그늘에서 사람이 물건이 든 카트를 민다.','별도 주체/영역인지 확인. 상자 운반자와 합치지 말 것.'),
(83,'자전거가 우→좌로 주행한다.','full_frame 좌→우와 반대. 같은 자전거/구간인지 확인.'),
(91,'Student Services 쪽 유리문 입구를 실제로 통과하는 사람이 있다.','문 앞을 지나가는 것인지 안으로 들어가는 것인지 확인.'),
(91,'꽃나무 옆 길에서 두 사람이 조깅한다.','빠른 보행과 구분. 주 점수에서는 보행/달리기 통합.'),
(97,'베이지색 키오스크 옆을 사람이 달려 지나간다.','full_frame 가방 운반 보행자와 같은 사람인지 확인.'),
(99,'세 사람이 오른쪽으로 향한 뒤 화면 왼쪽으로 나간다.','실제 방향 전환/시간 분리인지 문장 내부 경로 오류인지 확인.'),
(104,'한 사람이 유리문으로 들어간 뒤 잠시 후 다시 밖으로 나온다.','다른 사람을 동일인으로 연결했는지 확인.'),
(105,'입구에서 멀어지는 첫 사람은 화면 왼쪽으로 이동한다.','실제 문 통과도 보이는지 기록. 둘째 사람의 퇴장 방향에 이 방향을 전이하지 말 것.'),
(105,'다른 사람이 가방을 들고 문을 나온 후 오른쪽으로 돌아 화면 밖으로 나간다.','GT 우→좌 퇴장과 같은 주체인지, 퇴장 직후와 이후 회전 시점을 분리.'),
(110,'해당 영상에는 움직이는 사람이 없다.','ROI에는 빨간 옷의 사람이 초록 차량을 지나는 서술이 있음. GT 누락 영상.'),
(115,'파란 픽업트럭에서 두 사람이 내린다.','원본 GT는 탑승. 승차/하차 극성과 인원 검토.'),
(115,'방금 내린 두 사람이 다시 그 차량을 운전하여 떠난다.','명시되지 않은 재탑승을 사실로 보충하지 말 것. 다른 운전자일 가능성 확인.'),
(117,'초록 차량으로 걸어온 사람이 물러난 뒤 그 차량이 출발한다.','원본 탑승 GT와 비교. 작게 보이는 실제 승차가 누락되었는지 확인.')]
def main():
 rows=json.loads((ROOT/'per_video_details.json').read_text());lookup={r['source_index']:r for r in rows};gt=json.loads((ROOT/'inputs.json').read_text())['ground_truth'];items=[]
 def add(v,kind,priority,question,note,sources,refs=None):
  items.append({'video_id':v,'kind':kind,'priority':priority,'question_ko':question,'check_point_ko':note,'sources':sources,'gt_refs':refs or [],'answer':'unsure','start_sec':'','end_sec':'','actor_id':'','existing_gt_id':'','same_event_as':'','correction_ko':''})
 # Questions preserve distinct evidence clauses; only identical clauses are coalesced.
 for v in dict.fromkeys(r['video_id'] for r in rows):
  rs=[r for r in rows if r['video_id']==v];status=rs[0]['gt_status'];events=gt.get(v,{}).get('caption_events',[])
  pool={}
  for r in rs:
   for p in r['predicted_events']:
    if p['label']=='supported':continue
    key=(p['action'],p['evidence']);pool.setdefault(key,[]).append({'record_id':r['record_id'],'condition':r['condition'],'evidence':p['evidence'],'label':p['label']})
  for (action,evidence),sources in pool.items():
   add(v,'predicted_claim','high' if action!='locomotion' else 'normal',f"인용에 서술된 주체·장소에서 ‘{LABEL.get(action,action)}’ 행동이 실제로 보인다.",'아래 원문을 확인하고 해당 행동의 존재를 판정. 인원/색/방향/순서만 다르면 실제 행동은 별도로 기록하고 속성 오류를 수정란에 적는다. GT 비지원은 사실 오류 판정이 아니다.',sources)
  missing=sorted({j['action'] for r in rs for j in r['action_judgments'] if j['label'] in {'missed','uncertain'}})
  for a in missing:
   refs=[{'id':e['caption_event_id'],'original_action':e['action'],'start_sec':e['start_time'],'end_sec':e['end_time'],'tracks':e['source_track_ids']} for e in events if ('locomotion' if e['action'] in {'walking','running'} else e['action'])==a]
   add(v,'gt_omission','high',f"GT에 기록된 ‘{LABEL.get(a,a)}’ 행동이 해당 구간에서 실제로 확인된다.",'yes면 GT 누락 묘사인지 확인. no/unsure면 GT 자체 또는 관측 가능성을 점검. 여러 GT ID가 있으면 ID별 결과를 수정란에 구분하여 기록.',[{'record_id':r['record_id'],'condition':r['condition'],'evidence':'','label':next(j['label'] for j in r['action_judgments'] if j['action']==a)} for r in rs if any(j['action']==a and j['label'] in {'missed','uncertain'} for j in r['action_judgments'])],refs)
  ds=sorted({(j['action'],j['gt_direction']) for r in rs for j in r['direction_judgments'] if j['label']!='correct'})
  for a,d in ds:
   add(v,'direction','normal',f"GT의 ‘{LABEL.get(a,a)}’ 주체는 화면 {D[d]} 방향으로 움직인다.",'GT 트랙/구간에 연결. 다른 주체의 방향을 가져오지 말 것. 계단 상하와 화면 상하를 구분.',[{'record_id':r['record_id'],'condition':r['condition'],'evidence':j['evidence'],'label':j['label']} for r in rs for j in r['direction_judgments'] if (j['action'],j['gt_direction'])==(a,d)], [{'id':e['caption_event_id'],'start_sec':e['start_time'],'end_sec':e['end_time'],'tracks':e['source_track_ids']} for e in events if ('locomotion' if e['action'] in {'walking','running'} else e['action'])==a and e['direction']==d])
 for i,q,note in CUSTOM:
  r=lookup[i];add(r['video_id'],'detail','high',q,note,[{'record_id':r['record_id'],'condition':r['condition'],'evidence':r['description'],'label':'unverified_detail'}])
 ordered=[];text=['59-video 8b-fp16 영상별 검토 체크리스트','',
 '각 항목의 원문/주체/장소를 확인하고 yes/no/unsure로 답하세요. 초기값은 모두 unsure입니다. 원본 영상을 assistant가 검증한 결과가 아닙니다.',
 '행동 자체와 속성 오류를 구분하세요. 물건 운반은 실제지만 색/방향이 틀린 경우 행동 사실과 정정 내용을 따로 기록합니다. 시간은 원본 클립 기준 초; 모르면 공란. GT 시간은 annotation 참고값입니다.',
 '동일 사건은 same_event_as로 연결하고 기존 CE ID를 적으세요. 새로운 사실은 자동 GT 추가하지 않습니다. 다른 모델 설명에서 확인했던 답을 새 주체/서사에 자동 적용하지 않았습니다.',
 'HTML과 TSV는 별도 편집 경로입니다. HTML은 로컬 임시 저장을 시도하지만 JSON 내보내기를 사용하세요. HTML 답은 서버 파일/GT를 자동 변경하지 않습니다.',
 'high부터 검토한 뒤 normal 방향/보행을 확인하세요. detail은 인원/경로/서사 문제이며 predicted_claim/gt_omission과 같은 사건일 수 있습니다. 항목 수는 독립 사건 수가 아닙니다.','']
 for vi,v in enumerate(dict.fromkeys(r['video_id'] for r in rows),1):
  status=next(r['gt_status'] for r in rows if r['video_id']==v)
  text+=['',f'V{vi:02d} {v} [{status}]']
  group=sorted([i for i in items if i['video_id']==v],key=lambda i:(i['priority']!='high',i['kind']))
  for k,it in enumerate(group,1):
   it={'review_id':f'V{vi:02d}-{k:02d}',**it};ordered.append(it)
   text += [f"[unsure] {it['review_id']} [{it['priority']}/{it['kind']}] {it['question_ko']}",f"  확인 포인트: {it['check_point_ko']}"]
   for s in it['sources']:text += [f"  출처: {s['record_id']} ({s['condition']}, {s['label']})",'  원문: '+s['evidence']] if s['evidence'] else [f"  출처: {s['record_id']} ({s['condition']}, {s['label']})"]
   if it['gt_refs']:text += ['  GT 참조: '+json.dumps(it['gt_refs'],ensure_ascii=False)]
   text+=['  시간(초): ____ ~ ____ / 주체: ____ / CE ID: ____ / same_event_as: ____','  수정/추가 메모: ']
  text+=['  목록 밖 추가 사건(행동/시간/주체/방향): ']
 (ROOT/'CHECKLIST_ko.txt').write_text('\n'.join(text)+'\n')
 (ROOT/'review_candidates.json').write_text(json.dumps(ordered,ensure_ascii=False,indent=2)+'\n')
 with (ROOT/'review_answers.tsv').open('w',newline='') as f:
  fields=['review_id','video_id','kind','priority','question_ko','answer','start_sec','end_sec','actor_id','existing_gt_id','same_event_as','correction_ko'];w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows({k:i[k] for k in fields} for i in ordered)
 context={v:{'gt_status':rs[0]['gt_status'],'captions':[{'record_id':r['record_id'],'condition':r['condition'],'description':r['description']} for r in rs]} for v in dict.fromkeys(r['video_id'] for r in rows) if (rs:=[r for r in rows if r['video_id']==v])}
 payload=json.dumps({'items':ordered,'context':context},ensure_ascii=False).replace('<','\\u003c')
 template=(ROOT/'review_template.html').read_text();(ROOT/'review.html').write_text(template.replace('__PAYLOAD__',payload))
 result={'n_videos':len(context),'n_questions':len(ordered),'by_kind':dict(Counter(i['kind'] for i in ordered)),'by_priority':dict(Counter(i['priority'] for i in ordered)),'initial_answers':dict(Counter(i['answer'] for i in ordered))}
 assert len({i['review_id'] for i in ordered})==len(ordered)
 assert {i['video_id'] for i in ordered}==set(context)
 (ROOT/'review_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
