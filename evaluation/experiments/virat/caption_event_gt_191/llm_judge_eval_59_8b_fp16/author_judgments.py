#!/usr/bin/env python3
"""Serialize this conversation's manually authored semantic judgments. No lexical judge or API."""
import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/ssandree/Motion-guided-ROI/vlm-inference/outputs/fullframe_proposed_roi_59_8b_fp16/results.json')
GT=ROOT.parent/'gt'
# Explicit semantic extraction by the session LLM. Each excerpt is checked verbatim.
# Codes are output labels, not rules for searching/classifying caption words.
NAMES={'w':'walking','r':'running','c':'carrying_object','d':'driving','p':'parking','t':'talking','b':'cycling','e':'entering_facility','x':'exiting_facility','in':'getting_into_vehicle','out':'getting_out_of_vehicle','push':'pushing_cart','unload':'unloading_object_from_vehicle','put':'putting_object_down','s':'riding_scooter'}
S={
0:[('d','another vehicle drives past'),('w','pedestrians walk along sidewalks nearby'),('out','One man exits his own vehicle')],
1:[('t','engaging in conversation'),('w','walk away together toward the right side of the frame'),('d','both vehicles and people navigating through the open space','uncertain')],
2:[('d','A silver sedan drives into the parking lot'),('p','parks near the center-left section'),('out','A person exits the vehicle'),('w','walks toward the right side of the frame'),('in','entering and exiting their parked automobiles')],
3:[('d','A silver sedan drives through the parking lot entrance'),('w','pedestrians walk along the sidewalk'),('t','appearing engaged in conversation','uncertain'),('x','Another individual exits this same building shortly after entering'),('e','Another individual exits this same building shortly after entering')],
4:[('d','A black pickup truck drives past'),('w','pedestrians walk along the sidewalk')],
5:[('d','A black pickup truck drives past'),('w','pedestrians walk along the sidewalk')],
6:[('d','A black car drives past'),('w','pedestrians can be seen walking')],
7:[('d','A black car drives along the road'),('w','Two individuals walk across the parking lot’s open space')],
8:[('w','others walk along pathways')],
9:[('w','A person walks across the lower patio'),('c','A person walks across the lower patio carrying something')],
10:[('w','others walk around the plaza')],
11:[('w','A person walks across the foreground'),('x','someone exits through glass doors onto the patio'),('e','two more individuals entering the building entrance nearby')],
12:[('w','A person walks down the stairs')],
13:[('w','A woman walks across a grassy lawn')],
14:[('w','A person walks from left to right'),('x','Another individual exits through the glass doors into the courtyard space'),('t','some engaged in conversation')],
15:[('w','A person walks down a set of outdoor stairs'),('e','A few individuals enter through glass doors'),('x','while others exit toward the left')],
16:[],
17:[('w','A person walks down a set of outdoor stairs'),('x','another individual exits a building entrance nearby')],
18:[('w','A person walks down the stairs')],
19:[('w','A person wearing a pink shirt walks across the patio'),('x','another individual exits through glass doors nearby')],
20:[('w','A person walks from left to right across the foreground plaza')],
21:[('w','people walk around near tables and buildings'),('r','Another individual runs through the open plaza')],
22:[('w','people walking through the area')],
23:[('w','A person walks briskly past several tables'),('e','someone enters the building’s main entrance'),('x',"others entering and leaving the facility's doors")],
24:[('w','A few people walk up and down the stairs')],
25:[('w','A person walks across the paved patio area'),('e','People walk toward and enter the building entrance')],
26:[('w','A person walks down the stairs')],
27:[('w','people walk across the paved areas')],
28:[('w','A person walks from left to right along the pathway')],
29:[('w','two individuals walk across the paved area'),('c','another person walks past on the far side of the lawn toward the building entrance while carrying something blue')],
30:[('w','A person walks down the stairs')],
31:[('w','People walk up and down the stairs')],
32:[('w','A person walks down the stairs'),('x','passing through doorways within the building complex','uncertain'),('e','passing through doorways within the building complex','uncertain')],
33:[('w','People walk up and down stairs'),('t','engaged in conversation')],
34:[('w','A person walks down the stairs')],
35:[('w','people walk past them'),('t','engaged in conversation')],
36:[('w','A person walks down the stairs')],
37:[('w','A person walks down the steps'),('t','engaged in conversation'),('x','someone exits through a doorway located beside the staircase')],
38:[('w','A person walks down stairs')],
39:[('w','People walk across the paved courtyard')],
40:[('w','People walk up and down stairs')],
41:[('w','People walk across the courtyard')],
42:[('w','others walk along the pathways')],
43:[('w','A person walks across the foreground')],
44:[('d','A black sedan drives through a parking lot'),('w','Two individuals walk together'),('p','stopping among other parked vehicles without any further movement observed after their arrival')],
45:[('w','Two individuals walk across the lower left corner'),('d','several cars are seen driving along the road')],
46:[('w','Two individuals walk together through an empty parking lot toward the right side')],
47:[('w','A person walks across the parking lot'),('b','another individual rides a bicycle')],
48:[('w','then steps back away from the vehicle')],
49:[('unload','retrieves items from the trunk'),('put','places them on the ground beside the vehicle'),('w','then walks away toward the left side of the frame'),('b','two individuals are seen riding bicycles')],
50:[('b','A person rides a bicycle'),('w','another individual walks nearby')],
51:[('push','a person pushes a cart across the asphalt'),('w','people walk along a sidewalk')],
52:[('b','A person rides a bicycle'),('w','two individuals walk together')],
53:[('b','a person rides a bicycle'),('w','two individuals walk together'),('push','pushing a cart with items')],
54:[('w','A person walks along the sidewalk'),('d','A white van drives through an intersection')],
55:[('w','two individuals walk along the sidewalk'),('b','another person rides a bicycle')],
56:[('w','A person walks from right to left')],
57:[('w','a person walking past it'),('c','a person walking past it carrying bags'),('d','A gray SUV drives through the parking lot')],
58:[],
59:[('b','several people are seen riding bicycles'),('w','walking away together toward the right side of the frame')],
60:[('s','A person on a scooter moves from left to right')],
61:[('w','two individuals are seen walking across parked cars')],
62:[('w','A person walks from right to left')],
63:[('w','A person walks across a parking lot'),('b','another individual rides a bicycle')],
64:[('w','A person walks from the bottom right corner')],
65:[('w','two individuals walk across the asphalt'),('c','A person carrying a yellow object walks past a white dumpster')],
66:[('w','A person walks from left to right')],
67:[('w','Two individuals walk across the parking lot'),('c','walking diagonally towards the upper-right section of the parking lot while carrying something dark-colored')],
68:[('d','A white van drives through the parking lot'),('w','Two pedestrians walk along the sidewalk')],
69:[('d','A white van drives through the parking lot entrance'),('w','A person walks across the foreground')],
70:[('w','A person walks from left to right')],
71:[('w','two individuals walk across the asphalt')],
72:[('d','A silver sedan drives into the parking lot'),('p','parks near other vehicles')],
73:[('d','several cars are seen moving through designated lanes')],
74:[('d','A red car drives slowly through the parking lot')],
75:[('d','A red car drives through the parking lot'),('w','A person walks away from the edge of the frame')],
76:[('b','A person rides a bicycle'),('w','another individual walks briskly')],
77:[('w','two individuals walk across the asphalt'),('b','another person rides a bicycle')],
78:[('w','A person walks from left to right'),('out','another individual exits a vehicle')],
79:[('w','two individuals walk across the asphalt'),('out','another person exits a vehicle nearby'),('d','A silver sedan drives through the central lanes'),('push','someone pushes a cart with items'),('c','Another individual approaches the dumpster carrying boxes'),('put','then turns away after placing them down')],
80:[('w','A person walks steadily')],
81:[('w','A person walks across the parking lot'),('b','another individual rides a bicycle'),('d','A white van drives past')],
82:[('b','A person rides a bicycle')],
83:[('w','two individuals are seen walking along the sidewalk'),('b','A cyclist rides across the open space')],
84:[('w','A person walks past the Student Services building')],
85:[('w','A person wearing a green shirt walks across the paved courtyard'),('c','someone else moves past while carrying something that appears dark-colored')],
86:[('w','People walk along a paved pathway')],
87:[('w','Students walk along paved pathways')],
88:[('w','A person walks along a paved pathway')],
89:[('w','People walk along a paved pathway'),('e','some entering through glass doors')],
90:[('w','A woman walks from right to left'),('x',"another person exits the building's entrance nearby")],
91:[('w','People walk along a paved pathway'),('e','some pass through the entrance'),('r','two people jogging briskly')],
92:[('w','A person walks along a paved pathway')],
93:[('w','A person walks along a paved pathway')],
94:[('w','A person walks along a paved pathway')],
95:[('w','Two individuals walk along a paved pathway')],
96:[('w','A person walks from right to left'),('c','while carrying bags')],
97:[('r','a person runs past a beige kiosk'),('w','several individuals walk along the sidewalk')],
98:[('w','A person walks away from the building entrance'),('c','another individual approaches from the left carrying a backpack')],
99:[('w','Three individuals walk along a paved pathway'),('c','A person wearing shorts and carrying a backpack walks past a beige utility box')],
100:[('w','A person walks along a paved pathway')],
101:[('w','A person walks along a paved pathway')],
102:[('w','People walk along a paved pathway'),('t','others stand nearby conversing')],
103:[('w','Students walk across the campus courtyard'),('t','others stand conversing nearby')],
104:[('w','walking away from the building along a paved pathway'),('e','enters through the glass doors'),('x','then exits shortly after')],
105:[('w','A person walks away from the entrance'),('x','Another individual exits through the same doorway carrying bags'),('c','Another individual exits through the same doorway carrying bags')],
106:[('d','A red car drives through an intersection')],
107:[('d','two red cars drive past parked vehicles')],
108:[('d','A red car drives through an intersection')],
109:[('d','A red convertible drives through a parking lot intersection'),('w','A person walks along a sidewalk')],
110:[],
111:[('w','A person wearing red walks past a parked green car')],
112:[('w','two individuals walk from right to left')],
113:[('w','A person walks across a paved parking lot'),('c','while carrying a bag')],
114:[('d','A blue car drives through an intersection')],
115:[('out','two individuals exit the vehicle'),('d','then drive away through a landscaped roundabout'),('w','pedestrians walk nearby')],
116:[('d','A black car drives through an intersection')],
117:[('w','A person walks toward a parked green car'),('d','the car drives off through a landscaped roundabout')]
}
# Exact action/direction decisions after reading entire caption. Unlisted GT directions are missing.
DIR={
46:{('walking','left_to_right'):('correct','Two individuals walk together through an empty parking lot toward the right side','보행의 오른쪽 이동이 명시됨.')},
50:{('walking','right_to_left'):('contradictory','both moving from left to right','보행자를 포함한 두 주체가 좌→우로 이동한다고 명시하여 반대 방향.'),('walking','lower_right_to_upper_left'):('contradictory','both moving from left to right','서술된 보행은 좌→우이며 GT 우하→좌상과 불일치.')},
71:{('walking','right_to_left'):('correct','two people stroll past parked cars heading towards the left edge of the frame','좌측을 향한 보행이 명시됨.')}
}
NOTES={0:'entrances/pathways는 시설 통과로 확정하지 않음. 차량에서 내림은 명시됨.',1:'차량과 사람이 공간을 이동한다는 추상 문장은 차량 이동인지 불확실.',8:'계단이 건물로 이어진다는 위치 설명만으로 입장으로 인정하지 않음.',10:'walk around는 단순 보행이며 반복 배회 또는 달리기를 별도로 확정하지 않음.',13:'전화에 몰두했다는 상태는 실제 휴대전화 조작 동작을 확정하지 않음.',16:'정적인 장면 설명만 있어 핵심 동적 예측 0개. precision을 100%로 채우지 않음.',24:'계단의 상하 이동은 건물 출입을 입증하지 않음.',26:'건물에서 이어지는 계단을 내려감은 문 통과를 명시하지 않음.',32:'doorways 통과는 명시되지만 시설 경계 및 입장/퇴장 극성이 불명확. 두 극성 모두 uncertain으로 표시하며 사실 환각으로 확정하지 않음.',42:'계단이 건물로 이어진다는 표현은 입장 동작이 아님.',44:'도착 후 주차 차량 사이에 계속 멈춰 있다는 구절은 주차 의미로 판정.',46:'다른 보행자도 있어 좌→우 구절만으로 우→좌 GT를 모순 처리하지 않음.',49:'trunk에서 꺼내기는 명시되나 이미 열린 트렁크인지 알 수 없어 opening은 추론하지 않음.',51:'카트 밀기는 물체를 직접 들고 운반하는 carrying과 구분. 보행 방향은 별도 보행자에 연결되며 GT 방향 근거 없음.',59:'engage briefly는 대화/조작의 구체 행동을 확정하기 부족.',70:'다른 사람이 접근하는 구절은 방향 불명. 한 사람 좌→우를 전체 우→좌 GT의 모순으로 확정하지 않음.',71:'차량 내부 사람과 상호작용했다는 말은 승차/대화를 구체적으로 확정하지 않음.',79:'세단이 잠시 정차했다는 말만으로 주차를 추가하지 않음.',91:'유리문으로 이어지는 길에서 입구를 통과한다는 문맥을 입장으로 판정. 검토 필요.',98:'입구에서 멀어짐은 문 통과와 다름.',99:'오른쪽으로 향했다가 왼쪽으로 화면 이탈한다는 경로의 동일 주체/시간 연결 확인 필요.',104:'입장 후 퇴장은 명시되나 화면 방향 없음.',105:'첫 사람의 입구에서 좌측으로 멀어짐은 시설 퇴장 경계를 확정하지 않음. 명시적 퇴장자는 다른 사람이며 나중 우회전만 있어 GT 퇴장 방향은 missing.',115:'두 사람이 하차 후 운전해 떠나는 문장 연결은 확인 필요. 명시하지 않은 재탑승은 추론하지 않음.'}
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 runs=json.loads(SOURCE.read_text())['runs']; assert set(S)==set(range(len(runs)))
 records=[]; ground={}; judgments=[]; sources=[SOURCE,ROOT.parent/'annotation_policy.json']
 for i,r in enumerate(runs):
  assert not r.get('error'); cap=r['generated_description']; v=r['video_id']; p=GT/v/'caption_events.json'
  if p.exists() and v not in ground: ground[v]=json.loads(p.read_text());sources.append(p)
  ev=ground.get(v,{}).get('caption_events',[]); acts=sorted({e['action'] for e in ev})
  status='missing_gt' if not p.exists() else 'empty_gt' if not ev else 'nonempty_gt'
  row={'record_id':f'R{i:03d}','source_index':i,'video_id':v,'model':r['model'],'condition':r['condition'],'description':cap,'gt_status':status,'inference_time_sec':r.get('inference_time_sec'),'visual_tokens':r.get('visual_tokens')};records.append(row)
  pred=[]
  for spec in S[i]:
   code,evidence,*unc=spec;action=NAMES[code];assert evidence in cap,(i,evidence)
   label='unscored_missing_gt' if status=='missing_gt' else 'uncertain' if unc else 'supported' if action in acts else 'hallucinated_gt'
   reason='GT 파일이 없어 채점 제외.' if status=='missing_gt' else '문맥상 행동/출입 극성이 불확실.' if unc else '원본 GT의 동일 행동 의미에 대응.' if action in acts else '원본 GT에 이 행동이 없음. 실제 오류인지는 영상 검토 필요.'
   pred.append({'action':action,'label':label,'evidence':evidence,'reason_ko':reason,'semantic_uncertain':bool(unc),'factual_status':'uncertain_unverified' if label!='supported' else 'gt_action_supported_only'})
  assert len({p['action'] for p in pred})==len(pred)
  aj=[]
  for a in acts:
   match=next((p for p in pred if p['action']==a),None)
   aj.append({'action':a,'label':'uncertain' if match and match['semantic_uncertain'] else 'matched' if match else 'missed','evidence':match['evidence'] if match else '', 'reason_ko':('해당 행동 또는 출입 극성의 문맥이 불명확.' if match['semantic_uncertain'] else '해당 행동이 명시적으로 서술됨.') if match else '설명 전체에서 이 행동을 확정하는 구절 없음. 접근/계단/화면 이탈로 추론하지 않음.'})
  dj=[]
  for a,d in sorted({(e['action'],e['direction']) for e in ev if e['direction']}):
   lab,evidence,reason=DIR.get(i,{}).get((a,d),('missing','','동일 행동에 연결된 해당 화면 이동 방향이 명시되지 않음.'))
   assert not evidence or evidence in cap
   dj.append({'action':a,'gt_direction':d,'label':lab,'evidence':evidence,'reason_ko':reason})
  judgments.append({'record_id':row['record_id'],'action_judgments':aj,'direction_judgments':dj,'predicted_events':pred,'notes_ko':NOTES.get(i,'')})
 dump('inputs.json',{'records':records,'ground_truth':ground});dump('judge_results.json',judgments)
 dump('input_manifest.json',{'source_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]})
 print('Serialized',len(records),'captions,',len(ground),'GT files')
if __name__=='__main__':main()
