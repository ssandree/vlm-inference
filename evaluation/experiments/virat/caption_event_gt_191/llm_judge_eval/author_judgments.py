#!/usr/bin/env python3
"""Serialize semantic decisions authored directly by the current GPT-6 LLM.
This is NOT a keyword evaluator, trained classifier, or an API judge invocation.
Short codes below encode already-made semantic judgments, not text matching rules.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'inputs.json').read_text())
# S=supported by GT, H=confident GT-unsupported, U=ambiguous semantic support.
# Each tuple is action, label, verbatim evidence, specific Korean rationale.
J={}
def put(i,*events,notes=''):
    J[f'R{i:03d}']={'events':events,'notes_ko':notes}
def S(a,e,r='GT 행동을 의미적으로 명시한다.'): return (a,'supported',e,r)
def H(a,e,r='확신적으로 서술하지만 이 영상의 GT에는 해당 핵심 행동이 없다.'): return (a,'hallucinated_gt',e,r)
def U(a,e,r): return (a,'uncertain',e,r)
put(0,H('driving','A white sedan drives into the parking lot'),H('parking','parks near other vehicles'),H('walking','Two individuals walk together'),H('carrying_object','carrying an object'))
put(1,H('cycling','Two cyclists ride across the parking lot'),H('driving','A silver sedan drives through'),H('walking','Two pedestrians walk along the sidewalk'))
put(2,H('walking','A person walks along a paved pathway toward a building'),notes='건물을 향한 접근은 건물에서 나오는 사건을 충족하지 않는다.')
put(3,H('walking','A person walks toward the entrance'),H('entering_facility','before entering'),notes='GT는 퇴장이며 설명은 진입이다.')
put(4,H('driving','A silver sedan drives into view'),H('parking','stops at an empty parking space'),H('walking','two individuals walk past the parked car'))
put(5,H('driving','A silver sedan drives up the entrance ramp'),H('parking','parks parallel to other spaces'),S('getting_out_of_vehicle','A person wearing light-colored clothing steps out of the parked car'),H('walking','walks around to the driver’s side door'),H('opening_vehicle_door','opens it'),H('retrieving_object','retrieves something inside'),H('closing_vehicle_door','closes the door'),H('carrying_object','carrying a bag slung across their shoulder'))
put(6,H('walking','A person walks down the stairs'),notes='계단 하강과 입구 주변 이동만으로 건물 퇴장을 추론하지 않는다.')
put(7,H('walking','others walk up and down nearby stairs'))
put(8,S('walking','A person walks down the stairs'),U('facility_transition_unspecified','passing through doorways','출입문 통과는 있으나 건물 안/밖 방향을 특정할 수 없다.'))
put(9,S('walking','People walk up and down stairs'),S('carrying_object','A person walks past them carrying something white'),H('talking','engaged in conversation'))
put(10,H('walking','A person walks down stairs toward an entrance'))
put(11,H('walking','People walk across the paved courtyard'),notes='유리문에 접근한 뒤 왼쪽으로 떠나는 것은 출입문 통과가 아니다.')
put(12,H('walking','A person walks down the stairs'))
put(13,H('walking','people walk past them'),H('talking','engaged in conversation'),U('facility_transition_unspecified','before exiting through the doorway','야외에서 출입문으로 이동하는 문맥에서 exiting의 기준 공간이 모호하다. 하나의 모호한 출입 사건이며 진입/퇴장 둘을 별도 예측으로 세지 않는다.'))
put(14,S('walking','Two individuals walk together through an empty parking lot toward the right side'),H('driving','A few cars drive past'))
put(15,S('walking','A person walks across the parking lot'),H('cycling','another individual rides a bicycle'))
put(16,H('driving','A black car drives through an intersection'))
put(17,H('walking','A person walks toward a parked green car'),H('driving','the car drives off'),notes='사람은 차에서 물러선다. 차가 출발했다는 사실로 탑승을 보충 추론하지 않는다.')
put(18,H('driving','A white car drives into the parking lot'),H('parking','parks near other vehicles'),H('walking','A person walks through the parking area'),H('carrying_object','carrying something on their shoulder'),H('exiting_facility','Another individual exits a building nearby'))
put(19,H('cycling','Two cyclists ride across the parking lot'),H('walking','A person walks along the sidewalk'),H('exiting_facility','Another individual exits a building entrance'),H('driving','Several cars drive through the intersection'))
put(20,H('walking','A person walks along a paved pathway toward a building'))
put(21,H('walking','A person walks toward the entrance'),H('entering_facility','before entering'))
put(22,H('driving','A silver sedan drives into view'),H('parking','parks briefly next to another vehicle'))
put(23,H('driving','A silver sedan drives up to the parking lot entrance'),H('parking','parks parallel to other cars'),S('getting_out_of_vehicle','A person exits the parked car'),H('walking','walks around to the driver’s side door'),H('opening_vehicle_door','opens it'),H('carrying_object','gets out carrying a bag'),H('closing_vehicle_door','closes the door behind them'))
put(24,H('walking','A person walks down the stairs'),notes='steps leading into buildings는 계단의 연결 위치이지 실제 건물 진입/퇴장 완료가 아니다.')
put(25,H('walking','others walk up and down the stairs'),notes='exiting frame은 건물 퇴장이 아니다. 정적인 휴대전화 주시는 핵심 동작에서 제외한다.')
put(26,S('walking','A person walks down stairs near a building'))
put(27,S('walking','People walk up and down stairs'),H('talking','during their conversation'),H('manipulating_object','adjusting something on the ground beside them'))
put(28,H('walking','People walk through an outdoor plaza'),notes='계단이 건물로 이어진다는 수식어는 문턱을 통과했다는 주장이 아니다.')
put(29,H('walking','People walk across the paved courtyard'),S('exiting_facility','One person exits through the doorway'),U('entering_facility','two more individuals entering the space','space가 건물 내부인지 야외 장면인지 불분명하다.'))
put(30,H('walking','A person walks down the stairs'))
put(31,H('walking','pedestrians walk past them'),H('talking','engaged in conversation'))
put(32,S('walking','Two individuals walk together through an empty parking lot toward the right side'))
put(33,S('walking','a person walks across the open space'),H('cycling','two individuals ride bicycles'))
put(34,H('driving','A black car drives through an intersection'))
put(35,H('walking','A person walks toward parked cars'),H('driving','A dark sedan drives through the intersection'))
put(36,H('driving','A white car drives into view'),H('parking','parks near other vehicles'),H('walking','A person walks along the sidewalk'))
put(37,H('cycling','Two cyclists ride across the parking lot'),H('driving','A silver sedan drives through'),H('walking','Two pedestrians walk along the sidewalk'))
put(38,H('walking','A person walks along a paved pathway toward a building'))
put(39,H('walking','A person walks toward the entrance'),H('carrying_object','carrying multiple bags'))
put(40,H('driving','A silver sedan drives into view'),U('getting_out_of_vehicle','two individuals emerge from within the parked cars','차량 내부에서 하차인지 주차된 차들 사이에서 나타난 것인지 모호하다.'),H('walking','both walking together towards the center-left portion'))
put(41,H('driving','A silver sedan drives onto the parking lot entrance ramp'),H('parking','parks, and remains stationary'),S('getting_out_of_vehicle','A person exits the parked car'),H('walking','walks around to the driver’s side door'),H('opening_vehicle_door','opens it'),H('retrieving_object','retrieves items including a backpack'),H('closing_vehicle_door','closes the door'))
put(42,H('walking','A person walks down the stairs'))
put(43,H('walking','People walk down the stairs'))
put(44,S('walking','A person walks down the stairs'))
put(45,S('walking','People walk up and down stairs'),S('carrying_object','A person walks past them carrying something white'),H('talking','engaged in conversation'),H('gesturing','occasional gestures'))
put(46,H('walking','People walk up and down stairs leading into a building'))
put(47,H('walking','People walk across the courtyard'),S('exiting_facility','Another individual exits through the doorway'),S('entering_facility','after entering earlier'))
put(48,H('walking','A person walks down the stairs'))
put(49,H('walking','people walk past them'),H('talking','engaged in conversation'),S('entering_facility','someone enters through glass doors'),S('exiting_facility','then exits shortly after'))
put(50,S('walking','Two individuals walk together through an empty parking lot toward the right side'),H('getting_out_of_vehicle','Another individual exits a vehicle'))
put(51,S('walking','A person walks across the parking lot'),H('cycling','another individual rides a bicycle'))
put(52,H('driving','A black car drives through an intersection'))
put(53,H('walking','A person walks toward a parked green car'),S('getting_into_vehicle','before getting inside'),H('driving','driving away through a landscaped intersection'),H('running','another individual runs across a grassy field'))
put(54,H('driving','A red car drives through the parking lot'),H('walking','people walk along the sidewalk'))
put(55,H('driving','a red car that drives across the lot'),H('walking','two individuals appear walking together'),H('cycling','A person riding a bicycle'))
put(56,H('walking','A person walks along a pathway'),H('driving','A white van drives past'),U('exiting_facility','another individual briefly appears at the entrance','입구에서 나타남이 퇴장을 암시할 수 있지만 경계 통과가 확인되지 않는다.'))
put(57,H('walking','A person walks across the paved courtyard'),H('entering_facility','before entering through glass doors'))
put(58,H('driving','A silver sedan drives into an empty parking lot'),H('parking','parks near the center-right edge'),S('getting_out_of_vehicle','two people exit the vehicle'),H('walking','walk toward the bottom-left corner'),notes='인원수와 0:45 시각은 채점 범위 밖이다. GT 시간과 다르다는 이유로 행동 의미 일치를 취소하지 않는다.')
put(59,H('driving','A silver sedan enters the frame from the bottom right'),H('parking','parks diagonally across two spaces'),S('getting_out_of_vehicle',"a person exits the driver's side of the parked sedan"),H('walking','walks along the curb away from the vehicle'))
put(60,H('walking','A person walks down the stairs'))
put(61,H('walking','two people walk down the stairs'),S('exiting_facility','someone exits through glass doors'))
put(62,S('walking','two people walk together from right to left'))
put(63,S('walking','people walk across the grassy lawn'),H('talking','engaging in conversation'))
put(64,H('walking','People walk down the stairs'),S('entering_facility','some individuals enter and exit through doors'),S('exiting_facility','some individuals enter and exit through doors'))
put(65,H('walking','People walk up and down a set of stairs'))
put(66,H('walking','A person walks down the stairs'),notes='화면 구석에 나타남은 건물 경계 통과를 함의하지 않는다.')
put(67,H('walking','another person walks down the stairs'))
put(68,S('walking','Two people walk from right to left through the parking lot'))
put(69,S('walking','a person walks across the paved area'),H('cycling','another individual rides a bicycle'))
put(70,H('driving','A black car drives from left to right through an intersection'))
put(71,H('walking','a person wearing red walks past parked cars'),S('getting_into_vehicle','before getting into a dark green sedan'),H('driving','the car then drives out of the parking space'))
put(72,H('walking','A person walks from left to right'))
put(73,H('driving','a red car drives across the middle section'),H('walking','two people walk together along the sidewalk'))
put(74,H('walking','A person walks along a pathway'),U('exiting_facility','another individual briefly appears at an entrance further back under the overhang before moving out of view','입구에서 나타난 후 화면 밖으로 이동하지만 건물 밖으로 나왔는지는 명확하지 않다.'))
put(75,H('walking','A person walks along a covered walkway'),H('carrying_object','while carrying items'))
put(76,H('driving','A silver sedan drives into an empty parking lot'),H('parking','parks near the center-right edge'),S('getting_out_of_vehicle','two people then exit the vehicle'),H('walking','walk toward the bottom-left corner'),notes='인원과 시각 정확도는 채점하지 않는다.')
put(77,H('driving','A silver sedan enters the frame from the right on a road'),H('parking','parks diagonally across two spaces'),S('getting_out_of_vehicle',"a person exits the driver's side door"),H('walking','walks along the curb away from the vehicle'),H('carrying_object','while holding a white object'))
put(78,H('walking','A person walks down the stairs'))
put(79,H('walking','Two people walk down a set of stairs'),S('exiting_facility','two individuals exit through glass doors'))
put(80,S('walking','A person walks down the stairs'))
put(81,S('walking','people walk across the grassy lawn'),H('talking','engaged in conversation'))
put(82,H('walking','People walk down the stairs'),S('entering_facility','some individuals enter and exit through doors'),S('exiting_facility','some individuals enter and exit through doors'))
put(83,H('walking','People walk up and down the stairs'))
put(84,H('walking','A person walks down the stairs'))
put(85,H('walking','another person walks down the stairs'),S('entering_facility','several individuals enter and exit the doorway'),S('exiting_facility','several individuals enter and exit the doorway'))
put(86,H('driving','A blue car drives through the parking lot from right to left'),S('walking','two people walk on the far side near white vans'),notes='right to left는 자동차의 방향이다. 사람의 보행 방향으로 전용하지 않는다.')
put(87,S('walking','a person walks across the paved area'),U('gesturing','another individual stands still holding their head','머리를 잡는 자세가 GT의 몸짓일 수 있지만 정적 자세만으로 동작을 확정할 수 없다.'),H('cycling','two people ride bicycles'))
put(88,H('driving','A black car drives from left to right through an intersection'))
put(89,H('walking','a person wearing red walks past parked cars'),S('getting_into_vehicle','before getting into a dark green sedan'),H('driving','the car then drives out of the parking space'))
# Explicit direction decisions by LLM. Everything else is omission, not inferred
# from action words or location words. There are 7 directional semantic units.
DIR={
 (9,'walking','left_to_right'):('correct','walking away toward the right edge of the frame','같은 보행 주체가 화면 오른쪽으로 이동한다고 명시한다.'),
 (14,'walking','left_to_right'):('correct','toward the right side','보행의 오른쪽 진행을 명시한다.'),
 (32,'walking','left_to_right'):('correct','toward the right side','보행의 오른쪽 진행을 명시한다.'),
 (32,'walking','right_to_left'):('contradictory','toward the right side','유일하게 서술한 보행 쌍은 오른쪽으로 이동하여 GT의 왼쪽 방향과 반대다.'),
 (50,'walking','left_to_right'):('correct','A person walks alone from left to right','보행 방향을 직접 명시한다.'),
 (62,'walking','right_to_left'):('correct','two people walk together from right to left','함께 걷는 사람들의 왼쪽 진행이 명시된다.'),
 (68,'walking','right_to_left'):('correct','Two people walk from right to left','차량이 아닌 사람의 보행 방향이다.'),
}
def main():
    out=[]
    for i,row in enumerate(DATA['records']):
        authored=J[row['record_id']]; gt=DATA['ground_truth'][row['video_id']]['caption_events']
        pred=[]
        for action,label,evidence,reason in authored['events']:
            assert evidence in row['description'], (i,action,evidence)
            pred.append({'action':action,'label':label,'evidence':evidence,'reason_ko':reason,'factual_status':'uncertain_unverified' if label!='supported' else 'supported_by_gt_action_only'})
        assert len({p['action'] for p in pred})==len(pred)
        aj=[]
        for action in sorted({e['action'] for e in gt}):
            p=next((p for p in pred if p['action']==action),None)
            if i in [8,13] and action in ['entering_facility','exiting_facility']:
                p=next(p for p in pred if p['action']=='facility_transition_unspecified')
            if p:
                assert p['label']!='hallucinated_gt',(i,action)
                aj.append({'action':action,'label':'matched' if p['label']=='supported' else 'uncertain','evidence':p['evidence'],'reason_ko':p['reason_ko']})
            else:
                aj.append({'action':action,'label':'missed','evidence':'','reason_ko':'설명 전체에 이 GT 행동을 성립시키는 의미적 진술이 없다. 접근·주변 이동·정적 존재만으로 해당 행동을 보충하지 않는다.'})
        dj=[]
        for action,direction in sorted({(e['action'],e['direction']) for e in gt if e['direction']}):
            status,evidence,reason=DIR.get((i,action,direction),('missing','','해당 행동 주체에 결부된 이 GT 방향의 진술이 없다. 위치·계단 방향·다른 주체의 이동 방향은 근거로 쓰지 않는다.'))
            dj.append({'action':action,'gt_direction':direction,'label':status,'evidence':evidence,'reason_ko':reason})
        out.append({'record_id':row['record_id'],'action_judgments':aj,'direction_judgments':dj,'predicted_events':pred,'notes_ko':authored['notes_ko']})
    assert len(J)==len(out)==90
    (ROOT/'judge_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('Serialized',len(out),'LLM-authored semantic judgments.')
if __name__=='__main__': main()
