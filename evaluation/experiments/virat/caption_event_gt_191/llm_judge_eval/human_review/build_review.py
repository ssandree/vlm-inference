#!/usr/bin/env python3
"""Build an offline human review worksheet from LLM-curated semantic claims.
No keyword/regex scoring, no video inference, and no modification of GT.
"""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent
D=json.loads((SOURCE/'per_video_details.json').read_text())
RECORDS={r['record_id']:r for r in D}
GT=json.loads((SOURCE/'inputs.json').read_text())['ground_truth']
VIDEOS=list(GT)
# Fields: video index | priority | record numbers | statement to verify | review hint
# These propositions were manually curated by the conversation LLM from full captions.
SPEC='''
1|confirmed|4,22,40,58,76|은색 세단이 화면에 들어와 이동한다.|사용자가 사실로 확인함. 진입 위치·시각은 별도 기록.
1|confirmed|4,22,41,58,59,76,77|은색 세단이 빈 주차 공간에 정차/주차한다.|사용자가 사실로 확인함. 짧은 교통 정차인지 주차인지와 구간 기록.
1|core|5,23|은색 세단이 주차장 진입부에서 우회전한다.|좌회전 주장과 비교. 차량 기준 회전과 화면 이동 방향은 구분.
1|core|41,59,77|은색 세단이 주차장 진입부에서 좌회전한다.|다른 시점의 회전이면 우회전과 모두 사실일 수 있음.
1|detail|59,77|은색 세단이 두 주차 면을 가로질러 비스듬히 주차한다.|주차 자체와 주차 자세 오류를 분리.
1|core|5,22,23,40|은색 세단이 정차/주차한 뒤 다시 출발한다.|5번 설명은 다시 출발함과 계속 주차됨을 모두 주장. 주체·시점 확인.
1|core|5,41,77|은색 세단은 사람이 주변에서 행동하는 동안 계속 정지해 있다.|앞 항목과 같은 구간인지 확인. 차가 실제 움직인 구간을 기록.
1|core|5,23,41,58,59,76,77|사람이 은색 세단 안에서 밖으로 내린다.|기존 getting_out_of_vehicle GT와 대응. 새 사건을 중복 추가하지 말 것.
1|detail|40,58,76|은색 세단에서 내리는 사람이 두 명이다.|차 사이에서 나타난 보행자와 차량 내부에서 하차한 사람을 구분.
1|confirmed|5,23,41|하차한 사람이 운전석 문 쪽으로 걸어간다.|사용자가 사실로 확인함. 하차 위치와 이후 이동은 별도 구간.
1|confirmed|5,23,41|그 사람이 차량 문을 연다.|사용자가 사실로 확인함. 운전석/조수석/트렁크를 적을 것.
1|core|5,41|그 사람이 차량 안에서 물건을 꺼낸다.|문을 여는 것만으로 물건 꺼내기를 인정하지 말 것.
1|detail|41|차량에서 꺼낸 물건은 배낭이다.|물건 종류만 불분명하면 행동 자체를 부정하지 말 것.
1|confirmed|5,23,41|그 사람이 차량 문을 닫는다.|사용자가 사실로 확인함. 같은 문인지 기록.
1|confirmed|5|그 사람이 어깨에 가방을 메고 이동한다.|사용자가 사실로 확인함. 들고 있음/메고 있음과 이동 구간 기록.
1|core|23|그 사람이 보도를 따라 화면 왼쪽으로 걸어간다.|오른쪽/아래쪽 이동 주장과 같은 사람·구간인지 비교.
1|core|41|그 사람이 보도를 따라 화면 오른쪽으로 걸어간다.|문 조작 후의 구간을 확인.
1|core|58,59,76,77|그 사람이 차량에서 멀어져 화면 아래쪽으로 걸어간다.|좌하단 대각선인지 아래쪽인지 실제 8방향을 기록.
1|core|77|그 사람이 멀어졌다가 주차된 차 쪽으로 되돌아온다.|방향 전환/왕복이 실제면 구간을 나누어 기록.
1|core|77|그 사람이 흰색 물건을 들고 있다.|가방과 같은 물건인지 확인. 색상 불일치는 수정란에 기록.
1|detail|23|그 사람이 열쇠 또는 휴대전화처럼 보이는 작은 물건을 손에 들고 있다.|원문도 appears like로 불확실하게 표현함. 물건 종류를 강제 확정하지 말 것.
1|core|77|별도의 사람이 열린 조수석 문 옆에 나타난다.|기존 하차자와 같은 사람인지 확인.
1|core|77|조수석 문 옆의 사람이 보도를 따라 걸어간다.|앞 항목의 주체와 연결.
1|core|4,40,58,76|두 사람이 함께 주차된 차 근처에서 걸어간다.|두 명 하차했다는 주장과 분리. 함께 걷기만 실제일 수 있음.
1|core|22,40,59|흰색 픽업트럭이 인접 도로를 이동한다.|색/차종이 다르면 실제 주체 설명을 남길 것.
1|core|22|흰색 픽업트럭이 다른 주차 차량 옆에 잠시 주차한다.|정차 없이 통과한다는 설명과 대조.
1|core|59,77|어두운색 차량이 은색 세단 뒤쪽 도로/교차로를 통과한다.|은색 세단과 다른 차량인지 확인.
1|core|58,76|흰색 차량이 배경 도로를 지나간다.|픽업트럭과 같은 차량이면 사건 ID를 공유.
1|detail|76|같은 흰색 차량이 도로를 여러 차례 지나간다.|서로 다른 차량을 반복 이동으로 오인했는지 확인.
1|core|76|어두운 SUV가 주차장에 들어왔다가 정차하지 않고 나간다.|진입과 퇴장 구간을 기록. 다른 차량 설명과 중복 주의.
2|core|6,7,24,25,42,43,60,61,78,79|사람이 화면 왼쪽 계단을 내려온다.|기존 GT의 건물 퇴장과 별도 보행 사건일 수 있음.
2|core|6,7,24,25,42|사람이 계단을 올라간다.|계단 오르기와 건물 내부 진입은 분리.
2|core|6,24,42,60,78|사람이 건물 입구 주변 통로를 걸어간다.|계단 사건과 다른 사람/구간이면 새 사건으로 기록.
2|core|7,25,43|사람이 잔디밭 또는 그 주변 길을 가로질러 이동한다.|잔디 위인지 포장 경로인지 수정 가능.
2|detail|7|그 보행자는 파라솔 테이블 쪽으로 향한다.|같은 보행자의 목적지가 입구/화단인지 확인.
2|detail|25|그 보행자는 벽돌 화단 쪽으로 향한다.|직접 보이는 이동 경로만 기록.
2|detail|43|그 보행자는 오른쪽 건물 입구 쪽으로 향한다.|접근과 실제 건물 진입을 구분.
2|core|61,79|사람이 오른쪽 유리문을 통과해 건물 밖으로 나온다.|기존 퇴장 GT 6.34–8.26초와 같은 사건인지 확인. later는 모델 주장일 뿐.
2|detail|79|유리문을 통해 밖으로 나오는 사람이 두 명이다.|단순 인원 오류인지 별도 퇴장 사건 누락인지 확인.
2|context|6,7,24,25,42,43,60,61,78,79|전경 파라솔 아래에 앉아 있는 사람이 있다.|정적 상태를 GT에 넣을지는 별도 정책. 행동으로 자동 편입하지 않음.
2|context|25,43,61,79|그 사람은 휴대전화를 보고 있다.|작아서 식별 안 되면 보류. 고개 숙임만으로 휴대전화 단정 금지.
3|core|8,9,26,27,44,45,62,80|사람이 왼쪽 계단을 내려간다.|기존 walking GT 중 어떤 사건인지 연결.
3|core|8,9,27,44,45,62|사람이 왼쪽 계단을 올라간다.|내려가는 사람과 별도 주체인지 확인.
3|core|9,27,44,45,63,81|사람이 잔디밭 또는 주변 보행로를 가로질러 걷는다.|기존 여러 walking 사건과 중복 매칭 방지.
3|core|9,45|사람이 흰색 물건을 들고 앉아 있는 사람들 앞을 지나간다.|기존 carrying_object 3건 중 누구인지 확인. 실제 색/물건은 수정란.
3|core|9|사람이 앉아 있는 두 사람 쪽으로 다가간다.|그 뒤 돌아서는 사람과 같은 주체인지 확인.
3|core|9|그 사람이 앉아 있는 사람들에게서 멀어져 화면 오른쪽으로 걷는다.|접근과 이탈의 시간 구간을 나눌 것.
3|core|62|두 사람이 함께 화면 오른쪽에서 왼쪽으로 걷는다.|기존 그룹 CE017/CE018 중 어떤 사건인지 확인. 한 구절을 두 그룹에 자동 대응 금지.
3|core|8|배경 출입문을 통과하는 사람이 있다.|건물 안→밖인지 밖→안인지 실제 극성 기록. 기존 exiting과 연결.
3|core|9,27,45,63,81|테이블에 앉은 사람들이 대화하는 것으로 확인된다.|무음 영상에서 대화 자체가 불확실하면 보류하고 눈에 보이는 몸짓만 기록.
3|core|9,27|앉아 있는 사람이 상체를 앞으로 기울인다.|대화라는 추론과 분리한 관찰 가능한 동작.
3|core|27|앉아 있는 사람이 옆 바닥의 물건을 만지거나 조정한다.|상체 움직임만 있고 물건 조작은 없을 수 있음.
3|core|45|앉아 있는 사람이 손/팔로 몸짓한다.|가방 운반이나 자세 변화와 구분.
3|context|9,27,45,63,80,81|파라솔 아래 테이블에 두 사람이 함께 앉아 있다.|대화/몸짓의 주체 확인용. 인원수는 메모 가능.
4|core|12,30,48,49,66,67,84,85|사람이 왼쪽 계단을 내려간다.|GT에는 건물 진입/퇴장만 있어 보행 누락 후보.
4|core|12,30,48|사람이 위쪽 통로 또는 테이블 뒤에서 오른쪽→왼쪽으로 걷는다.|설명별 위치가 다름. 실제 인물·구간 기록.
4|core|12,48|사람이 화면 좌하단에서 들어와 중앙 좌석 쪽으로 걷는다.|화면 진입을 건물 진입으로 처리하지 말 것.
4|core|13,31,49,67,85|사람이 파라솔/테이블 주변 파티오를 걸어간다.|여러 보행자가 있으면 개별 사건을 적을 것.
4|core|31,49|파티오를 걷는 사람이 화면 오른쪽으로 이동한다.|건물 퇴장 여부와는 별개.
4|core|13,31|사람이 잔디 가장자리/화분 옆 길을 걷는다.|같은 사건의 설명 변형인지 확인.
4|core|31|잔디 가장자리의 사람이 화면 왼쪽으로 걷는다.|이동 방향은 같은 주체에만 연결.
4|core|13|파티오를 걷던 사람이 출입문 경계를 통과한다.|원문 exiting through the doorway의 실제 극성 확인. 화면 밖 이탈과 구분.
4|core|49,85|사람이 유리문/출입문을 통해 건물 안으로 들어간다.|기존 entering GT 2.46–4.63초와 연결하거나 별도 사건 기록.
4|core|49,85|사람이 유리문/출입문을 통해 건물 밖으로 나온다.|기존 exiting GT 5.26–7.18초와 연결하거나 별도 사건 기록.
4|detail|49|건물에 들어간 사람과 곧이어 나온 사람이 동일인이다.|진입·퇴장 각각 사실이어도 동일인 연결은 틀릴 수 있음.
4|core|13,31,49|테이블에 앉아 있는 사람들이 대화한다.|눈에 보이는 동작만 확인 가능하면 보류하고 수정.
4|context|12,13,30,31,48,49,66,67,84,85|파라솔 아래에 앉아 있는 사람이 있다.|정적 상태; 보행/출입 동작과 분리.
5|core|10,28,29,46,64,65,82,83|사람이 계단을 내려간다.|GT 진입/퇴장 외 보행 누락 후보.
5|core|10,11,28,29,46,47,65,83|사람이 계단을 올라간다.|건물로 이어지는 계단이라는 이유만으로 진입을 인정하지 말 것.
5|core|10|사람이 테이블 근처를 화면 왼쪽→오른쪽으로 걷는다.|계단 보행자와 별도 주체인지 확인.
5|core|11,29,47,65,83|사람이 잔디 옆 보행로/마당을 걷는다.|각 보행자의 구간을 분리.
5|core|11,29,47|사람이 건물의 유리문 쪽으로 접근한다.|문을 통과하는 것과 별개 질문.
5|core|11|유리문에 접근한 사람이 돌아서 화면 왼쪽으로 걸어간다.|진입 없이 되돌아오는 사건인지 확인.
5|core|29,47,64,82|사람이 출입문을 통과해 건물 밖으로 나온다.|기존 exiting 14.35–16.27초와 비교.
5|core|47,64,82|사람이 출입문을 통과해 건물 안으로 들어간다.|기존 entering 3.63–5.55초와 비교.
5|detail|29|한 사람이 문으로 나온 뒤 다른 사람들이 건물 안으로 들어간다.|entering the space가 실제 건물 진입인지 확인. 인원/순서는 메모.
5|detail|47|건물에서 나온 사람이 앞서 건물에 들어갔던 동일인이다.|두 GT 주체가 다르므로 특히 확인. 동일인으로 자동 연결 금지.
5|core|29|계단을 올라간 사람이 꼭대기에서 되돌아 내려온다.|서로 다른 보행자를 왕복으로 묶은 것은 아닌지 확인.
5|core|47|사람이 계단 옆 쓰레기통 쪽으로 접근한다.|이후 멀어짐은 별도 시간으로 기록.
5|core|47|쓰레기통에 접근한 사람이 다시 멀어진다.|실제 물체 조작이 있는지도 자유 메모.
5|core|47|사람이 화분 뒤쪽을 지나간다.|녹색 옷이라는 속성은 틀리면 수정.
6|core|14,32,50|함께 걷는 두 사람이 화면 오른쪽으로 이동한다.|다른 설명의 오른쪽→왼쪽과 비교. 같은 그룹·같은 구간인지 기록.
6|core|68|함께 걷는 두 사람이 화면 오른쪽에서 왼쪽으로 이동한다.|기존 group walking GT와 연결.
6|core|14,32,50|함께 걷던 두 사람이 서로 갈라진다.|한 명 정지/다른 한 명 계속 이동의 실제 여부 확인.
6|core|14,32,50|두 사람 중 한 사람이 주차 차량 근처에 멈춘다.|그 사람이 gesturing GT 주체인지 확인.
6|core|50|별도의 사람이 먼 쪽 통로에서 왼쪽→오른쪽으로 걷는다.|기존 방향별 walking 사건 중 어느 것인지 확인.
6|core|15,33,51,69,87|사람이 주차된 밴 근처의 열린 주차 공간을 가로질러 걷는다.|기존 보행과 동일 사건이면 연결만 하고 중복 추가 금지.
6|core|33|그 사람이 걷다가 멈춰 주변을 둘러본다.|걷기와 고개/시선 동작을 분리 기록.
6|core|68|다른 사람이 화면 오른쪽에 나타나 서 있다가 다시 걸어간다.|앞의 보행자들과 같은 주체인지 확인.
6|core|69|두 사람이 주차 차량 옆 보도를 함께 걷는다.|주차장 내부 그룹 보행과 같은 사건일 수 있음.
6|core|15,33,51,69,87|운동장 펜스 옆 길에서 실제 자전거 주행이 있다.|보행자를 자전거로 오인했는지 우선 확인.
6|core|15,33,51,87|먼 쪽 도로/모퉁이에서 추가 자전거 주행이 있다.|같은 자전거를 여러 장면에서 중복 서술했을 수 있음.
6|core|33|전경 쪽에서도 자전거가 주행한다.|배경 자전거와 별도 주체인지 확인.
6|core|50|화면 오른쪽 차량에서 사람이 내린다.|현재 GT에 없는 getting_out_of_vehicle 후보.
6|core|14,86|주차장 또는 인접 도로에서 차량이 움직인다.|차량이 정지해 있다는 설명과 대조.
6|core|86|파란 차량이 화면 오른쪽에서 왼쪽으로 이동한다.|색이 다르면 수정. 이 방향을 보행자에게 전용하지 말 것.
6|core|87|서 있는 사람이 손으로 머리를 잡거나 만진다.|기존 gesturing의 구체 동작인지 확인. 정적 자세와 동작도 구분.
7|confirmed|0,18,36|흰색 세단이 주차장으로 들어온다.|사용자가 사실로 확인함. 진입 방향은 별도 확인.
7|confirmed|0,18,36|흰색 세단이 다른 차량 근처에 주차한다.|사용자가 사실로 확인함. 사건 시간 기록 필요.
7|core|0,36|그 흰색 세단이 주차 후 다시 움직여 화면 오른쪽으로 나간다.|진입·주차 사실만으로 이 후속 출발까지 참으로 처리하지 말 것.
7|core|54,55,73|빨간 차량이 주차장 안을 이동한다.|흰색 세단의 색상 오류인지 별도 차량인지 확인.
7|core|54,73|그 빨간 차량은 화면 왼쪽→오른쪽으로 이동한다.|약 0:37은 모델 주장이지 검증된 탐색 시각이 아님.
7|core|1,19,36,37|은색 세단이 화면 좌하단 쪽 주차장을 이동한다.|흰색 세단과 동일 차량인지도 확인.
7|detail|37|그 은색 세단이 우회전한다.|차량 회전과 화면 오른쪽 이동은 구분.
7|core|1,37|다른 차량이 화면 우상단 쪽에서 들어와 주차 차량들 근처에 멈춘다.|실제 경로·주체를 기록. 정차와 주차 구분.
7|core|18|검은 SUV가 화면 우하단 근처로 와서 멈춘다.|서 있는 차량을 움직인 것으로 오인했는지 확인.
7|core|19,73|앞서 언급한 차들 외에도 차량이 주차장/교차로를 통과한다.|확인된 차량별로 분리. 일반적 교통 서술을 여러 사건에 반복 대응하지 말 것.
7|core|1,19,37|주차장 중앙에서 실제 자전거 주행이 있다.|달리는 사람을 자전거로 잘못 묘사했는지 우선 확인.
7|core|1,19,37|중앙 자전거는 화면 오른쪽→왼쪽으로 이동한다.|반대 방향 주장과 같은 주체/구간인지 확인.
7|core|55|중앙 자전거는 화면 왼쪽→오른쪽으로 이동한다.|양방향이 모두 실제일 수 있으므로 시간 구간을 적을 것.
7|core|0,18,36,54,55,73|사람이 먼 쪽 건물/운동장 옆 보도를 걷는다.|여러 주체면 개별 사건으로 기록.
7|core|1,19,37|사람이 화면 오른쪽 건물/흰 밴 근처 보도를 걷는다.|먼 쪽 보행자와 동일인인지 확인.
7|core|0,36,55|두 사람이 함께 주차장 중앙을 걷는다.|상우/하우로 향한다는 설명은 실제 방향을 별도 기록.
7|core|18,19,73|두 사람이 함께 외곽 보도를 걷는다.|중앙 그룹과 동일인·동일 사건인지 확인.
7|core|0|사람이 차량 쪽으로 다가간다.|GT 트렁크 조작/탑승 주체와 연결할 수 있음.
7|core|0|그 사람이 차량에서 멀어져 중앙 쪽으로 걷는다.|접근과 이탈을 별도 구간으로 확인.
7|core|0,18|사람이 물건 또는 어깨에 멘 가방을 운반하며 걷는다.|운반 자체를 확인하고 물건 종류·주체를 적을 것.
7|core|18,19|사람이 건물 밖으로 나온다.|문 근처에서 나타남이 아니라 실제 문 경계 통과인지 확인.
7|core|18|건물에서 나온 그 사람이 물건을 들고 이동한다.|별도의 운반자와 혼동 금지.
7|core|73|사람이 건물 출입문 쪽으로 걸어간다.|접근 자체만 확인. 건물 진입으로 확대하지 말 것.
7|core|72|사람이 중앙 왼쪽 부근에서 왼쪽→오른쪽으로 걷는다.|GT running 주체를 보행으로 오인했을 가능성 확인.
8|core|2,3,20,21,38,39,56,57,74,75|사람이 건물 앞 통로/차양 아래를 걷는다.|GT exiting 이후의 보행인지 이전 접근인지 구간 확인.
8|core|2,3,20,21,38,39,57|그 사람은 건물 입구 쪽으로 접근한다.|문을 향해 가는지 이미 나온 뒤 멀어지는지 구분.
8|core|3,21,57|사람이 출입문을 통과해 건물 안으로 들어간다.|현재 GT는 exiting. 반대 극성 오인인지 실제 별도 진입도 있는지 확인.
8|core|56,74|차양 아래 입구에서 나타나는 사람은 건물 안에서 밖으로 나온 것이다.|모델의 appears at an entrance 해석을 확인하는 질문. 기존 GT 6.38–8.97초와 연결.
8|core|39,75|걷는 사람이 물건/가방을 운반한다.|여러 개의 가방인지 여부는 메모. 운반 자체와 개수 분리.
8|core|75|그 사람이 카메라에서 멀어져 화면 배경 쪽으로 걷는다.|원근 방향과 좌우 화면 방향은 별도 기록.
8|core|56|건물 뒤 배경 도로에서 흰색 밴이 움직인다.|실제 관측 가능한 차량인지 확인.
8|detail|56|그 흰색 밴은 화면 오른쪽→왼쪽으로 이동한다.|차량 움직임이 실제일 때만 방향 판정.
8|context|2,20,38|통로 옆에 서 있는 다른 사람들이 있다.|주인공과 별도 사람인지 확인. 정적 상태는 선택 검토.
9|core|16,34,35,52,70,88|검은색/어두운 차량이 교차로를 통과한다.|아래 탑승 후 출발한 차량과 같은 것인지 확인.
9|detail|70,88|교차로를 지나는 차량은 화면 왼쪽→오른쪽으로 이동한다.|차량 식별 후 방향 확인.
9|core|17,35,53,71,89|사람이 주차 차량 쪽으로 걸어간다.|현재 GT 탑승 전의 보행 누락 후보.
9|core|53,71,89|그 사람이 주차된 차량에 탄다.|기존 getting_into_vehicle 13.88–17.28초와 연결.
9|core|17|차량에 접근한 사람이 차량에서 물러나는 움직임을 보인다.|탑승 전 잠깐 물러나는 것인지 끝내 타지 않는 것인지 수정란에 구분.
9|core|17,53,71,89|그 차량이 주차 위치에서 출발한다.|차량 이동 자체와 사람 탑승의 인과 연결은 구분.
9|detail|71,89|출발한 차량이 도로로 우회전한다.|직진/좌회전과 혼동 여부 확인.
9|core|35|사람이 주차 차량/중앙 녹지 옆을 걸어 화면 밖으로 나간다.|탑승하는 사람을 잘못 추적한 설명일 수도 있음. 실제 주체 기록.
9|core|53|펜스 옆 잔디밭을 가로질러 달리는 사람이 있다.|그림자나 영상 전환을 사람의 달리기로 오인했는지 우선 확인.
9|context|71,89|배경 나무/건물 위로 새가 날아간다.|기존 핵심 사람·차량 평가 밖의 선택 항목. 포함 여부를 별도 결정.
'''
rows=[]; counts={}
for line in SPEC.strip().splitlines():
    vi,priority,refs,question,hint=line.split('|')
    vi=int(vi); video=VIDEOS[vi-1]; counts[vi]=counts.get(vi,0)+1
    ids=['R'+n.strip().zfill(3) for n in refs.split(',')]
    assert all(RECORDS[r]['video_id']==video for r in ids),(video,ids)
    confirmed=priority=='confirmed'
    rows.append({'review_id':f'V{vi:02d}-{counts[vi]:02d}','video_id':video,'priority':priority,'statement_ko':question,'review_hint_ko':hint,'source_record_ids':ids,'answer':'yes' if confirmed else '', 'answer_source':'user_prior_confirmation' if confirmed else '', 'start_sec':'','end_sec':'','actor_id':'','actual_action':'','actual_screen_direction':'','actual_transition_polarity':'','existing_gt_event_ids':'','same_event_as':'','correction_or_evidence':'사용자의 이전 메시지에서 사실로 확인. 시간/주체/세부 속성은 아직 미확정.' if confirmed else ''})
meta={'schema_version':1,'purpose':'Human fact review candidates, not new GT or hallucination verdicts','source_report_sha256':hashlib.sha256((SOURCE/'report.html').read_bytes()).hexdigest(),'source_details_sha256':hashlib.sha256((SOURCE/'per_video_details.json').read_bytes()).hexdigest(),'video_order':VIDEOS,'n_claims':len(rows),'n_prior_confirmed':sum(r['answer']=='yes' for r in rows),'semantic_curation':'Current conversation LLM read full captions and selected/deduplicated action, direction, actor-link claims; no keyword scoring.','not_exhaustive':'Core dynamic claims plus selected contextual claims. Weather/OCR/scenery/general negative statements omitted. Review full video for events absent from every caption.','video_verified_by_assistant':False}
(ROOT/'review_candidates.json').write_text(json.dumps({'metadata':meta,'claims':rows},ensure_ascii=False,indent=2)+'\n')
with (ROOT/'review_answers.tsv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t'); w.writeheader()
    for r in rows: w.writerow({**r,'source_record_ids':','.join(r['source_record_ids'])})
# Minimal GT provenance kept separate from review answers.
(ROOT/'existing_gt_reference.json').write_text(json.dumps(GT,ensure_ascii=False,indent=2)+'\n')
contexts={v:{'existing_gt':GT[v]['caption_events'],'captions':[{'record_id':r['record_id'],'model':r['model'],'condition':r['condition'],'description':r['description']} for r in D if r['video_id']==v]} for v in VIDEOS}
(ROOT/'source_context.json').write_text(json.dumps(contexts,ensure_ascii=False,indent=2)+'\n')
intro='''영상별 사람 검토표 사용법

이 목록은 report.html과 원본 설명을 읽어 LLM이 의미 단위로 정리한 후보다.
영상을 직접 확인한 결과가 아니며, 기존 hallucinated_gt 표시는 답의 근거로 쓰지 않는다.
원본 GT와 이전 보고서는 변경하지 않았다. 이미 사용자가 사실이라고 확인한 8개 항목만 yes로 표시했다.
이 yes는 사용자가 확인한 행동에만 적용하며, 색·방향·시각·주체 연결까지 확인했다는 뜻은 아니다.

검토 순서
1. 각 영상을 한 번 전체 재생하고, 목록 밖 사건도 발견하면 아래 추가 사건란에 기록한다.
2. core와 confirmed를 먼저 확인하고, detail(방향/인원/주체 연결 등)과 context는 이후 확인한다.
3. 각 문장이 사실이면 yes, 틀리면 no. 식별이 안 되면 unsure로 남긴다. 억지로 yes/no를 선택하지 않는다.
4. 가능하면 원본 클립 기준 초 단위 시작·끝, 주체 ID, 실제 행동/방향, 기존 CE ID를 적는다.
5. 동작은 실제인데 방향/색/주체 연결만 틀리면 no + 수정 내용을 기록한다. no를 사건 전체의 환각으로 자동 변환하지 않는다.
6. 이미 GT에 있는 사건은 CE ID로 연결한다. 같은 사건을 묘사한 여러 항목은 same_event_as로 묶는다.
7. yes이면서 기존 GT에 없는 확인된 사건만 나중에 GT 추가 대상으로 삼는다. 여기서는 자동 편입하지 않는다.

HTML 사용
review.html은 인터넷 없이 동작한다. 먼저 모델 정보 없이 질문을 보고, 필요할 때 원문을 펼친다.
같은 브라우저에서 가능한 경우 로컬 자동 저장한다. 브라우저가 로컬 저장을 차단할 수 있으므로 JSON 내보내기를 반드시 사용한다.
다운로드한 JSON을 같은 검토표에서 다시 가져올 수 있다. HTML 입력은 서버 파일이나 GT에 직접 저장되지 않는다.
TSV는 별도 편집 경로다. HTML과 TSV를 동시에 수정한 내용이 자동 합쳐지지는 않는다.
시간 공란은 사건 부재를 뜻하지 않는다. 모델이 언급한 시간도 검증된 시각으로 취급하지 않는다.
현재 GT 자체의 오류나 누락도 메모한다. 계단 이동/화면 이탈/출입문 통과를 구분한다.
대화·시선·작은 물체처럼 영상만으로 확정하기 어려운 주장은 unsure가 적절하다.
추가 사건은 주체와 시간으로 기존 항목과 중복되지 않는지 확인한다.
'''
(ROOT/'README_ko.txt').write_text(intro)
lines=[intro,'\n영상별 체크리스트\n']
for v in VIDEOS:
    lines.append('\n'+v)
    lines.append('기존 GT: '+', '.join(f"{e['caption_event_id']}={e['action']}({e['start_time']}–{e['end_time']}s; {e.get('direction') or '방향 없음'})" for e in GT[v]['caption_events']))
    for r in rows:
        if r['video_id']==v:
            lines.extend([f"[{r['answer'] or ' '}] {r['review_id']} [{r['priority']}] {r['statement_ko']}", '  확인 포인트: '+r['review_hint_ko'],'  출처: '+', '.join(r['source_record_ids'])])
    lines.append('추가 발견: 목록에 없는 사건 / 시작–끝 / 주체 / 행동 / 방향 / 기존 GT 관계를 기록.')
(ROOT/'CHECKLIST_ko.txt').write_text('\n'.join(lines)+'\n')
payload=json.dumps({'metadata':meta,'claims':rows,'contexts':contexts},ensure_ascii=False).replace('<','\\u003c')
template=(ROOT/'review_template.html').read_text()
(ROOT/'review.html').write_text(template.replace('__DATA__',payload))
print(json.dumps(meta,ensure_ascii=False,indent=2))
print('Per video:',counts)
