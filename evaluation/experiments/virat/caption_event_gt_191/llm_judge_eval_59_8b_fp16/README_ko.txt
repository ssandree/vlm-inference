59영상 8b-fp16 평가 결과

REPORT_ko.txt / report.html: 원본 GT 기반 요약 및 상세 판정
PRIORITY_CHECKLIST_ko.txt: 먼저 확인할 구체적 의문 41개
CHECKLIST_ko.txt: 전체 59영상, 291개 항목
review.html: yes/no/unsure 입력, 로컬 임시 저장 및 JSON 가져오기/내보내기
review_answers.tsv: 텍스트 편집 경로. HTML과 자동 동기화되지 않음

평가 대상: 비어 있지 않은 GT 33영상(주 평가), 빈 GT 11영상(민감도), GT 누락 15영상(채점 제외). 전체 118개 설명은 읽고 판정/검토 자료에 보존.
보행/달리기는 locomotion으로 통합. 원본 GT에서 건물 입장/퇴장은 별도 행동. 이전 사람 검토 GT v2의 점수와 직접 비교 불가.
GT 비지원은 실제 환각이 아님. 원본 영상 검증 전의 체크리스트 초기값은 모두 unsure. 항목들은 같은 사건을 행동/방향/세부 측면에서 중복 검토할 수 있으므로 독립 사건 수로 해석하지 말 것.

재집계: python evaluate.py
판정 직렬화: python author_judgments.py (새 모델/API 실행 아님)
체크리스트 생성: python build_review.py (생성 대상 TXT/TSV/HTML을 덮어쓰므로 작성한 응답은 별도 JSON으로 내보내 보관)
