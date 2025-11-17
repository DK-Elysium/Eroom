import json
import random

# --- 1. 재료 (이전과 동일) ---
SCHOOLS = ["중앙대학교", "가천대학교", "중앙대학교", "해외대학교", "고려대학교"]
GPAS = ["3.25", "3.5", "3.75", "3.9", "4.1", "4.3"]
ENG_SCORES = ["오픽: IH", "오픽: AL", "토익: 850", "토익: 900", "토익스피킹: IM3", "토익스피킹: AL", None]

MAJOR_SKILLS = {
    "기계공학과": ["CAD", "CATIA", "Ansys", "MOS"],
    "컴퓨터공학과": ["Python", "Django", "AWS", "React", "정보처리기사"],
    "시각디자인과": ["Figma", "Adobe XD", "Photoshop", "Illustrator"],
    "경영학과": ["MOS excel", "SQL", "GA", "Python(데이터분석)"]
}
AWARDS = ["IoT가전 캡스톤디자인 경진대회(총장상)", "교내 해커톤 대상", "레드닷 디자인 어워드", None, None]
templates = [
    "{school} / {major} / 학점 {gpa} / {eng} / {award} / 스킬: {skill}",
    "{school}({major}) | 학점: {gpa} | {eng} | {skill} 보유 | {award}",
    "전공: {major} (학점 {gpa}) | {school} | {skill}, {skill2} | {eng}",
    "{eng} {gpa}점, {school} {major} | {award} | {skill} 경험"
]

# --- 2. [신규] 모든 스킬을 모아둔 '범용 스킬 리스트' 생성 ---
ALL_SKILLS = []
for skills in MAJOR_SKILLS.values():
    ALL_SKILLS.extend(skills)

# 중복 제거
ALL_SKILLS = list(set(ALL_SKILLS)) 

print(f"범용 스킬 목록: {ALL_SKILLS}")


# --- 3. [개선] 가상 데이터를 '확률 기반'으로 생성하는 함수 ---
def generate_realistic_spec():
    # 1. 전공 먼저 선택
    major = random.choice(list(MAJOR_SKILLS.keys()))
    
    # --- ▼▼▼ 여기가 핵심 개선점 ▼▼▼ ---
    # 80% 확률 (random.random()이 0.8보다 작으면)
    if random.random() < 0.8:
        # 80%는 전공 관련 스킬을 선택
        skill1 = random.choice(MAJOR_SKILLS[major])
        skill2 = random.choice(MAJOR_SKILLS[major])
    else:
        # 20%는 전공과 '무관한' 스킬을 '범용 리스트'에서 선택
        print(f"** 비전공 스킬 조합 발생! (전공: {major}) **") # 확인용 로그
        skill1 = random.choice(ALL_SKILLS) 
        skill2 = random.choice(ALL_SKILLS)
    # --- ▲▲▲ 여기까지 개선점 ▲▲▲ ---

    # 2. 나머지 정보 랜덤 선택
    school = random.choice(SCHOOLS)
    gpa = random.choice(GPAS)
    eng = random.choice(ENG_SCORES)
    award = random.choice(AWARDS)
    
    # 3. 템플릿 랜덤 선택
    template = random.choice(templates)
    
    # 4. 템플릿에 데이터 채우기
    spec_text = template.format(
        school=school, major=major, gpa=gpa,
        eng=eng if eng else "어학성적 없음",
        award=award if award else "수상내역 없음",
        skill=skill1, skill2=skill2
    )
    
    # 5. '없음' 항목 정리
    spec_text = spec_text.replace(" / 어학성적 없음", "") \
                         .replace(" | 어학성적 없음", "") \
                         .replace(" / 수상내역 없음", "") \
                         .replace(" | 수상내역 없음", "")
    return spec_text

# --- 4. [실행] 파일로 저장 (이하 동일) ---
OUTPUT_FILE = "specs_realistic.jsonl"
NUM_TO_GENERATE = 1000 

print(f"\n'{OUTPUT_FILE}' (확률 기반) 파일 생성을 시작합니다...")

try:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in range(NUM_TO_GENERATE):
            spec_text = generate_realistic_spec()
            json_line = {"text": spec_text}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            
    print(f"총 {NUM_TO_GENERATE}개의 '더 현실적인' 모의 데이터를 저장했습니다.")

except Exception as e:
    print(f"작업 중 오류 발생: {e}")