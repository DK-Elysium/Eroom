# ======================= 이력서 OCR → CSV 파이프라인 (개선 버전) =======================
# 사용법: python resume_ocr_pipeline.py --input /path/to/pdf_or_folder --out resume_dataset.csv

import os
import re
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pandas as pd
import easyocr

# ---------------------------
# 1) 라벨 스키마
# ---------------------------
RESUME_LABELS = {
    # 기본 정보
    "name": None,
    "email": None,
    "phone": None,
    
    # 학력 정보
    "university": None,
    "university_type": None,  # 인서울/지방대/전문대/기타
    "major": None,
    "major_category": None,  # 주요 전공 or Other
    "gpa": None,
    "gpa_scale": None,
    "grad_year": None,
    
    # 어학 점수
    "english_test_type": None,  # TOEIC/TOEFL/IELTS
    "english_score": None,
    
    # 인턴 경험
    "intern_experiences": [],  # [{company, company_scale, months}]
    "intern_count": 0,
    "intern_total_months": 0,
    
    # 수상 경력
    "awards": [],  # [{name, scale}]
    "award_count": 0,
    
    # 프로젝트
    "projects": [],
    "project_count": 0,
    
    # 자격증
    "certifications": [],
    "certification_count": 0,
    
    # 해외 경험
    "overseas_experiences": [],  # [{type, country, duration}]
    "overseas_count": 0,
    
    # 메타 정보
    "source_pdf": None,
    "error": None,
}

# ---------------------------
# 2) 참조 데이터
# ---------------------------
# 서울 소재 대학교 (주요 대학)
SEOUL_UNIVERSITIES = [
    "서울대", "연세대", "고려대", "성균관대", "한양대", "중앙대", "경희대", "한국외대",
    "서울시립대", "건국대", "동국대", "홍익대", "숙명여대", "이화여대", "서강대",
    "Seoul National", "Yonsei", "Korea University", "Sungkyunkwan", "Hanyang",
    "Chung-Ang", "Kyung Hee", "HUFS", "Seoul National University of Science"
]

# 전문대 키워드
COLLEGE_KEYWORDS = ["전문대", "College", "Polytechnic", "기능대학"]

# 주요 전공 리스트
MAJOR_CATEGORIES = {
    "컴퓨터공학": ["컴퓨터", "Computer Science", "Software", "소프트웨어"],
    "전기전자공학": ["전기", "전자", "Electrical", "Electronic", "반도체"],
    "기계공학": ["기계", "Mechanical", "자동차"],
    "화학공학": ["화학", "Chemical", "화공"],
    "경영학": ["경영", "Business", "Management", "MBA"],
    "경제학": ["경제", "Economics"],
    "산업공학": ["산업공학", "Industrial Engineering"],
    "건축학": ["건축", "Architecture"],
    "생명공학": ["생명", "Biotechnology", "Bio"],
    "수학": ["수학", "Mathematics", "통계"],
    "물리학": ["물리", "Physics"],
    "디자인": ["디자인", "Design"],
}

# 대기업 리스트 (확장 가능)
LARGE_COMPANIES = [
    "삼성", "Samsung", "LG", "SK", "현대", "Hyundai", "기아", "Kia",
    "포스코", "POSCO", "한화", "Hanwha", "롯데", "Lotte", "GS", "두산", "Doosan",
    "네이버", "Naver", "카카오", "Kakao", "쿠팡", "Coupang", "배달의민족", 
    "구글", "Google", "메타", "Meta", "아마존", "Amazon", "마이크로소프트", "Microsoft",
    "애플", "Apple", "테슬라", "Tesla", "넷플릭스", "Netflix"
]

# 중견기업 키워드
MIDSIZE_KEYWORDS = ["중견", "코스닥", "KOSDAQ", "상장"]

# ---------------------------
# 3) PDF → 이미지
# ---------------------------
def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """PDF를 이미지 리스트로 변환"""
    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

# ---------------------------
# 4) OCR 수행
# ---------------------------
def ocr_images(images: List[Image.Image], languages: List[str]) -> str:
    """이미지에서 텍스트 추출"""
    reader = easyocr.Reader(languages, gpu=False)
    all_texts: List[str] = []
    
    for img in images:
        texts = reader.readtext(np.array(img), detail=0)
        page_text = "\n".join(t.strip() for t in texts if t.strip())
        all_texts.append(page_text)
    
    return "\n".join(all_texts)

# ---------------------------
# 5) 대학교 분류
# ---------------------------
def classify_university(university_name: str) -> str:
    """대학을 인서울/지방대/전문대/기타로 분류"""
    if not university_name:
        return "기타"
    
    # 전문대 체크
    for keyword in COLLEGE_KEYWORDS:
        if keyword in university_name:
            return "전문대"
    
    # 인서울 체크
    for seoul_uni in SEOUL_UNIVERSITIES:
        if seoul_uni in university_name:
            return "인서울"
    
    # 대학교/대학 키워드가 있으면 지방대
    if "대학교" in university_name or "University" in university_name:
        return "지방대"
    
    return "기타"

# ---------------------------
# 6) 전공 분류
# ---------------------------
def classify_major(major_name: str) -> str:
    """전공을 주요 카테고리 또는 Other로 분류"""
    if not major_name:
        return "Other"
    
    for category, keywords in MAJOR_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in major_name.lower():
                return category
    
    return "Other"

# ---------------------------
# 7) 회사 규모 분류
# ---------------------------
def classify_company_scale(company_name: str) -> str:
    """회사를 대기업/중견/중소로 분류"""
    if not company_name:
        return "중소"
    
    # 대기업 체크
    for large_comp in LARGE_COMPANIES:
        if large_comp.lower() in company_name.lower():
            return "대기업"
    
    # 중견기업 체크
    for keyword in MIDSIZE_KEYWORDS:
        if keyword in company_name:
            return "중견"
    
    return "중소"

# ---------------------------
# 8) 수상 규모 분류
# ---------------------------
def classify_award_scale(award_text: str) -> str:
    """수상을 국제/전국/지역/교내로 분류"""
    award_lower = award_text.lower()
    
    if any(kw in award_lower for kw in ["international", "world", "global", "국제", "세계"]):
        return "국제"
    elif any(kw in award_lower for kw in ["national", "전국", "한국", "korea"]):
        return "전국"
    elif any(kw in award_lower for kw in ["regional", "지역", "시", "도"]):
        return "지역"
    elif any(kw in award_lower for kw in ["university", "college", "school", "대학", "교내", "학교"]):
        return "교내"
    
    return "교내"  # 기본값

# ---------------------------
# 9) 정규식 헬퍼
# ---------------------------
def _match_first(regexes: List[re.Pattern], text: str) -> Optional[str]:
    """여러 정규식 중 첫 번째 매칭 결과 반환"""
    for rgx in regexes:
        m = rgx.search(text)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None

# ---------------------------
# 10) 텍스트 파싱 (핵심 로직)
# ---------------------------
def parse_text_to_features(text: str) -> Dict[str, Any]:
    """텍스트에서 이력서 정보 추출"""
    data = {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in RESUME_LABELS.items()}
    
    norm = text.replace("\r", "\n")
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]
    lower = norm.lower()

    # ========== 기본 정보 ==========
    # 이름 (최상단 라인)
    if lines:
        top = lines[0]
        if len(top) <= 50 and not re.search(r"@|github|linkedin|http", top, re.I):
            data["name"] = top

    # 이메일
    email = _match_first([re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")], norm)
    data["email"] = email

    # 전화번호
    phone = _match_first([
        re.compile(r"(\+82[-\s]?\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4})"),
        re.compile(r"(01[0-9][-\s]?\d{3,4}[-\s]?\d{4})"),
    ], norm)
    data["phone"] = phone

    # ========== 학력 정보 ==========
    # 대학교
    uni = _match_first([
        re.compile(r"([A-Z][A-Za-z&.\s]{2,}University)"),
        re.compile(r"([가-힣A-Za-z&.\s]{2,}대학교)"),
        re.compile(r"([가-힣A-Za-z&.\s]{2,}대학)"),
    ], norm)
    data["university"] = uni
    if uni:
        data["university_type"] = classify_university(uni)

    # 전공
    major = _match_first([
        re.compile(r"(Computer Science|Software Engineering|Electrical Engineering|Information Technology|Data Science)", re.I),
        re.compile(r"([가-힣A-Za-z/&\s]{2,20}(?:과|학과|전공))"),
    ], norm)
    data["major"] = major
    if major:
        data["major_category"] = classify_major(major)

    # 학점
    gpa_match = _match_first([
        re.compile(r"GPA\s*[:=]?\s*([0-4]\.\d{1,2})\s*/\s*([0-4]\.?\d*)", re.I),
        re.compile(r"평점\s*[:=]?\s*([0-4]\.\d{1,2})\s*/\s*([0-4]\.?\d*)"),
    ], norm)
    if gpa_match:
        parts = re.findall(r"(\d\.\d+)", gpa_match)
        if len(parts) >= 2:
            data["gpa"] = float(parts[0])
            data["gpa_scale"] = float(parts[1])
        elif len(parts) == 1:
            data["gpa"] = float(parts[0])
            data["gpa_scale"] = 4.5

    # 졸업 연도
    grad_year = _match_first([
        re.compile(r"(20\d{2})\s*(?:년)?\s*(?:졸업|Graduation)", re.I)
    ], norm)
    if grad_year:
        data["grad_year"] = int(grad_year)

    # ========== 어학 점수 ==========
    toeic = _match_first([re.compile(r"TOEIC\s*[:=]?\s*(\d{3,4})", re.I)], norm)
    ielts = _match_first([re.compile(r"IELTS\s*[:=]?\s*(\d(?:\.\d)?)", re.I)], norm)
    toefl = _match_first([re.compile(r"TOEFL\s*[:=]?\s*(\d{2,3})", re.I)], norm)
    
    if toeic:
        data["english_test_type"] = "TOEIC"
        data["english_score"] = int(toeic)
    elif ielts:
        data["english_test_type"] = "IELTS"
        data["english_score"] = float(ielts)
    elif toefl:
        data["english_test_type"] = "TOEFL"
        data["english_score"] = int(toefl)

    # ========== 인턴 경험 ==========
    # 인턴 키워드 찾기
    intern_patterns = [
        re.compile(r"(인턴|Intern|Internship)\s+(?:at\s+)?([^\n,]+?)(?:\s+\()?(\d+)\s*(?:개월|months?|mos?)", re.I),
        re.compile(r"([^\n,]+?)\s+(?:인턴|Intern).*?(\d+)\s*(?:개월|months?)", re.I),
    ]
    
    intern_list = []
    for pattern in intern_patterns:
        for match in pattern.finditer(norm):
            try:
                if len(match.groups()) >= 3:
                    company = match.group(2).strip()
                    months = int(match.group(3))
                elif len(match.groups()) >= 2:
                    company = match.group(1).strip()
                    months = int(match.group(2))
                else:
                    continue
                
                scale = classify_company_scale(company)
                intern_list.append({
                    "company": company,
                    "company_scale": scale,
                    "months": months
                })
            except:
                continue
    
    data["intern_experiences"] = intern_list
    data["intern_count"] = len(intern_list)
    data["intern_total_months"] = sum(i["months"] for i in intern_list)

    # ========== 수상 경력 ==========
    award_patterns = [
        re.compile(r"(수상|Award|Prize)\s*[:：]?\s*([^\n]+)", re.I),
        re.compile(r"([^\n]+?)\s+(?:대회|Competition|Contest).*?(?:수상|상|Award|Prize)", re.I),
    ]
    
    award_list = []
    for pattern in award_patterns:
        for match in pattern.finditer(norm):
            award_text = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            award_text = award_text.strip()[:100]  # 길이 제한
            
            scale = classify_award_scale(award_text)
            award_list.append({
                "name": award_text,
                "scale": scale
            })
    
    # 중복 제거
    seen = set()
    unique_awards = []
    for award in award_list:
        key = award["name"][:50]
        if key not in seen:
            seen.add(key)
            unique_awards.append(award)
    
    data["awards"] = unique_awards
    data["award_count"] = len(unique_awards)

    # ========== 프로젝트 ==========
    project_patterns = [
        re.compile(r"(Project|프로젝트)\s*[:：]?\s*([^\n]+)", re.I),
    ]
    
    project_list = []
    for pattern in project_patterns:
        for match in pattern.finditer(norm):
            proj_name = match.group(2).strip()[:100]
            if proj_name and len(proj_name) > 3:
                project_list.append(proj_name)
    
    data["projects"] = list(set(project_list))[:10]  # 최대 10개
    data["project_count"] = len(data["projects"])

    # ========== 자격증 ==========
    cert_patterns = [
        re.compile(r"(자격증|Certificate|Certification)\s*[:：]?\s*([^\n]+)", re.I),
        re.compile(r"([가-힣A-Za-z\s]+기사)", re.I),
        re.compile(r"(SQLD|ADsP|정보처리기사|네트워크관리사)", re.I),
    ]
    
    cert_list = []
    for pattern in cert_patterns:
        for match in pattern.finditer(norm):
            cert_name = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            cert_name = cert_name.strip()[:50]
            if cert_name and len(cert_name) > 2:
                cert_list.append(cert_name)
    
    data["certifications"] = list(set(cert_list))[:10]
    data["certification_count"] = len(data["certifications"])

    # ========== 해외 경험 ==========
    overseas_patterns = [
        re.compile(r"(교환학생|Exchange Student|Study Abroad).*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I),
        re.compile(r"(어학연수|Language Training).*?([A-Z][a-z]+)", re.I),
        re.compile(r"(해외 인턴|International Intern).*?([A-Z][a-z]+)", re.I),
    ]
    
    overseas_list = []
    for pattern in overseas_patterns:
        for match in pattern.finditer(norm):
            exp_type = match.group(1).strip()
            country = match.group(2).strip() if len(match.groups()) >= 2 else "Unknown"
            
            overseas_list.append({
                "type": exp_type,
                "country": country,
                "duration": None  # 기간은 추가 파싱 필요
            })
    
    data["overseas_experiences"] = overseas_list
    data["overseas_count"] = len(overseas_list)

    return data

# ---------------------------
# 11) 단일 PDF 처리
# ---------------------------
def process_resume(pdf_path: str, languages: List[str], dpi: int = 200) -> Dict[str, Any]:
    """PDF 이력서를 처리하여 데이터 추출"""
    try:
        # 1. PDF → 이미지
        images = pdf_to_images(pdf_path, dpi=dpi)
        
        # 2. OCR 수행
        ocr_text = ocr_images(images, languages=languages)
        
        # 3. PyMuPDF 텍스트 레이어도 추출 (보강)
        doc = fitz.open(pdf_path)
        pdf_text = []
        for page in doc:
            pdf_text.append(page.get_text("text"))
        doc.close()
        
        # 두 텍스트 합치기 (긴 것 우선)
        combined_text = ocr_text if len(ocr_text) > len("\n".join(pdf_text)) else "\n".join(pdf_text)
        
        # 4. 파싱
        features = parse_text_to_features(combined_text)
        features["source_pdf"] = os.path.basename(pdf_path)
        
        return features
        
    except Exception as e:
        row = {k: None for k in RESUME_LABELS.keys()}
        row["source_pdf"] = os.path.basename(pdf_path)
        row["error"] = str(e)
        return row

# ---------------------------
# 12) 배치 처리
# ---------------------------
def batch_build_dataset(input_path: str, languages: List[str], dpi: int = 200) -> pd.DataFrame:
    """여러 PDF를 처리하여 데이터셋 생성"""
    paths: List[str] = []
    
    if os.path.isdir(input_path):
        for name in os.listdir(input_path):
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(input_path, name))
    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        paths = [input_path]
    else:
        raise FileNotFoundError("PDF 파일 또는 PDF가 있는 폴더를 입력하세요.")

    print(f"📄 총 {len(paths)}개의 PDF 파일을 처리합니다...")
    
    rows = []
    for idx, pdf_path in enumerate(sorted(paths), 1):
        print(f"[{idx}/{len(paths)}] 처리 중: {os.path.basename(pdf_path)}")
        result = process_resume(pdf_path, languages=languages, dpi=dpi)
        rows.append(result)
    
    return pd.DataFrame(rows)

# ---------------------------
# 13) CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="이력서 PDF를 OCR로 분석하여 구조화된 CSV 데이터셋 생성"
    )
    parser.add_argument("--input", required=True, help="PDF 파일 경로 또는 폴더 경로")
    parser.add_argument("--out", default="resume_dataset.csv", help="저장할 CSV 파일명 (기본: resume_dataset.csv)")
    parser.add_argument("--dpi", type=int, default=200, help="PDF 렌더링 DPI (기본: 200)")
    parser.add_argument("--langs", nargs="+", default=["ko", "en"], help="EasyOCR 언어 리스트 (기본: ko en)")
    
    args = parser.parse_args()

    # 데이터셋 생성
    df = batch_build_dataset(args.input, languages=args.langs, dpi=args.dpi)
    
    # CSV 저장
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 완료! {len(df)}개의 이력서 데이터를 저장했습니다.")
    print(f"📊 저장 경로: {os.path.abspath(args.out)}")
    print(f"\n📈 통계:")
    print(f"  - 인서울: {(df['university_type'] == '인서울').sum()}명")
    print(f"  - 지방대: {(df['university_type'] == '지방대').sum()}명")
    print(f"  - 전문대: {(df['university_type'] == '전문대').sum()}명")
    print(f"  - 평균 인턴 경험: {df['intern_count'].mean():.1f}회")
    print(f"  - 평균 수상: {df['award_count'].mean():.1f}회")

if __name__ == "__main__":
    main()