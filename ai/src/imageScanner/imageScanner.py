# ======================= 이력서 OCR → CSV 파이프라인 (개선 버전) =======================
# 사용법: python imageScanner.py --input /path/to/pdf_or_folder --out resume_dataset.csv

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
    "school": None, #학교 이름 
    "major" : None, #전공 이름
    "gpa" : None, #학점

    "intern_company_scale" : None, #인턴 회사 규모
    "intern_total_months" : 0, #인턴 총 개월 수
    "intern_count" : 0, #인턴 횟수

    "award_level" : None, #수상 레벨 요약

    "project_count" : 0, #프로젝트 개수
    "cert_count" : 0, #자격증 개수

    "has_language_cert" : False, #어학 자격증 보유 여부
    "overseas_experience" : None, #해외 경험 종류

    "company_size" : None, #희망 회사 규모
    "industry" : None, #희망 산업 분야
    "job_role" : None, #희망 직무

}

# ---------------------------
# 2) 참조 데이터
# ---------------------------

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

    # ========== 학력 정보 ==========
    # 학교 이름
    uni = _match_first([
        re.compile(r"([A-Z][A-Za-z&.\s]{2,}University)"),
        re.compile(r"([가-힣A-Za-z&.\s]{2,}대학교)"),
        re.compile(r"([가-힣A-Za-z&.\s]{2,}대학)"),
    ], norm)
    data["school"] = uni
    
    # 전공 이름
    major = _match_first([
        re.compile(r"(Computer Science|Software Engineering|Electrical Engineering|Information Technology|Data Science)", re.I),
        re.compile(r"([가-힣A-Za-z/&\s]{2,20}(?:과|학과|전공))"),
    ], norm)
    data["major"] = major
    
    # 학점
        # ========== GPA / 학점 ==========
    gpa_value = None

    # 패턴 1: "GPA 3.75 / 4.5"
    m = re.search(r"GPA\s*[:=]?\s*([0-4](?:\.\d{1,2})?)\s*/\s*[0-4](?:\.\d{1,2})?", norm, re.I)
    # 패턴 2: "평점 3.8 / 4.5"
    if not m:
        m = re.search(r"평점\s*[:=]?\s*([0-4](?:\.\d{1,2})?)\s*/\s*[0-4](?:\.\d{1,2})?", norm)

    # 패턴 3: "GPA 3.75" (스케일 없이 단독 숫자만 써놓은 경우)
    if not m:
        m = re.search(r"GPA\s*[:=]?\s*([0-4](?:\.\d{1,2})?)", norm, re.I)

    # 패턴 4: "평점 3.8"
    if not m:
        m = re.search(r"평점\s*[:=]?\s*([0-4](?:\.\d{1,2})?)", norm)

    if m:
        try:
            gpa_value = float(m.group(1))
        except ValueError:
            gpa_value = None

    data["gpa"] = gpa_value



    # ========== 인턴 경험 ==========
         # ========== 인턴 경험 ==========
    intern_patterns = [
        # 패턴 1: "인턴 at 회사명 3개월" / "Internship at 카카오 6 months"
        re.compile(r"(인턴|Intern|Internship)\s+(?:at\s+)?([^\n,]+?)(?:\s+\()?(\d+)\s*(?:개월|months?|mos?)", re.I),

        # 패턴 2: "카카오 인턴 3개월" / "네이버 Intern 6개월"
        re.compile(r"([^\n,]+?)\s+(?:인턴|Intern).*?(\d+)\s*(?:개월|months?)", re.I),
    ]

    intern_count = 0              # 인턴 횟수
    intern_total_months = 0       # 인턴 총 개월 수
    intern_scales = []            # 회사 규모들 모아두기 (대기업/중견/중소 등)
    seen_interns = set()          # (company, months) 중복 방지용

    for pattern in intern_patterns:
        for match in pattern.finditer(norm):
            try:
                # 그룹 개수에 따라 company / months 위치가 조금 다름
                if len(match.groups()) >= 3:
                    # 패턴 1: (인턴단어, 회사명, 개월수)
                    company = match.group(2).strip()
                    months = int(match.group(3))
                elif len(match.groups()) >= 2:
                    # 패턴 2: (회사명, 개월수)
                    company = match.group(1).strip()
                    months = int(match.group(2))
                else:
                    continue

                # 🔒 중복 체크 (같은 회사 + 같은 개월 수면 한 번만 카운트)
                key = (company, months)
                if key in seen_interns:
                    continue
                seen_interns.add(key)

                # 회사 규모 분류 (대기업/중견/중소/기타 등)
                scale = classify_company_scale(company)

                # 인턴 개수 +1
                intern_count += 1

                # 총 개월 수 누적
                intern_total_months += months

                # 회사 규모 목록에 추가 (나중에 제일 큰 규모 뽑기 위함)
                if scale:
                    intern_scales.append(scale)

            except Exception:
                # 파싱 실패하면 그냥 그 케이스만 건너뜀
                continue

    # 인턴 관련 요약 라벨 채우기
    data["intern_count"] = intern_count
    data["intern_total_months"] = intern_total_months

    # intern_company_scale: 여러 인턴 중 "제일 큰" 회사 규모 하나만 요약해서 넣기
    best_scale = None
    priority = ["대기업", "중견", "중소", "기타"]

    for cand in priority:
        if cand in intern_scales:
            best_scale = cand
            break

    if best_scale is None and intern_scales:
        best_scale = intern_scales[0]

    data["intern_company_scale"] = best_scale


    # ========== 수상 경력 / Award Level 요약 ==========
    award_patterns = [
        # 패턴 1: "수상: 교내 캡스톤디자인 경진대회 최우수상", "Award: ..."
        re.compile(r"(수상|Award|Prize)\s*[:：]?\s*([^\n]+)", re.I),

        # 패턴 2: "교내 캡스톤 경진대회 최우수상 수상", "XXX Competition Award"
        re.compile(r"([^\n]+?)\s+(?:대회|Competition|Contest).*?(?:수상|상|Award|Prize)", re.I),
    ]

    award_scales = []   # "국제", "전국", "지역", "교내" 같은 scale 들만 모아두기

    for pattern in award_patterns:
        for match in pattern.finditer(norm):
            # 그룹 개수에 따라 award 텍스트 가져오기
            award_text = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            award_text = award_text.strip()[:100]  # 너무 길면 100자까지만

            # 수상 텍스트에서 scale 분류 (국제/전국/지역/교내/기타 ...)
            scale = classify_award_scale(award_text)
            if scale:
                award_scales.append(scale)

    # award_level: 여러 수상 중 "제일 높은" 레벨 하나만 결정
    # 우선순위 예시: 국제 > 전국 > 지역 > 교내
    best_award_level = None
    priority = ["국제", "전국", "지역", "교내"]

    for cand in priority:
        if cand in award_scales:
            best_award_level = cand
            break

    # priority 안에 없는 스케일들만 있다면 (예: "기타", "교외") 그 중 하나라도 사용
    if best_award_level is None and award_scales:
        best_award_level = award_scales[0]

    data["award_level"] = best_award_level

    # ========== 프로젝트 개수 추출 ==========
    project_patterns = [
        re.compile(r"(Project|프로젝트)\s*[:：]?\s*([^\n]+)", re.I),
    ]

    project_count = 0

    for pattern in project_patterns:
        for match in pattern.finditer(norm):
            proj_name = match.group(2).strip()[:100]

            # 프로젝트 이름이 너무 짧으면 잡히는 "프로젝트" 키워드 잡음 제거
            if proj_name and len(proj_name) > 3:
                project_count += 1

    data["project_count"] = project_count

    # ========== 자격증 개수 추출 ==========
    cert_patterns = [
        # "자격증: 정보처리기사" / "Certificate: AWS" 같은 케이스
        re.compile(r"(자격증|Certificate|Certification)\s*[:：]?\s*([^\n]+)", re.I),

        # "~기사" 계열(정보처리기사, 전자기사 등)
        re.compile(r"([가-힣A-Za-z\s]+기사)", re.I),

        # 대표적인 IT 자격증들
        re.compile(r"(SQLD|ADsP|정보처리기사|네트워크관리사)", re.I),
    ]

    cert_count = 0

    for pattern in cert_patterns:
        for match in pattern.finditer(norm):
            cert_name = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            cert_name = cert_name.strip()[:50]

            # 너무 짧거나 쓰레기 매칭된 이름 제거
            if cert_name and len(cert_name) > 2:
                cert_count += 1

    data["cert_count"] = cert_count

    
    # ========== 어학 점수 존재 여부만 추출 ==========
    toeic = _match_first([re.compile(r"TOEIC\s*[:=]?\s*(\d{3,4})", re.I)], norm)
    ielts = _match_first([re.compile(r"IELTS\s*[:=]?\s*(\d(?:\.\d)?)", re.I)], norm)
    toefl = _match_first([re.compile(r"TOEFL\s*[:=]?\s*(\d{2,3})", re.I)], norm)

    # 영어 성적 있는지만 체크
    has_language_cert = False
    if toeic or ielts or toefl:
        has_language_cert = True

    data["has_language_cert"] = has_language_cert


    # ========== 해외 경험 ==========
        # ========== 해외 경험 문자열 추출 (교환학생/어학연수/해외 인턴 등) ==========
    overseas_patterns = [
        (re.compile(r"교환학생|Exchange Student|Study Abroad", re.I), "Exchange"),
        (re.compile(r"어학연수|Language Training", re.I), "LanguageStudy"),
        (re.compile(r"해외 인턴|International Intern", re.I), "OverseasIntern"),
    ]

    overseas_type = None

    for pattern, label in overseas_patterns:
        if pattern.search(norm):
            overseas_type = label
            break

    data["overseas_experience"] = overseas_type  # 문자열로 저장 (None or "교환학생" 등)


    # ======== 지원 회사 분야 ========
    # 1) 희망 회사 규모 (company_size)
       # ========== 희망 회사 규모 추출 (classify_company_scale 재활용) ==========
    hope_company = None

    # 희망 회사/지원 회사 텍스트 찾기
    hope_patterns = [
        re.compile(r"(?:희망\s*회사|지원\s*회사|지원\s*기업|입사\s*희망)\s*[:：]?\s*([^\n]+)", re.I),
        re.compile(r"([A-Za-z가-힣0-9&.\s]+)\s+(?:입사\s*희망|지원하고자)", re.I),
        re.compile(r"(?:at\s+)?([A-Za-z가-힣0-9&.\s]+)\s+입사", re.I),
    ]

    for pattern in hope_patterns:
        m = pattern.search(norm)
        if m:
            hope_company = m.group(1).strip()[:50]
            break

    # 회사 규모 분류 함수 활용
    if hope_company:
        company_scale = classify_company_scale(hope_company)
        data["company_size"] = company_scale  # 기존 함수 결과 그대로 사용
    else:
        data["company_size"] = None



    # 2) 희망 산업 분야 (industry)
    industry = None

    # IT / 소프트웨어 / 데이터 / AI
    if re.search(r"IT|소프트웨어|SW|개발자|백엔드|프론트엔드|웹\s*개발|앱\s*개발|인공지능|AI|데이터\s*사이언스|데이터\s*분석", norm, re.I):
        industry = "IT/Software"
    # 금융
    elif re.search(r"금융|은행|증권|보험|핀테크|Fintech", norm, re.I):
        industry = "Finance"
    # 제조 / 전자 / 반도체
    elif re.search(r"제조|Manufacturing|생산|공장|반도체|전자\s*산업", norm, re.I):
        industry = "Manufacturing"
    # 헬스케어 / 바이오 / 의료
    elif re.search(r"헬스케어|의료|병원|바이오|제약|Bio|Healthcare", norm, re.I):
        industry = "Healthcare"
    # 교육
    elif re.search(r"교육|에듀테크|Edtech", norm, re.I):
        industry = "Education"
    # 커머스 / 유통 / 이커머스
    elif re.search(r"커머스|이커머스|e[-\s]?commerce|유통|리테일", norm, re.I):
        industry = "Commerce/Retail"

    data["industry"] = industry

    # ---------------------------
    # 3) 희망 직무 (job_role)
    # ---------------------------
    job_role = None

    # 먼저 '희망 직무/지원 직무/지원 분야' 같은 단어 근처에서 한 번 더 강하게 찾고,
    # 없으면 전체 텍스트에서 찾아도 됨. 여기서는 간단히 전체 텍스트 기준으로만 처리.

    # Backend / Server
    if re.search(r"백엔드|서버\s*개발|Backend", norm, re.I):
        job_role = "Backend"
    # Frontend / Web UI
    elif re.search(r"프론트엔드|웹\s*개발|Frontend", norm, re.I):
        job_role = "Frontend"
    # Mobile (Android / iOS)
    elif re.search(r"모바일\s*앱|Android\s*개발|iOS\s*개발|모바일\s*개발", norm, re.I):
        job_role = "Mobile"
    # Data Analyst / Scientist
    elif re.search(r"데이터\s*분석|Data\s*Analyst|데이터\s*사이언티스트|Data\s*Scientist", norm, re.I):
        job_role = "Data Scientist"
    # ML / AI Engineer
    elif re.search(r"머신러닝|Machine\s*Learning|ML\s*Engineer|AI\s*Engineer|인공지능\s*엔지니어", norm, re.I):
        job_role = "ML/AI Engineer"
    # DevOps / Infra
    elif re.search(r"DevOps|데브옵스|인프라\s*엔지니어|클라우드\s*엔지니어", norm, re.I):
        job_role = "DevOps"
    # Product Manager / PM
    elif re.search(r"Product\s*Manager|PM\s*\(Product\)|프로덕트\s*매니저", norm, re.I):
        job_role = "Product Manager"
    # 일반적인 "SW Engineer", "Software Engineer"
    elif re.search(r"Software\s*Engineer|SW\s*Engineer|소프트웨어\s*엔지니어", norm, re.I):
        job_role = "Software Engineer"

    data["job_role"] = job_role

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
    parser.add_argument(
        "--out",
        default="resume_dataset.csv",
        help="저장할 CSV 파일명 (기본: resume_dataset.csv)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF 렌더링 DPI (기본: 200)"
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["ko", "en"],
        help="EasyOCR 언어 리스트 (기본: ko en)"
    )
    
    args = parser.parse_args()

    # CSV 저장
    df = batch_build_dataset(args.input, languages=args.langs, dpi=args.dpi)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료! {len(df)}개의 이력서 데이터를 저장했습니다.")
    print(f"📊 저장 경로: {os.path.abspath(args.out)}")

    # ===================== 새 통계 출력 =====================
    print("\n📈 통계:")

    # 1) 총 이력서 개수
    print(f"  - 총 이력서 개수: {len(df)}개")

    # 2) school (대학교 타입 요약) 통계가 있으면 출력
    if "school" in df.columns:
        print("  - School 분포:")
        print(df["school"].value_counts(dropna=False))

    # 3) 인턴 / 프로젝트 / 자격증 평균
    if "intern_count" in df.columns:
        print(f"  - 평균 인턴 횟수: {df['intern_count'].mean():.1f}회")
    if "intern_total_months" in df.columns:
        print(f"  - 평균 인턴 총 개월 수: {df['intern_total_months'].mean():.1f}개월")

    if "project_count" in df.columns:
        print(f"  - 평균 프로젝트 개수: {df['project_count'].mean():.1f}개")

    if "cert_count" in df.columns:
        print(f"  - 평균 자격증 개수: {df['cert_count'].mean():.1f}개")

    # 4) 어학 자격 보유 비율
    if "has_language_cert" in df.columns:
        lang_ratio = (df["has_language_cert"] == True).mean() * 100
        print(f"  - 어학 자격/점수 보유 비율: {lang_ratio:.1f}%")

    # 5) 해외 경험 분포 (overseas_experience: 문자열)
    if "overseas_experience" in df.columns:
        print("  - 해외 경험 유형 분포:")
        print(df["overseas_experience"].value_counts(dropna=False))

    # 6) 희망 회사 규모 / 산업 / 직무 분포
    if "company_size" in df.columns:
        print("  - 희망 회사 규모 분포:")
        print(df["company_size"].value_counts(dropna=False))

    if "industry" in df.columns:
        print("  - 희망 산업 분야 분포:")
        print(df["industry"].value_counts(dropna=False))

    if "job_role" in df.columns:
        print("  - 희망 직무 분포:")
        print(df["job_role"].value_counts(dropna=False))
    # =======================================================


if __name__ == "__main__":
    main()




    # python imageScanner.py  --input . --out resume_dataset.csv  
    # python imageScanner.py --input ./pdfs --out resume_dataset.csv
