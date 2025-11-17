# ======================= 복붙 시작: resume_ocr_pipeline.py =======================
# 이력서 PDF → 이미지 렌더링 → EasyOCR 텍스트 추출 → 스펙 라벨 파싱 → CSV 저장
# 사용법:
#   python resume_ocr_pipeline.py --input /path/to/pdf_or_folder --out resume_dataset.csv

import os
import re
import argparse
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pandas as pd
import easyocr

# ---------------------------
# 1) 라벨 스키마(원하는대로 커스텀)
# ---------------------------
RESUME_LABELS = {
    #기본정보
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

PROGRAMMING_LANG_WORDS = [
    "python","c","c++","java","javascript","typescript","go","golang","rust",
    "kotlin","swift","r","matlab","sql","scala","php","ruby","dart"
]

FRAMEWORK_TOOL_WORDS = [
    "react","react native","vue","angular","svelte","django","flask","fastapi",
    "spring","spring boot","tensorflow","pytorch","sklearn","scikit-learn","keras",
    "node","node.js","express","next.js","nest","graphql","docker","kubernetes",
    "terraform","aws","gcp","azure","hadoop","spark","hive","airflow","redis",
    "rabbitmq","mysql","postgres","mongodb","sqlite","firebase"
]

LANGUAGE_WORDS = [
    "korean","english","japanese","chinese","spanish","german","french","italian",
    "russian","portuguese","arabic","hindi"
]

# ---------------------------
# 2) PDF → 이미지 (PIL.Image)
# ---------------------------
def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

# ---------------------------
# 3) 간단 이미지 메트릭 (품질 점검용)
# ---------------------------
def image_metrics(images: List[Image.Image]) -> Dict[str, Any]:
    metrics = {}
    for idx, img in enumerate(images, start=1):
        arr = np.array(img.convert("L"))
        thresh = arr.mean() - arr.std() * 0.2
        text_ratio = (arr < thresh).mean()
        metrics[f"page_{idx}"] = {
            "width": int(img.width),
            "height": int(img.height),
            "mean_gray": float(arr.mean()),
            "std_gray": float(arr.std()),
            "text_pixel_ratio": float(text_ratio),
        }
    return metrics

# ---------------------------
# 4) OCR (EasyOCR)
# ---------------------------
def ocr_images(images: List[Image.Image], languages: List[str]) -> str:
    reader = easyocr.Reader(languages)  # ['ko','en'] 추천
    all_texts: List[str] = []
    for img in images:
        # detail=0 → 텍스트만, 좌표 불필요
        texts = reader.readtext(np.array(img), detail=0)
        page_text = "\n".join(t.strip() for t in texts if t.strip())
        all_texts.append(page_text)
    return "\n".join(all_texts)

# ---------------------------
# 5) 정규식 보조
# ---------------------------
def _match_first(regexes: List[re.Pattern], text: str) -> Optional[str]:
    for rgx in regexes:
        m = rgx.search(text)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None

# ---------------------------
# 6) 텍스트 → 스펙 라벨 파싱
# ---------------------------
def parse_text_to_features(text: str) -> Dict[str, Any]:
    data = {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in RESUME_LABELS.items()}
    norm = text.replace("\r", "\n")
    lines = [ln.strip() for ln in norm.split("\n") if ln.strip()]
    lower = norm.lower()

    # 이름 후보: 최상단 라인
    if lines:
        top = lines[0]
        if len(top) <= 50 and not re.search(r"@|github|linkedin|http", top, re.I):
            data["name"] = top

    # 이메일/전화/링크
    email = _match_first([re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")], norm)
    phone = _match_first([
        re.compile(r"(\+82[-\s]?\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4})"),
        re.compile(r"(\d{2,3}[-\s]?\d{3,4}[-\s]?\d{4})"),
    ], norm)
    links = re.findall(r"(https?://[^\s)]+)", norm)
    data["email"] = email
    data["phone"] = phone
    data["links"] = links

    # 학위/대학/전공/졸업년도/학점
    degree = _match_first([
        re.compile(r"(Ph\.?D\.?)", re.I),
        re.compile(r"(M\.?S\.?|Master[’'s]?)", re.I),
        re.compile(r"(B\.?S\.?|Bachelor[’'s]?)", re.I),
        re.compile(r"(학사|석사|박사)", re.I),
    ], norm)
    data["highest_degree"] = degree

    uni = _match_first([
        re.compile(r"([A-Z][A-Za-z&.\s]{2,}University)"),
        re.compile(r"([가-힣A-Za-z&.\s]{2,}(대학교|대학원))"),
    ], norm)
    data["university"] = uni

    major = _match_first([
        re.compile(r"(Computer Science|Software Engineering|Electrical Engineering|Information Technology)", re.I),
        re.compile(r"(\b[A-Za-z가-힣/&\s]{2,20}(과|학과|전공)\b)"),
    ], norm)
    data["major"] = major

    grad_year = _match_first([re.compile(r"(20\d{2}|19\d{2})\s*(?:년)?\s*(?:졸업|Graduation|Grad)", re.I)], norm)
    data["grad_year"] = grad_year

    gpa = _match_first([
        re.compile(r"GPA\s*[:=]?\s*([0-4]\.\d{1,2})\s*/\s*([0-4]\.?\d*)", re.I),
        re.compile(r"평점\s*[:=]?\s*([0-4]\.\d{1,2})\s*/\s*([0-4]\.?\d*)"),
    ], norm)
    if gpa: data["gpa"] = gpa

    # 영어 점수
    toeic = _match_first([re.compile(r"TOEIC\s*[:=]?\s*(\d{3,4})", re.I)], norm)
    ielts = _match_first([re.compile(r"IELTS\s*[:=]?\s*(\d(?:\.\d)?)", re.I)], norm)
    toefl = _match_first([re.compile(r"TOEFL\s*[:=]?\s*(\d{2,3})", re.I)], norm)
    if toeic: data["english_scores"]["TOEIC"] = int(toeic)
    if ielts: data["english_scores"]["IELTS"] = float(ielts)
    if toefl: data["english_scores"]["TOEFL"] = int(toefl)

    # 구사 언어
    langs = sorted({w for w in LANGUAGE_WORDS if re.search(rf"\b{re.escape(w)}\b", lower)})
    if langs: data["languages"] = langs

    # 언어/프레임워크 키워드
    prog = sorted({w for w in PROGRAMMING_LANG_WORDS if re.search(rf"\b{re.escape(w)}\b", lower)})
    tools = sorted({w for w in FRAMEWORK_TOOL_WORDS if re.search(rf"\b{re.escape(w)}\b", lower)})
    data["programming_langs"] = prog
    data["frameworks_tools"] = tools

    # 자격/수상
    certs = re.findall(r"(자격증|Certificate|Certification|자격)\s*[:\-]?\s*([^\n]+)", norm, re.I)
    awards = re.findall(r"(수상|Award|Prize)\s*[:\-]?\s*([^\n]+)", norm, re.I)
    data["certifications"] = [c[1].strip() for c in certs]
    data["awards"] = [a[1].strip() for a in awards]

    # 프로젝트 개수
    projects_count = len(re.findall(r"\b(Project|프로젝트)\b", norm, re.I))
    data["projects_count"] = projects_count

    # 경력 연차(기간 문자열 기반 근사)
    periods = re.findall(r"(20\d{2})\s*[./-]\s*(\d{1,2})?\s*[-~–]\s*(20\d{2}|Present|현재)\s*[./-]?\s*(\d{1,2})?", norm, re.I)
    total_months = 0
    for y1, m1, y2, m2 in periods:
        y1 = int(y1)
        m1 = int(m1) if (m1 and m1.isdigit()) else 1
        if isinstance(y2, str) and y2.lower() in {"present","현재"}:
            y2 = 2025
            m2 = 11
        else:
            y2 = int(y2)
            m2 = int(m2) if (m2 and str(m2).isdigit()) else 1
        total_months += (y2 - y1) * 12 + (m2 - m1)
    if total_months > 0:
        data["experience_years"] = round(total_months / 12, 1)

    # 요약 키워드(간단 TF)
    tokens = re.findall(r"[A-Za-z가-힣+#./-]{2,}", lower)
    stop = set(["and","the","for","with","of","in","to","a","an","on","at","by","from"])
    counts = {}
    for t in tokens:
        if t in stop:
            continue
        counts[t] = counts.get(t, 0) + 1
    topk = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:15]
    data["summary_keywords"] = [w for w, _ in topk]

    return data

# ---------------------------
# 7) 단일 PDF 처리
# ---------------------------
def process_resume(pdf_path: str, languages: List[str], dpi: int = 200) -> Dict[str, Any]:
    try:
        images = pdf_to_images(pdf_path, dpi=dpi)
        metrics = image_metrics(images)
        text = ocr_images(images, languages=languages)

        # 텍스트 레이어가 있는 PDF라면 OCR 결과가 짧을 수 있으니 보강
        if len(text.strip()) < 20:
            # PyMuPDF의 텍스트 레이어도 같이 시도
            doc = fitz.open(pdf_path)
            raw = []
            for p in doc:
                raw.append(p.get_text("text"))
            doc.close()
            text = (text + "\n" + "\n".join(raw)).strip()

        features = parse_text_to_features(text)
        features["image_metrics"] = metrics
        features["source_pdf"] = os.path.basename(pdf_path)
        return features
    except Exception as e:
        row = {k: None for k in RESUME_LABELS.keys()}
        row["source_pdf"] = os.path.basename(pdf_path)
        row["error"] = str(e)
        return row

# ---------------------------
# 8) 배치 처리 (파일 or 폴더)
# ---------------------------
def batch_build_dataset(input_path: str, languages: List[str], dpi: int = 200) -> pd.DataFrame:
    paths: List[str] = []
    if os.path.isdir(input_path):
        for name in os.listdir(input_path):
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(input_path, name))
    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        paths = [input_path]
    else:
        raise FileNotFoundError("PDF 파일 또는 PDF가 있는 폴더를 입력하세요.")

    rows = []
    for p in sorted(paths):
        rows.append(process_resume(p, languages=languages, dpi=dpi))
    return pd.DataFrame(rows)

# ---------------------------
# 9) CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Resume OCR → Dataset")
    ap.add_argument("--input", required=True, help="PDF 파일 경로 또는 폴더 경로")
    ap.add_argument("--out", default="resume_dataset.csv", help="저장할 CSV 경로")
    ap.add_argument("--dpi", type=int, default=200, help="PDF 렌더링 DPI(기본 200)")
    ap.add_argument("--langs", nargs="+", default=["ko","en"], help="EasyOCR 언어 리스트")
    args = ap.parse_args()

    df = batch_build_dataset(args.input, languages=args.langs, dpi=args.dpi)
    df.to_csv(args.out, index=False)
    print(f"[OK] {len(df)}행 저장: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()

