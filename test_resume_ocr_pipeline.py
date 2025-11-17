# ======================= 복붙 시작: test_resume_ocr_pipeline.py =======================
import os
import pandas as pd

# 1️⃣ 더미 PDF 경로 (파일이 없어도 무방)
dummy_pdf_path = "sample_resume.pdf"

# 2️⃣ 더미 OCR 결과 (실제 OCR을 돌리지 않고 흉내만 냄)
dummy_data = [
    {
        "name": "Hong Gil-Dong",
        "email": "gildong@example.com",
        "phone": "010-1234-5678",
        "links": ["https://github.com/gildong"],
        "highest_degree": "Bachelor",
        "university": "Dankook University",
        "major": "Computer Science",
        "grad_year": "2024",
        "gpa": "3.92/4.3",
        "english_scores": {"TOEIC": 925},
        "languages": ["korean", "english"],
        "programming_langs": ["python", "c", "java"],
        "frameworks_tools": ["react", "django", "firebase"],
        "certifications": ["정보처리기사", "AWS Certified Developer"],
        "awards": ["2024 DKU Hackathon 1st Prize"],
        "projects_count": 5,
        "experience_years": 1.5,
        "summary_keywords": ["project", "dku", "app", "firebase", "ai", "resume"],
        "image_metrics": {"page_1": {"width": 1200, "height": 1600}},
        "source_pdf": dummy_pdf_path,
        "error": None,
    }
]

# 3️⃣ CSV로 저장
df = pd.DataFrame(dummy_data)
csv_path = "dummy_resume_dataset.csv"
df.to_csv(csv_path, index=False)

# 4️⃣ 결과 확인 출력
print(f"[테스트 완료] CSV 파일 생성됨: {os.path.abspath(csv_path)}")
print("\nCSV 내용 미리보기 ↓\n")
print(df.head(1).to_string(index=False))
# ======================= 복붙 끝 =======================
