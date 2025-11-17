import os
import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class RoadmapAI:
    def __init__(self, data_dir='roadmap_data'):
        """데이터 저장 디렉토리 및 AI 모델 초기화"""
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self.model_file = os.path.join(data_dir, 'duration_model.pkl')
        self.encoder_file = os.path.join(data_dir, 'encoders.pkl')
        self.training_stats_file = os.path.join(data_dir, 'training_stats.json')
        
        # 모델 및 인코더
        self.duration_model = None
        self.school_encoder = LabelEncoder()
        self.major_encoder = LabelEncoder()
        self.company_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.job_encoder = LabelEncoder()
        self.training_data = None
        
        # 모델 로드
        self._load_models()
    
    def calculate_estimated_duration(self, profile):
        """합격자 스펙 기반 소요 기간 계산"""
        base_months = 18
        
        # 학교 티어 반영
        school = str(profile.get('school', '')).strip()
        if school == 'College':
            base_months -= 3
        
        # GPA가 높을수록 빠름
        gpa = float(profile['gpa'])
        if gpa >= 4.0:
            base_months -= 4
        elif gpa >= 3.7:
            base_months -= 3
        elif gpa >= 3.5:
            base_months -= 2
        elif gpa >= 3.0:
            base_months -= 1
        else:
            base_months += 2  # 학점 낮으면 더 오래 걸림
        
        # 인턴 경험 많을수록 빠름 (개당 2개월 단축)
        intern_count = int(profile['intern_count'])
        base_months -= intern_count * 2
        
        # 프로젝트 많을수록 빠름 (개당 1개월 단축)
        project_count = int(profile['project_count'])
        base_months -= project_count * 1
        
        # 자격증 많을수록 빠름 (개당 1개월 단축)
        cert_count = int(profile['cert_count'])
        base_months -= cert_count * 1
        
        # 어학 자격증 있으면 1개월 단축
        if int(profile.get('has_language_cert', 0)) == 1:
            base_months -= 1
        
        # 수상 경력에 따라 조정
        award_level = str(profile.get('award_level', '없음'))
        if '전국' in award_level or '국제' in award_level:
            base_months -= 3
        elif '지역' in award_level:
            base_months -= 2
        elif '교내' in award_level or '학교' in award_level:
            base_months -= 1
        
        # 해외 경험 있으면 1개월 단축
        overseas = str(profile.get('overseas_experience', '없음'))
        if overseas != '없음' and overseas.lower() != 'nan':
            base_months -= 1
        
        # 최소 1개월, 최대 48개월로 제한
        return max(1, min(base_months, 48))
    
    def load_csv_data(self, csv_file_path='AI/training_data_passSpec.csv'):
        """
        CSV 파일에서 합격자 데이터 로드
        
        CSV 컬럼:
        school, major, gpa, intern_company_scale, intern_total_months, intern_count,
        award_level, project_count, cert_count, has_language_cert, overseas_experience,
        company_size, industry, job_role, label, label_text
        """
        print(f"\n{'='*60}")
        print("CSV 데이터 로드 중...")
        print(f"{'='*60}")
        
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"✓ 총 {len(df)}개의 데이터를 로드했습니다.\n")
        
        # 컬럼명 출력
        print("CSV 컬럼:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        print()
        
        # 합격자만 필터링 (label이 1인 경우)
        df_pass = df[df['label'] == 1].copy()
        print(f"✓ 합격자 데이터: {len(df_pass)}개\n")
        
        # 데이터 변환
        training_data = []
        
        for idx, row in df_pass.iterrows():
            # ✅ 수정: 규칙 기반으로 소요 기간 계산 (school 포함)
            profile_for_duration = {
                'school': str(row['school']) if pd.notna(row['school']) else "기타",
                'gpa': float(row['gpa']) if pd.notna(row['gpa']) else 0.0,
                'intern_count': int(row['intern_count']) if pd.notna(row['intern_count']) else 0,
                'project_count': int(row['project_count']) if pd.notna(row['project_count']) else 0,
                'cert_count': int(row['cert_count']) if pd.notna(row['cert_count']) else 0,
                'has_language_cert': int(row['has_language_cert']) if pd.notna(row['has_language_cert']) else 0,
                'award_level': str(row['award_level']) if pd.notna(row['award_level']) else '없음',
                'overseas_experience': str(row['overseas_experience']) if pd.notna(row['overseas_experience']) else '없음'
            }
            
            # 규칙 기반으로 소요 기간 계산
            duration = self.calculate_estimated_duration(profile_for_duration)
            
            data = {
                "school": str(row['school']) if pd.notna(row['school']) else "기타",
                "major": str(row['major']) if pd.notna(row['major']) else "기타",
                "gpa": float(row['gpa']) if pd.notna(row['gpa']) else 0.0,
                "intern_company_scale": str(row['intern_company_scale']) if pd.notna(row['intern_company_scale']) else "없음",
                "intern_count": int(row['intern_count']) if pd.notna(row['intern_count']) else 0,
                "award_level": str(row['award_level']) if pd.notna(row['award_level']) else "없음",
                "project_count": int(row['project_count']) if pd.notna(row['project_count']) else 0,
                "cert_count": int(row['cert_count']) if pd.notna(row['cert_count']) else 0,
                "has_language_cert": int(row['has_language_cert']) if pd.notna(row['has_language_cert']) else 0,
                "overseas_experience": str(row['overseas_experience']) if pd.notna(row['overseas_experience']) else "없음",
                "company_size": str(row['company_size']) if pd.notna(row['company_size']) else "중소기업",
                "industry": str(row['industry']) if pd.notna(row['industry']) else "기타",
                "job_role": str(row['job_role']) if pd.notna(row['job_role']) else "개발자",
                "duration_months": duration  # ✅ 수정: 규칙 기반 계산값 사용
            }
            
            training_data.append(data)
        
        self.training_data = training_data
        return training_data
    
    def train_model(self, training_data=None):
        """AI 모델 학습"""
        if training_data is None:
            training_data = self.training_data
        
        if training_data is None:
            raise Exception("학습 데이터가 없습니다. load_csv_data()를 먼저 실행하세요.")
        
        print(f"\n{'='*60}")
        print("AI 모델 학습 시작")
        print(f"{'='*60}\n")
        
        # 1. 인코더 학습
        schools = [d['school'] for d in training_data]
        majors = [d['major'] for d in training_data]
        companies = [d['company_size'] for d in training_data]
        industries = [d['industry'] for d in training_data]
        jobs = [d['job_role'] for d in training_data]
        
        self.school_encoder.fit(schools)
        self.major_encoder.fit(majors)
        self.company_encoder.fit(companies)
        self.industry_encoder.fit(industries)
        self.job_encoder.fit(jobs)
        
        print(f"✓ 학습 가능한 학교: {len(self.school_encoder.classes_)}개")
        print(f"✓ 학습 가능한 전공: {len(self.major_encoder.classes_)}개")
        print(f"✓ 학습 가능한 회사 규모: {len(self.company_encoder.classes_)}개")
        print(f"✓ 학습 가능한 산업: {len(self.industry_encoder.classes_)}개")
        print(f"✓ 학습 가능한 직무: {len(self.job_encoder.classes_)}개\n")
        
        # 2. 특징 추출
        X = []
        y = []
        
        for data in training_data:
            features = self._extract_features(data)
            X.append(features)
            y.append(data['duration_months'])
        
        X = np.array(X)
        y = np.array(y)
        
        # 3. 학습/테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 4. 모델 학습
        print("AI 학습 중...")
        self.duration_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.duration_model.fit(X_train, y_train)
        
        # 5. 성능 평가
        y_pred = self.duration_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print("학습 완료!")
        print(f"{'='*60}")
        print(f"✓ 학습 데이터: {len(training_data)}개")
        print(f"✓ 예측 오차: ±{mae:.2f}개월")
        print(f"✓ 예측 정확도: 약 {max(0, 100 - mae*10):.1f}%")
        
        # 6. 통계 저장
        self._save_training_stats(training_data)
        
        # 7. 모델 저장
        self._save_models()
        
        return mae
    
    def _extract_features(self, data):
        """데이터에서 AI 학습용 특징 추출"""
        features = [
            float(data['gpa']),
            int(data['intern_count']),
            int(data['project_count']),
            int(data['cert_count']),
            int(data['has_language_cert']),
            self.school_encoder.transform([data['school']])[0],
            self.major_encoder.transform([data['major']])[0],
            self.company_encoder.transform([data['company_size']])[0],
            self.industry_encoder.transform([data['industry']])[0],
            self.job_encoder.transform([data['job_role']])[0],
        ]
        return features
    
    def _save_training_stats(self, training_data):
        """학습 데이터 통계 저장"""
        stats = {
            'total_count': len(training_data),
            'avg_gpa': float(np.mean([float(d['gpa']) for d in training_data])),
            'avg_intern_count': float(np.mean([int(d['intern_count']) for d in training_data])),
            'avg_project_count': float(np.mean([int(d['project_count']) for d in training_data])),
            'avg_cert_count': float(np.mean([int(d['cert_count']) for d in training_data])),
            'avg_duration': float(np.mean([d['duration_months'] for d in training_data])),
            'schools': list(self.school_encoder.classes_),
            'majors': list(self.major_encoder.classes_),
            'company_sizes': list(self.company_encoder.classes_),
            'industries': list(self.industry_encoder.classes_),
            'job_roles': list(self.job_encoder.classes_)
        }
        
        with open(self.training_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def _load_training_stats(self):
        """학습 통계 로드"""
        try:
            with open(self.training_stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def _save_models(self):
        """모델 저장"""
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.duration_model, f)
        
        with open(self.encoder_file, 'wb') as f:
            pickle.dump({
                'school': self.school_encoder,
                'major': self.major_encoder,
                'company': self.company_encoder,
                'industry': self.industry_encoder,
                'job': self.job_encoder
            }, f)
        
        print(f"✓ 모델이 저장되었습니다.\n")
    
    def _load_models(self):
        """모델 로드"""
        try:
            with open(self.model_file, 'rb') as f:
                self.duration_model = pickle.load(f)
            
            with open(self.encoder_file, 'rb') as f:
                encoders = pickle.load(f)
                self.school_encoder = encoders['school']
                self.major_encoder = encoders['major']
                self.company_encoder = encoders['company']
                self.industry_encoder = encoders['industry']
                self.job_encoder = encoders['job']
            
            print("✓ 기존 학습된 AI 모델을 로드했습니다.\n")
        except FileNotFoundError:
            pass
    
    def analyze_user_gap(self, user_profile):
        """사용자와 합격자 평균 비교 분석"""
        stats = self._load_training_stats()
        
        if stats is None:
            raise Exception("학습 통계가 없습니다. train_model()을 먼저 실행하세요.")
        
        # 목표 회사/직무에 맞는 합격자 필터링
        if self.training_data:
            similar_cases = [
                d for d in self.training_data
                if d['company_size'] == user_profile['company_size']
                and d['job_role'] == user_profile['job_role']
            ]
            
            if similar_cases:
                target_stats = {
                    'avg_gpa': float(np.mean([float(d['gpa']) for d in similar_cases])),
                    'avg_intern_count': float(np.mean([int(d['intern_count']) for d in similar_cases])),
                    'avg_project_count': float(np.mean([int(d['project_count']) for d in similar_cases])),
                    'avg_cert_count': float(np.mean([int(d['cert_count']) for d in similar_cases])),
                    'count': len(similar_cases)
                }
            else:
                target_stats = stats
        else:
            target_stats = stats
        
        # 부족한 부분 분석
        gaps = {
            'gpa': {
                'current': float(user_profile['gpa']),
                'target': target_stats['avg_gpa'],
                'gap': target_stats['avg_gpa'] - float(user_profile['gpa'])
            },
            'intern_count': {
                'current': int(user_profile['intern_count']),
                'target': target_stats['avg_intern_count'],
                'gap': target_stats['avg_intern_count'] - int(user_profile['intern_count'])
            },
            'project_count': {
                'current': int(user_profile['project_count']),
                'target': target_stats['avg_project_count'],
                'gap': target_stats['avg_project_count'] - int(user_profile['project_count'])
            },
            'cert_count': {
                'current': int(user_profile['cert_count']),
                'target': target_stats['avg_cert_count'],
                'gap': target_stats['avg_cert_count'] - int(user_profile['cert_count'])
            }
        }
        
        # 개선이 필요한 항목
        weaknesses = []
        recommendations = []
        
        for key, value in gaps.items():
            if value['gap'] > 0.5:
                weakness_name = {
                    'gpa': '학점',
                    'intern_count': '인턴 경험',
                    'project_count': '프로젝트',
                    'cert_count': '자격증'
                }[key]
                
                weaknesses.append(weakness_name)
                
                if key == 'gpa':
                    recommendations.append(
                        f"학점: 현재 {value['current']:.2f} → 합격자 평균 {value['target']:.2f} (차이: {value['gap']:.2f})"
                    )
                else:
                    recommendations.append(
                        f"{weakness_name}: 현재 {int(value['current'])}개 → 합격자 평균 {int(value['target'])}개 (부족: {int(np.ceil(value['gap']))}개)"
                    )
        
        return {
            'gaps': gaps,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'similar_count': target_stats.get('count', stats['total_count'])
        }
    
    def predict_duration(self, user_profile):
        """AI 기반 소요 기간 예측"""
        if self.duration_model is None:
            raise Exception("모델이 학습되지 않았습니다.")
        
        features = np.array([self._extract_features(user_profile)])
        predicted_months = self.duration_model.predict(features)[0]
        predicted_months = max(0, int(predicted_months))
        
        years = predicted_months // 12
        months = predicted_months % 12
        
        return years, months, predicted_months
    
    def generate_detailed_roadmap(self, user_profile, total_months, gap_analysis):
        """AI 분석 기반 세부 로드맵 생성"""
        gaps = gap_analysis['gaps']
        weaknesses = gap_analysis['weaknesses']
        
        roadmap = []
        
        # AI 예측 기간에 맞춰서 로드맵 생성
        if total_months <= 6:
            roadmap = self._generate_short_roadmap(weaknesses, gaps, total_months)
        elif total_months <= 18:
            roadmap = self._generate_mid_roadmap(weaknesses, gaps, total_months)
        else:
            roadmap = self._generate_long_roadmap(weaknesses, gaps, total_months)
        
        return roadmap
    
    def _generate_short_roadmap(self, weaknesses, gaps, total_months):
        """단기 로드맵 (6개월 이하) - AI 예측 기간에 맞춤"""
        roadmap = []
        
        # 기간에 따라 동적으로 로드맵 생성
        if total_months == 1:
            # 1개월
            tasks = ["목표 회사 직무 요구사항 급속 분석"]
            
            if '프로젝트' in weaknesses:
                shortage = int(np.ceil(gaps['project_count']['gap']))
                tasks.append(f"즉시 완성 가능한 프로젝트 {shortage}개 집중 작업")
            
            if '자격증' in weaknesses:
                shortage = int(np.ceil(gaps['cert_count']['gap']))
                tasks.append(f"단기 취득 가능 자격증 {shortage}개 집중")
            
            tasks.extend([
                "이력서 및 포트폴리오 완성",
                "목표 회사 즉시 지원"
            ])
            
            roadmap.append({
                "period": "1개월차",
                "focus": "초단기 집중 스펙 보강",
                "tasks": tasks
            })
            
        elif total_months == 2:
            # 2개월
            tasks = ["목표 회사 직무 요구사항 급속 분석"]
            
            if '프로젝트' in weaknesses:
                shortage = int(np.ceil(gaps['project_count']['gap']))
                tasks.append(f"즉시 완성 가능한 프로젝트 {shortage}개 집중 작업")
            
            if '자격증' in weaknesses:
                shortage = int(np.ceil(gaps['cert_count']['gap']))
                tasks.append(f"단기 취득 가능 자격증 {shortage}개 집중")
            
            tasks.extend([
                "이력서 및 포트폴리오 완성",
                "목표 회사 즉시 지원"
            ])
            
            roadmap.append({
                "period": "1-2개월차",
                "focus": "초단기 집중 스펙 보강",
                "tasks": tasks
            })
            
        elif total_months == 3:
            # 3개월
            tasks_1 = ["목표 회사 직무 요구사항 분석"]
            tasks_2 = ["이력서 및 포트폴리오 최종 정리", "목표 회사 지원 및 면접"]
            
            if '인턴 경험' in weaknesses:
                shortage = int(np.ceil(gaps['intern_count']['gap']))
                tasks_1.append(f"단기 인턴십 {shortage}개 지원")
            
            if '프로젝트' in weaknesses:
                shortage = int(np.ceil(gaps['project_count']['gap']))
                tasks_1.append(f"프로젝트 {shortage}개 기획 및 시작")
                tasks_2.insert(0, f"프로젝트 {shortage}개 완성")
            
            if '자격증' in weaknesses:
                shortage = int(np.ceil(gaps['cert_count']['gap']))
                tasks_1.append(f"자격증 {shortage}개 준비")
            
            roadmap.append({
                "period": "1-2개월차",
                "focus": "핵심 스펙 보강",
                "tasks": tasks_1
            })
            roadmap.append({
                "period": "3개월차",
                "focus": "입사 준비",
                "tasks": tasks_2
            })
            
        elif total_months == 4:
            # 4개월
            tasks_1 = ["목표 회사 직무 요구사항 분석"]
            tasks_2 = ["이력서 및 포트폴리오 최종 정리", "목표 회사 지원 및 면접"]
            
            if '인턴 경험' in weaknesses:
                shortage = int(np.ceil(gaps['intern_count']['gap']))
                tasks_1.append(f"단기 인턴십 {shortage}개 지원")
            
            if '프로젝트' in weaknesses:
                shortage = int(np.ceil(gaps['project_count']['gap']))
                tasks_1.append(f"프로젝트 {shortage}개 기획 및 시작")
                tasks_2.insert(0, f"프로젝트 {shortage}개 완성")
            
            if '자격증' in weaknesses:
                shortage = int(np.ceil(gaps['cert_count']['gap']))
                tasks_1.append(f"자격증 {shortage}개 준비")
            
            roadmap.append({
                "period": "1-2개월차",
                "focus": "핵심 스펙 보강",
                "tasks": tasks_1
            })
            roadmap.append({
                "period": "3-4개월차",
                "focus": "입사 준비",
                "tasks": tasks_2
            })
            
        elif total_months == 5:
            # 5개월
            tasks_month_1_2 = ["목표 회사 직무 분석"]
            tasks_month_3_4 = []
            tasks_month_5 = ["이력서 및 포트폴리오 최종 정리"]
            
            if '인턴 경험' in weaknesses:
                shortage = int(np.ceil(gaps['intern_count']['gap']))
                tasks_month_1_2.append(f"단기 인턴십 {shortage}개 이상 지원 및 합격")
                tasks_month_3_4.append("인턴십 프로젝트 성과 정리")
            
            if '프로젝트' in weaknesses:
                shortage = int(np.ceil(gaps['project_count']['gap']))
                tasks_month_1_2.append(f"즉시 시작 가능한 프로젝트 {shortage}개 기획")
                tasks_month_3_4.append(f"프로젝트 {shortage}개 완성 및 GitHub 업로드")
            
            if '자격증' in weaknesses:
                shortage = int(np.ceil(gaps['cert_count']['gap']))
                tasks_month_3_4.append(f"필수 자격증 {shortage}개 취득")
            
            tasks_month_5.extend([
                "기술 면접 대비",
                "현직자 네트워킹",
                "목표 회사 지원 및 면접"
            ])
            
            roadmap.append({"period": "1-2개월차", "focus": "급속 스펙 보강", "tasks": tasks_month_1_2})
            roadmap.append({"period": "3-4개월차", "focus": "핵심 역량 완성", "tasks": tasks_month_3_4})
            roadmap.append({"period": "5개월차", "focus": "입사 지원 준비", "tasks": tasks_month_5})
            
        else:
            # 6개월
            tasks_month_1_2 = ["목표 회사 직무 분석"]
            tasks_month_3_4 = []
            tasks_month_5_6 = ["이력서 및 포트폴리오 최종 정리"]
            
            if '인턴 경험' in weaknesses:
                shortage = int(np.ceil(gaps['intern_count']['gap']))
                tasks_month_1_2.append(f"단기 인턴십 {shortage}개 이상 지원 및 합격")
                tasks_month_3_4.append("인턴십 프로젝트 성과 정리")
            
            if '프로젝트' in weaknesses:
                shortage = int(np.ceil(gaps['project_count']['gap']))
                tasks_month_1_2.append(f"즉시 시작 가능한 프로젝트 {shortage}개 기획")
                tasks_month_3_4.append(f"프로젝트 {shortage}개 완성 및 GitHub 업로드")
            
            if '자격증' in weaknesses:
                shortage = int(np.ceil(gaps['cert_count']['gap']))
                tasks_month_3_4.append(f"필수 자격증 {shortage}개 취득")
            
            tasks_month_5_6.extend([
                "기술 면접 대비",
                "현직자 네트워킹",
                "목표 회사 지원 및 면접"
            ])
            
            roadmap.append({"period": "1-2개월차", "focus": "급속 스펙 보강", "tasks": tasks_month_1_2})
            roadmap.append({"period": "3-4개월차", "focus": "핵심 역량 완성", "tasks": tasks_month_3_4})
            roadmap.append({"period": "5-6개월차", "focus": "입사 지원 준비", "tasks": tasks_month_5_6})
        
        return roadmap
    
    def _generate_mid_roadmap(self, weaknesses, gaps, total_months):
        """중기 로드맵 (6-18개월)"""
        roadmap = []
        
        tasks_q1 = ["목표 회사 기술 스택 완벽 학습", "CS 기초 다지기"]
        if '프로젝트' in weaknesses:
            shortage = int(np.ceil(gaps['project_count']['gap']))
            tasks_q1.append(f"토이 프로젝트 {min(shortage, 2)}개 시작")
        
        roadmap.append({"period": "1-3개월차", "focus": "기술 기반 구축", "tasks": tasks_q1})
        
        tasks_q2 = []
        if '인턴 경험' in weaknesses:
            shortage = int(np.ceil(gaps['intern_count']['gap']))
            tasks_q2.append(f"인턴십 {shortage}개 지원 및 합격")
        
        if '프로젝트' in weaknesses:
            shortage = int(np.ceil(gaps['project_count']['gap']))
            tasks_q2.append(f"중대형 프로젝트 {shortage}개 완성")
        
        tasks_q2.append("기술 블로그 운영 시작")
        roadmap.append({"period": "4-6개월차", "focus": "실전 경험 축적", "tasks": tasks_q2})
        
        tasks_q3 = ["오픈소스 컨트리뷰션", "기술 컨퍼런스 참여"]
        if '자격증' in weaknesses:
            shortage = int(np.ceil(gaps['cert_count']['gap']))
            tasks_q3.append(f"자격증 {shortage}개 취득")
        tasks_q3.append("현직자 네트워킹 강화")
        
        roadmap.append({"period": "7-9개월차", "focus": "전문성 강화", "tasks": tasks_q3})
        
        if total_months > 9:
            tasks_q4 = ["포트폴리오 완성도 극대화", "모의 면접 진행", "목표 회사 지원", "면접 준비 및 경험 축적"]
            roadmap.append({"period": f"10-{total_months}개월차", "focus": "입사 준비", "tasks": tasks_q4})
        
        return roadmap
    
    def _generate_long_roadmap(self, weaknesses, gaps, total_months):
        """장기 로드맵 (18개월 이상)"""
        years = total_months // 12
        roadmap = []
        
        tasks_y1 = ["핵심 기술 스택 마스터", "알고리즘 일일 학습"]
        
        if '프로젝트' in weaknesses:
            shortage = int(np.ceil(gaps['project_count']['gap']))
            tasks_y1.append(f"대형 프로젝트 {max(shortage, 3)}개 완성")
        else:
            tasks_y1.append("프로젝트 품질 극대화")
        
        if '자격증' in weaknesses:
            shortage = int(np.ceil(gaps['cert_count']['gap']))
            tasks_y1.append(f"자격증 {max(shortage, 2)}개 취득")
        
        tasks_y1.append("기술 블로그 주 1회 포스팅")
        roadmap.append({"period": "1년차", "focus": "탄탄한 기술 기반 구축", "tasks": tasks_y1})
        
        if years >= 2:
            tasks_y2 = []
            
            if '인턴 경험' in weaknesses:
                shortage = int(np.ceil(gaps['intern_count']['gap']))
                tasks_y2.append(f"장기 인턴십 {shortage}개 경험")
            else:
                tasks_y2.append("실무 프로젝트 참여")
            
            tasks_y2.extend([
                "오픈소스 활발한 기여",
                "기술 컨퍼런스 발표자로 참여",
                "목표 회사 현직자와 정기 교류",
                "포트폴리오 사이트 운영"
            ])
            roadmap.append({"period": "2년차", "focus": "실무 경험 및 네트워킹", "tasks": tasks_y2})
        
        if years >= 3:
            roadmap.append({
                "period": "3년차 이상",
                "focus": "전문가 도약",
                "tasks": [
                    "특정 기술 분야 전문성 확보",
                    "기술 아티클 작성 및 배포",
                    "사이드 프로젝트 실사용자 확보",
                    "업계 네트워킹 확대",
                    "목표 회사 리쿠르팅 이벤트 참여"
                ]
            })
        
        roadmap.append({
            "period": "최종 6개월",
            "focus": "입사 준비 완료",
            "tasks": [
                "포트폴리오 최종 검토",
                "모의 면접 집중 훈련 (주 2회)",
                "추천서 확보",
                "여러 회사 동시 지원",
                "목표 회사 최종 지원 및 면접"
            ]
        })
        
        return roadmap
    
    def create_full_roadmap(self, user_profile):
        """전체 파이프라인 실행"""
        if self.duration_model is None:
            raise Exception("AI 모델이 학습되지 않았습니다. train_model()을 먼저 실행하세요.")
        
        print(f"\n{'='*60}")
        print("AI 분석 시작")
        print(f"{'='*60}\n")
        
        # 1. 부족한 부분 분석
        print("[1단계] 합격자 평균과 비교 분석 중...")
        gap_analysis = self.analyze_user_gap(user_profile)
        print(f"✓ {gap_analysis['similar_count']}명의 유사 합격자 데이터 분석 완료\n")
        
        # 2. 소요 기간 예측
        print("[2단계] AI 기반 소요 기간 예측 중...")
        years, months, total_months = self.predict_duration(user_profile)
        print(f"✓ 예상 소요 기간: {years}년 {months}개월\n")
        
        # 3. 로드맵 생성
        print("[3단계] 맞춤형 로드맵 생성 중...")
        roadmap = self.generate_detailed_roadmap(user_profile, total_months, gap_analysis)
        print(f"✓ {len(roadmap)}단계 로드맵 생성 완료\n")
        
        return {
            'duration': {
                'years': years,
                'months': months,
                'total_months': total_months
            },
            'gap_analysis': gap_analysis,
            'roadmap': roadmap
        }


# 사용 예시
if __name__ == "__main__":
    # ============================================
    # 1단계: CSV 파일에서 합격자 데이터 로드 및 학습
    # ============================================
    
    ai = RoadmapAI()
    
    # CSV 파일 로드 (AI 폴더의 training_data_passSpec.csv)
    training_data = ai.load_csv_data('AI/training_data_passSpec.csv')
    
    # AI 학습
    ai.train_model(training_data)
    
    # ============================================
    # 2단계: 사용자 입력 받아서 로드맵 생성
    # ============================================
    
    # 사용자 프로필 예시 (CSV 형식에 맞춤)
    user_profile = {
        'school': 'Local',
        'major': 'Engineering',
        'gpa': 2.3,
        'intern_company_scale': 'Large',
        'intern_count': 1,
        'award_level': 'School',
        'project_count': 2,
        'cert_count': 1,
        'has_language_cert': 1,
        'overseas_experience': 'LanguageStudy',
        'company_size': 'Large',
        'industry': 'IT',
        'job_role': 'Developer'
    }
    
    try:
        # AI 분석 및 로드맵 생성
        result = ai.create_full_roadmap(user_profile)
        
        # ============================================
        # 결과 출력
        # ============================================
        
        print(f"{'='*60}")
        print("AI 로드맵 분석 결과")
        print(f"{'='*60}\n")
        
        print(f"목표: {user_profile['company_size']} - {user_profile['industry']} - {user_profile['job_role']}")
        print(f"예상 소요 기간: {result['duration']['years']}년 {result['duration']['months']}개월\n")
        
        # 합격자와 비교 분석
        print(f"{'='*60}")
        print(f"합격자 평균 대비 부족한 부분 ({result['gap_analysis']['similar_count']}명 분석)")
        print(f"{'='*60}\n")
        
        if result['gap_analysis']['recommendations']:
            for rec in result['gap_analysis']['recommendations']:
                print(f"• {rec}")
        else:
            print("✓ 모든 항목이 합격자 평균 이상입니다!")
        
        print(f"\n{'='*60}")
        print("AI 맞춤형 로드맵")
        print(f"{'='*60}\n")
        
        for i, phase in enumerate(result['roadmap'], 1):
            print(f"[{phase['period']}]")
            print(f"핵심 목표: {phase['focus']}\n")
            print("세부 과제:")
            for j, task in enumerate(phase['tasks'], 1):
                print(f"  {j}. {task}")
            
            if i < len(result['roadmap']):
                print(f"\n{'-'*60}\n")
        
        print(f"\n※ 이 로드맵은 합격자 데이터를 학습한 AI가 생성했습니다.")
        
    except Exception as e:
        print(f"\n⚠ 오류: {e}\n")
        print("=" * 60)
        print("사용 방법")
        print("=" * 60)
        print("\n1단계: CSV 파일 준비")
        print("  - 파일명: AI/training_data_passSpec.csv")
        print("  - AI 폴더에 배치")
        print("\n2단계: 학습")
        print("  ai = RoadmapAI()")
        print("  data = ai.load_csv_data('AI/training_data_passSpec.csv')")
        print("  ai.train_model(data)")
        print("\n3단계: 로드맵 생성")
        print("  user_profile = {")
        print("      'school': '서울대학교',")
        print("      'major': '컴퓨터공학',")
        print("      'gpa': 3.8,")
        print("      'intern_company_scale': '대기업',")
        print("      'intern_count': 1,")
        print("      'award_level': '전국',")
        print("      'project_count': 2,")
        print("      'cert_count': 1,")
        print("      'has_language_cert': 1,")
        print("      'overseas_experience': '있음',")
        print("      'company_size': '대기업',")
        print("      'industry': 'IT',")
        print("      'job_role': '백엔드 개발자'")
        print("  }")
        print("  result = ai.create_full_roadmap(user_profile)")
        print("\n" + "=" * 60)