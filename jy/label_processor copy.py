#!/usr/bin/env python3
"""
수화 데이터셋 라벨 처리 및 JSON 저장 도구

사용자 요구사항:
1. SEN 번호 기반으로 gloss(단어)와 sentence(문장) 구분 저장
2. 시간 정보는 30fps 기준으로 프레임 번호 변환 (round 사용)
3. 키포인트 x,y 좌표는 8배 스케일링 후 정수로 변환 (round 사용)
4. 같은 SEN 번호는 같은 의미(gloss) 포함
5. JSON 형식으로 저장
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageLabelProcessor:
    """수화 라벨 데이터 처리 클래스"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.fps = 30  # 30fps 기준
        self.keypoint_scale = 8  # 키포인트 x,y 좌표 8배 스케일링
        
    def parse_sen_number(self, filename: str) -> Optional[int]:
        """파일명에서 SEN 번호 추출
        
        예: NIA_SL_SEN1410_REAL03_F_morpheme.json -> 1410
        """
        try:
            parts = filename.split('_SEN')
            if len(parts) > 1:
                sen_part = parts[1].split('_')[0]
                return int(sen_part)
        except (IndexError, ValueError) as e:
            logger.warning(f"SEN 번호 추출 실패: {filename} - {e}")
        return None
    
    def process_label_file(self, label_path: Path) -> Optional[Dict[str, Any]]:
        """개별 라벨 파일 처리"""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # SEN 번호 추출
            sen_id = self.parse_sen_number(label_path.name)
            if sen_id is None:
                logger.warning(f"SEN ID 추출 불가: {label_path}")
                return None
            
            # 메타데이터
            metadata = data.get('metaData', {})
            
            # 글로스(단어) 정보 추출
            glosses = []
            for item in data.get('data', []):
                start_time = item.get('start', 0.0)
                end_time = item.get('end', 0.0)
                
                # 시간을 30fps 기준 프레임 번호로 변환 (round 사용)
                start_frame = round(start_time * self.fps)
                end_frame = round(end_time * self.fps)
                
                # 속성(단어) 정보
                for attr in item.get('attributes', []):
                    gloss_name = attr.get('name', '')
                    if gloss_name:
                        glosses.append({
                            'gloss': gloss_name,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'start_time': start_time,
                            'end_time': end_time
                        })
            
            return {
                'sen_id': sen_id,
                'metadata': {
                    'video_name': metadata.get('name', ''),
                    'duration': metadata.get('duration', 0.0),
                    'url': metadata.get('url', ''),
                    'exported_on': metadata.get('exportedOn', '')
                },
                'glosses': glosses
            }
            
        except Exception as e:
            logger.error(f"라벨 파일 처리 실패: {label_path} - {e}")
            return None
    
    def collect_sen_data(self, direction: str = 'F') -> Dict[int, Dict[str, Any]]:
        """특정 방향의 모든 SEN 데이터 수집
        
        Args:
            direction: 영상 방향 (F, D, L, R, U)
        
        Returns:
            {sen_id: processed_data} 형태의 딕셔너리
        """
        sen_data = {}
        
        # 모든 라벨 폴더 검색 (01-16)
        for folder_num in range(1, 17):
            label_dir = self.data_root / f"[라벨]{folder_num:02d}_real_sen_morpheme" / "morpheme"
            
            if not label_dir.exists():
                continue
                
            # 각 하위 폴더 (01-16) 검색
            for sub_folder in range(1, 17):
                sub_dir = label_dir / f"{sub_folder:02d}"
                if not sub_dir.exists():
                    continue
                
                # 해당 방향의 morpheme 파일들 검색
                pattern = f"*_REAL{sub_folder:02d}_{direction}_morpheme.json"
                for label_file in sub_dir.glob(pattern):
                    processed = self.process_label_file(label_file)
                    if processed:
                        sen_id = processed['sen_id']
                        sen_data[sen_id] = processed
                        logger.debug(f"처리 완료: SEN{sen_id:04d} from {label_file}")
        
        logger.info(f"총 {len(sen_data)}개의 SEN {direction} 방향 데이터 수집 완료")
        return sen_data
    
    def create_gloss_mapping(self, sen_data: Dict[int, Dict[str, Any]]) -> Dict[str, List[int]]:
        """글로스(단어)별 SEN ID 매핑 생성"""
        gloss_mapping = {}
        
        for sen_id, data in sen_data.items():
            for gloss_info in data['glosses']:
                gloss = gloss_info['gloss']
                if gloss not in gloss_mapping:
                    gloss_mapping[gloss] = []
                if sen_id not in gloss_mapping[gloss]:
                    gloss_mapping[gloss].append(sen_id)
        
        # SEN ID 정렬
        for gloss in gloss_mapping:
            gloss_mapping[gloss].sort()
        
        logger.info(f"총 {len(gloss_mapping)}개의 고유 글로스 발견")
        return gloss_mapping
    
    def create_sentence_mapping(self, sen_data: Dict[int, Dict[str, Any]]) -> Dict[int, List[str]]:
        """SEN ID별 문장(글로스 시퀀스) 매핑 생성"""
        sentence_mapping = {}
        
        for sen_id, data in sen_data.items():
            # 시간 순서대로 글로스 정렬
            sorted_glosses = sorted(data['glosses'], key=lambda x: x['start_time'])
            gloss_sequence = [g['gloss'] for g in sorted_glosses]
            sentence_mapping[sen_id] = gloss_sequence
        
        return sentence_mapping
    
    def save_processed_data(self, output_dir: str, direction: str = 'F'):
        """처리된 데이터를 JSON 파일들로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 모든 SEN 데이터 수집
        logger.info(f"{direction} 방향 데이터 수집 시작...")
        sen_data = self.collect_sen_data(direction)
        
        # 2. 글로스 매핑 생성
        gloss_mapping = self.create_gloss_mapping(sen_data)
        
        # 3. 문장 매핑 생성
        sentence_mapping = self.create_sentence_mapping(sen_data)
        
        # 4. 통합 데이터 구조 생성
        integrated_data = {
            'direction': direction,
            'fps': self.fps,
            'keypoint_scale': self.keypoint_scale,
            'timestamp': str(Path().absolute()),
            'statistics': {
                'total_sen_count': len(sen_data),
                'total_gloss_count': len(gloss_mapping),
                'unique_glosses': list(gloss_mapping.keys())
            },
            'gloss_to_sen_mapping': gloss_mapping,
            'sen_to_sentence_mapping': sentence_mapping,
            'sen_data': sen_data
        }
        
        # 5. 파일별로 저장
        files_to_save = [
            (f'sen_data_{direction}_complete.json', integrated_data),
            (f'gloss_mapping_{direction}.json', {
                'direction': direction,
                'gloss_to_sen': gloss_mapping,
                'statistics': {
                    'total_glosses': len(gloss_mapping),
                    'total_sens': len(sen_data)
                }
            }),
            (f'sentence_mapping_{direction}.json', {
                'direction': direction,
                'sen_to_sentence': sentence_mapping,
                'fps': self.fps
            }),
            (f'sen_details_{direction}.json', {
                'direction': direction,
                'sen_data': sen_data,
                'fps': self.fps,
                'keypoint_scale': self.keypoint_scale
            })
        ]
        
        for filename, data in files_to_save:
            file_path = output_path / filename
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"저장 완료: {file_path}")
            except Exception as e:
                logger.error(f"파일 저장 실패: {file_path} - {e}")
        
        # 6. 요약 리포트 생성
        self._generate_summary_report(output_path, direction, integrated_data)
    
    def _generate_summary_report(self, output_path: Path, direction: str, data: Dict[str, Any]):
        """요약 리포트 생성"""
        report = {
            'direction': direction,
            'summary': {
                'total_sen_videos': data['statistics']['total_sen_count'],
                'unique_glosses': data['statistics']['total_gloss_count'],
                'fps': f"시간 × {self.fps} = 프레임 번호",
                'keypoint_scaling': f"키포인트 x,y × {self.keypoint_scale} = 정수 좌표"
            },
            'top_glosses': {},
            'sample_sentences': {}
        }
        
        # 가장 많이 사용된 글로스 TOP 10
        gloss_counts = {gloss: len(sen_list) for gloss, sen_list in data['gloss_to_sen_mapping'].items()}
        top_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        report['top_glosses'] = {gloss: {'count': count, 'sen_ids': data['gloss_to_sen_mapping'][gloss][:5]} 
                                for gloss, count in top_glosses}
        
        # 샘플 문장 5개
        sample_sens = list(data['sen_to_sentence_mapping'].keys())[:5]
        for sen_id in sample_sens:
            report['sample_sentences'][f'SEN{sen_id:04d}'] = {
                'sentence': ' + '.join(data['sen_to_sentence_mapping'][sen_id]),
                'gloss_count': len(data['sen_to_sentence_mapping'][sen_id]),
                'details': data['sen_data'][sen_id]['glosses']
            }
        
        # 리포트 저장
        report_path = output_path / f'summary_report_{direction}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"요약 리포트 저장: {report_path}")


def main():
    """메인 실행 함수"""
    # 데이터 경로 설정
    data_root = "/home/ty/rtmw/02/mmpose/jy/data/1.Training"
    output_dir = "/home/ty/rtmw/02/mmpose/jy/processed_labels"
    
    # 처리기 초기화
    processor = SignLanguageLabelProcessor(data_root)
    
    # F 방향 데이터 처리 (videos/03과 매칭)
    logger.info("=== F 방향 수화 라벨 데이터 처리 시작 ===")
    processor.save_processed_data(output_dir, direction='F')
    
    logger.info("=== 처리 완료 ===")
    print(f"\n결과 파일들이 저장되었습니다: {output_dir}")
    print("\n생성된 파일들:")
    print("- sen_data_F_complete.json: 전체 통합 데이터")
    print("- gloss_mapping_F.json: 글로스 → SEN ID 매핑")
    print("- sentence_mapping_F.json: SEN ID → 문장 매핑")
    print("- sen_details_F.json: SEN 상세 데이터")
    print("- summary_report_F.json: 요약 리포트")


if __name__ == "__main__":
    main()
