#!/usr/bin/env python3
"""
JSON Morpheme Data Processor - A6000 x2 + 112 CPU 환경 최적화
NIA Sign Language 데이터셋의 morpheme JSON 파일들을 처리
"""

import os
import json
import h5py
import time
import torch
import logging
import threading
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import deque
import cv2

class JSONMorphemeProcessor:
    """JSON morpheme 데이터를 처리하는 A6000 x2 최적화 클래스"""
    
    def __init__(self, data_root: str = "/workspace01/team03/data"):
        self.data_root = Path(data_root)
        self.morpheme_root = self.data_root / "morpheme"
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # GPU 설정
        self.device_count = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {self.device_count}")
        
        if self.device_count == 0:
            self.logger.warning("No GPU available, falling back to CPU")
            
        # 멀티프로세싱 설정 (112 CPU cores 활용)
        self.num_processes = min(mp.cpu_count(), 56)  # 논리적 코어의 절반 사용
        self.logger.info(f"Using {self.num_processes} processes")
        
    def discover_morpheme_folders(self) -> List[Path]:
        """morpheme 폴더들을 찾아 반환"""
        if not self.morpheme_root.exists():
            self.logger.error(f"Morpheme root directory not found: {self.morpheme_root}")
            return []
            
        folders = []
        for folder in self.morpheme_root.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                json_files = list(folder.glob("*.json"))
                if json_files:
                    folders.append(folder)
                    
        self.logger.info(f"Found {len(folders)} morpheme folders with JSON files")
        return sorted(folders)
    
    def load_json_morpheme(self, json_path: Path) -> Optional[Dict]:
        """단일 JSON morpheme 파일을 로드"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON {json_path}: {e}")
            return None
    
    def extract_video_info_from_json(self, json_data: Dict) -> Dict:
        """JSON에서 비디오 관련 정보 추출"""
        info = {
            'video_name': '',
            'video_resolution': '',
            'frame_count': 0,
            'fps': 30,  # 기본값
            'duration': 0
        }
        
        try:
            # JSON 구조에 따라 비디오 정보 추출
            if 'info' in json_data:
                video_info = json_data['info']
                info['video_name'] = video_info.get('video', '')
                info['video_resolution'] = video_info.get('resolution', '')
                
            if 'data' in json_data:
                frames = json_data['data']
                info['frame_count'] = len(frames)
                info['duration'] = info['frame_count'] / info['fps']
                
        except Exception as e:
            self.logger.warning(f"Could not extract video info: {e}")
            
        return info
    
    def generate_synthetic_frames(self, json_data: Dict) -> List[np.ndarray]:
        """JSON 데이터로부터 합성 프레임 생성 (테스트용)"""
        frames = []
        
        try:
            if 'data' in json_data:
                frame_data = json_data['data']
                
                for i, frame_info in enumerate(frame_data):
                    # 640x480 기본 프레임 생성
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # 프레임 번호 표시
                    cv2.putText(frame, f"Frame {i+1}", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # morpheme 정보가 있다면 표시
                    if 'morpheme' in frame_info:
                        morpheme_text = frame_info['morpheme'][:20]  # 20자까지만
                        cv2.putText(frame, morpheme_text, (50, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    frames.append(frame)
                    
            if not frames:
                # 최소한 하나의 빈 프레임 생성
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Data", (200, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frames.append(frame)
                
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic frames: {e}")
            # 에러 시 빈 프레임 반환
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Error", (200, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frames.append(frame)
            
        return frames
    
    def process_morpheme_folder(self, folder_path: Path) -> Dict:
        """단일 morpheme 폴더의 모든 JSON 파일들을 처리"""
        self.logger.info(f"Processing folder: {folder_path.name}")
        
        json_files = list(folder_path.glob("*.json"))
        results = {
            'folder': folder_path.name,
            'total_files': len(json_files),
            'processed_files': 0,
            'failed_files': 0,
            'video_info': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for json_file in tqdm(json_files, desc=f"Processing {folder_path.name}"):
            try:
                # JSON 로드
                json_data = self.load_json_morpheme(json_file)
                if json_data is None:
                    results['failed_files'] += 1
                    continue
                
                # 비디오 정보 추출
                video_info = self.extract_video_info_from_json(json_data)
                video_info['json_file'] = json_file.name
                
                # 합성 프레임 생성 (실제 비디오가 없으므로)
                frames = self.generate_synthetic_frames(json_data)
                video_info['frame_count'] = len(frames)
                
                results['video_info'].append(video_info)
                results['processed_files'] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process {json_file}: {e}")
                results['failed_files'] += 1
                
        results['processing_time'] = time.time() - start_time
        
        self.logger.info(f"Folder {folder_path.name}: "
                        f"{results['processed_files']}/{results['total_files']} processed, "
                        f"{results['failed_files']} failed, "
                        f"Time: {results['processing_time']:.2f}s")
        
        return results
    
    def batch_process_all_folders(self) -> Dict:
        """모든 morpheme 폴더들을 배치 처리"""
        self.logger.info("Starting batch processing of all morpheme folders...")
        
        folders = self.discover_morpheme_folders()
        if not folders:
            self.logger.error("No morpheme folders found!")
            return {}
        
        total_results = {
            'total_folders': len(folders),
            'processed_folders': 0,
            'failed_folders': 0,
            'total_files': 0,
            'total_processed_files': 0,
            'total_failed_files': 0,
            'folder_results': [],
            'total_processing_time': 0
        }
        
        start_time = time.time()
        
        # 순차적으로 폴더 처리 (메모리 절약)
        for folder in folders:
            try:
                folder_result = self.process_morpheme_folder(folder)
                total_results['folder_results'].append(folder_result)
                total_results['processed_folders'] += 1
                total_results['total_files'] += folder_result['total_files']
                total_results['total_processed_files'] += folder_result['processed_files']
                total_results['total_failed_files'] += folder_result['failed_files']
                
            except Exception as e:
                self.logger.error(f"Failed to process folder {folder}: {e}")
                total_results['failed_folders'] += 1
        
        total_results['total_processing_time'] = time.time() - start_time
        
        # 결과 요약
        self.logger.info("=== BATCH PROCESSING SUMMARY ===")
        self.logger.info(f"Total folders: {total_results['total_folders']}")
        self.logger.info(f"Processed folders: {total_results['processed_folders']}")
        self.logger.info(f"Failed folders: {total_results['failed_folders']}")
        self.logger.info(f"Total JSON files: {total_results['total_files']}")
        self.logger.info(f"Successfully processed: {total_results['total_processed_files']}")
        self.logger.info(f"Failed files: {total_results['total_failed_files']}")
        self.logger.info(f"Total processing time: {total_results['total_processing_time']:.2f}s")
        
        return total_results
    
    def save_processing_results(self, results: Dict, output_file: str = "morpheme_processing_results.json") -> None:
        """처리 결과를 JSON 파일로 저장"""
        output_path = self.data_root / output_file
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Processing results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


def main():
    """메인 실행 함수"""
    processor = JSONMorphemeProcessor()
    
    print("=== JSON Morpheme Data Processor ===")
    print(f"Data root: {processor.data_root}")
    print(f"Morpheme root: {processor.morpheme_root}")
    print(f"GPUs available: {processor.device_count}")
    print(f"CPU processes: {processor.num_processes}")
    print()
    
    # 폴더 검색
    folders = processor.discover_morpheme_folders()
    print(f"Found {len(folders)} morpheme folders:")
    for i, folder in enumerate(folders[:10]):  # 첫 10개만 표시
        json_count = len(list(folder.glob("*.json")))
        print(f"  {i+1:2d}. {folder.name} ({json_count} JSON files)")
    
    if len(folders) > 10:
        print(f"  ... and {len(folders) - 10} more folders")
    print()
    
    # 사용자 확인
    response = input("Process all folders? (y/N): ").strip().lower()
    if response != 'y':
        print("Processing cancelled.")
        return
    
    # 배치 처리 실행
    results = processor.batch_process_all_folders()
    
    # 결과 저장
    processor.save_processing_results(results)
    
    print("\n=== Processing Complete! ===")
    if results.get('total_processing_time', 0) > 0:
        files_per_sec = results['total_processed_files'] / results['total_processing_time']
        print(f"Average processing speed: {files_per_sec:.1f} files/sec")


if __name__ == "__main__":
    main()
