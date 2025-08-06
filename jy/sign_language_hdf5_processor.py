#!/usr/bin/env python3
"""
F 방향 수화 영상 HDF5 배치 처리기
크롭 이미지, 포즈, 라벨을 분리된 HDF5 파일로 저장
"""

import os
import h5py
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
import glob
from datetime import datetime
import logging

from video_processor_yolo11l import VideoProcessorYOLO11L

class SignLanguageHDF5BatchProcessor:
    """F 방향 수화 영상 HDF5 배치 처리기"""
    
    def __init__(self, 
                 data_root: str,
                 rtmw_config: str,
                 rtmw_checkpoint: str,
                 output_root: str = "sign_language_dataset",
                 videos_per_batch: int = 250):
        """
        Args:
            data_root: 데이터 루트 경로 (data/)
            rtmw_config: RTMW 설정 파일 경로
            rtmw_checkpoint: RTMW 체크포인트 경로
            output_root: 출력 루트 경로
            videos_per_batch: 배치당 영상 수 (기본 250개)
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.videos_per_batch = videos_per_batch
        
        # 출력 디렉토리 생성
        self.frames_dir = self.output_root / "frames"
        self.poses_dir = self.output_root / "poses" 
        self.labels_dir = self.output_root / "labels"
        self.metadata_dir = self.output_root / "metadata"
        
        for dir_path in [self.frames_dir, self.poses_dir, self.labels_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 비디오 처리기 초기화
        self.video_processor = VideoProcessorYOLO11L(
            rtmw_config=rtmw_config,
            rtmw_checkpoint=rtmw_checkpoint,
            output_dir="temp_processing"
        )
        
        # 로깅 설정
        self._setup_logging()
        
        # F 방향 영상 목록 수집
        self.f_videos = self._collect_f_videos()
        self.total_batches = (len(self.f_videos) + self.videos_per_batch - 1) // self.videos_per_batch
        
        print(f"🎬 F 방향 영상 총 {len(self.f_videos)}개 발견")
        print(f"📦 총 {self.total_batches}개 배치로 분할")
        
    def _setup_logging(self):
        """로깅 설정"""
        log_file = self.metadata_dir / "processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _collect_f_videos(self) -> List[Dict[str, str]]:
        """F 방향 영상 파일들을 수집"""
        # 비디오 파일 경로: data/1.Training/videos/03/*_F.mp4
        video_dir = self.data_root / "1.Training" / "videos" / "03"
        video_pattern = str(video_dir / "*_F.mp4")
        video_files = glob.glob(video_pattern)
        
        # 라벨 파일 기본 경로: data/1.Training/[라벨]01_real_sen_morpheme/morpheme/03/
        label_dir = self.data_root / "1.Training" / "[라벨]01_real_sen_morpheme" / "morpheme" / "03"
        
        f_videos = []
        for video_path in sorted(video_files):
            # 문장 번호 추출 (예: NIA_SL_SEN0001_REAL03_F.mp4 -> 0001)
            filename = os.path.basename(video_path)
            try:
                sen_num = filename.split('_SEN')[1].split('_')[0]
                sen_id = int(sen_num)
                
                # 해당 형태소 라벨 파일 찾기
                # NIA_SL_SEN0001_REAL03_F.mp4 -> NIA_SL_SEN0001_REAL03_F_morpheme.json
                label_filename = filename.replace('.mp4', '_morpheme.json')
                morpheme_path = label_dir / label_filename
                
                if morpheme_path.exists():
                    f_videos.append({
                        'video_path': video_path,
                        'morpheme_path': str(morpheme_path),
                        'sen_id': sen_id,
                        'filename': filename
                    })
                else:
                    # 다른 라벨 폴더들에서도 찾아보기 (01~16)
                    found = False
                    for label_folder in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16']:
                        alt_label_dir = self.data_root / "1.Training" / "[라벨]01_real_sen_morpheme" / "morpheme" / label_folder
                        alt_morpheme_path = alt_label_dir / label_filename
                        if alt_morpheme_path.exists():
                            f_videos.append({
                                'video_path': video_path,
                                'morpheme_path': str(alt_morpheme_path),
                                'sen_id': sen_id,
                                'filename': filename
                            })
                            found = True
                            break
                    
                    if not found:
                        self.logger.warning(f"형태소 파일 없음: {label_filename}")
                    
            except (IndexError, ValueError) as e:
                self.logger.warning(f"파일명 파싱 실패: {filename} - {e}")
                continue
                
        return f_videos
    
    def _load_morpheme_data(self, morpheme_path: str) -> Dict[str, Any]:
        """형태소 라벨 데이터 로드"""
        try:
            with open(morpheme_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"형태소 데이터 로드 실패: {morpheme_path} - {e}")
            return None
            
    def _process_single_video(self, video_info: Dict[str, str]) -> Dict[str, Any]:
        """단일 영상 처리"""
        video_path = video_info['video_path']
        morpheme_path = video_info['morpheme_path']
        sen_id = video_info['sen_id']
        
        self.logger.info(f"처리 중: SEN{sen_id:04d} - {os.path.basename(video_path)}")
        
        # 비디오 처리
        try:
            stats = self.video_processor.process_video_file(
                video_path=video_path,
                show_progress=False
            )
            
            if not stats:
                self.logger.error(f"비디오 처리 실패: {video_path}")
                return None
                
            # 결합 데이터 수집
            combined_data_dir = Path(self.video_processor.output_dir) / "combined_data"
            json_files = list(combined_data_dir.glob("*_combined.json"))
            
            if not json_files:
                self.logger.warning(f"결합 데이터 없음: {video_path}")
                return None
            
            # 프레임별 데이터 수집
            frames_data = []
            poses_data = []
            
            for json_file in sorted(json_files):
                try:
                    crop_image, keypoints, scores, metadata = self.video_processor.load_combined_data(str(json_file))
                    
                    if crop_image is not None:
                        frames_data.append({
                            'frame_idx': metadata['frame_info']['frame_idx'],
                            'image': crop_image,
                            'bbox': metadata['frame_info']['bbox']
                        })
                        
                        poses_data.append({
                            'frame_idx': metadata['frame_info']['frame_idx'],
                            'keypoints': keypoints,
                            'scores': scores,
                            'stats': metadata['pose_data']['stats']
                        })
                except Exception as e:
                    self.logger.warning(f"프레임 데이터 로드 실패: {json_file} - {e}")
                    continue
            
            # 형태소 데이터 로드
            morpheme_data = self._load_morpheme_data(morpheme_path)
            
            return {
                'sen_id': sen_id,
                'video_info': {
                    'path': video_path,
                    'filename': video_info['filename'],
                    'stats': stats
                },
                'frames_data': frames_data,
                'poses_data': poses_data,
                'morpheme_data': morpheme_data
            }
            
        except Exception as e:
            self.logger.error(f"영상 처리 실패: {video_path} - {e}")
            return None
    
    def _save_batch_frames(self, batch_idx: int, batch_data: List[Dict[str, Any]]):
        """배치 프레임 데이터를 HDF5로 저장"""
        h5_path = self.frames_dir / f"batch_{batch_idx:02d}_F_frames.h5"
        
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['batch_idx'] = batch_idx
            h5f.attrs['total_videos'] = len(batch_data)
            h5f.attrs['created_at'] = datetime.now().isoformat()
            
            for video_data in batch_data:
                if not video_data or not video_data['frames_data']:
                    continue
                    
                sen_id = video_data['sen_id']
                group_name = f"video_{sen_id:04d}"
                video_group = h5f.create_group(group_name)
                
                # 프레임 데이터 준비
                frames_list = []
                frame_indices = []
                bboxes = []
                
                for frame_data in video_data['frames_data']:
                    frames_list.append(frame_data['image'])
                    frame_indices.append(frame_data['frame_idx'])
                    bboxes.append(frame_data['bbox'])
                
                if frames_list:
                    # 프레임 이미지들 저장 (N, H, W, C)
                    frames_array = np.stack(frames_list, axis=0)
                    video_group.create_dataset('frames', data=frames_array, 
                                             compression='gzip', compression_opts=6)
                    
                    # 프레임 인덱스 저장
                    video_group.create_dataset('frame_indices', data=np.array(frame_indices))
                    
                    # 바운딩박스 저장
                    video_group.create_dataset('bboxes', data=np.array(bboxes))
                    
                    # 메타데이터 저장
                    video_group.attrs['sen_id'] = sen_id
                    video_group.attrs['filename'] = video_data['video_info']['filename']
                    video_group.attrs['total_frames'] = len(frames_list)
                    video_group.attrs['frame_shape'] = frames_array.shape[1:]  # (H, W, C)
        
        self.logger.info(f"✅ 프레임 배치 저장: {h5_path}")
    
    def _save_batch_poses(self, batch_idx: int, batch_data: List[Dict[str, Any]]):
        """배치 포즈 데이터를 HDF5로 저장"""
        h5_path = self.poses_dir / f"batch_{batch_idx:02d}_F_poses.h5"
        
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['batch_idx'] = batch_idx
            h5f.attrs['total_videos'] = len(batch_data)
            h5f.attrs['created_at'] = datetime.now().isoformat()
            
            for video_data in batch_data:
                if not video_data or not video_data['poses_data']:
                    continue
                    
                sen_id = video_data['sen_id']
                group_name = f"video_{sen_id:04d}"
                video_group = h5f.create_group(group_name)
                
                # 포즈 데이터 준비
                keypoints_list = []
                scores_list = []
                frame_indices = []
                valid_frames = []
                
                for pose_data in video_data['poses_data']:
                    keypoints_list.append(pose_data['keypoints'])
                    scores_list.append(pose_data['scores'])
                    frame_indices.append(pose_data['frame_idx'])
                    # 유효한 프레임 판단 (평균 신뢰도 > 0.3)
                    valid_frames.append(pose_data['stats']['average_confidence'] > 0.3)
                
                if keypoints_list:
                    # 키포인트 저장 (N, 133, 2)
                    keypoints_array = np.stack(keypoints_list, axis=0)
                    video_group.create_dataset('keypoints', data=keypoints_array,
                                             compression='gzip', compression_opts=6)
                    
                    # 신뢰도 저장 (N, 133)
                    scores_array = np.stack(scores_list, axis=0)
                    video_group.create_dataset('scores', data=scores_array,
                                             compression='gzip', compression_opts=6)
                    
                    # 프레임 인덱스 저장
                    video_group.create_dataset('frame_indices', data=np.array(frame_indices))
                    
                    # 유효 프레임 마스크 저장
                    video_group.create_dataset('valid_frames', data=np.array(valid_frames))
                    
                    # 포즈 통계 저장
                    stats_group = video_group.create_group('pose_stats')
                    if video_data['poses_data']:
                        avg_stats = {
                            'avg_confidence': np.mean([p['stats']['average_confidence'] for p in video_data['poses_data']]),
                            'avg_valid_keypoints': np.mean([p['stats']['valid_keypoints'] for p in video_data['poses_data']]),
                            'avg_high_conf_keypoints': np.mean([p['stats']['high_confidence_keypoints'] for p in video_data['poses_data']])
                        }
                        for key, value in avg_stats.items():
                            stats_group.attrs[key] = value
                    
                    # 메타데이터 저장
                    video_group.attrs['sen_id'] = sen_id
                    video_group.attrs['total_frames'] = len(keypoints_list)
                    video_group.attrs['pose_shape'] = keypoints_array.shape[1:]  # (133, 2)
        
        self.logger.info(f"✅ 포즈 배치 저장: {h5_path}")
    
    def _save_batch_labels(self, batch_idx: int, batch_data: List[Dict[str, Any]]):
        """배치 라벨 데이터를 HDF5로 저장"""
        h5_path = self.labels_dir / f"batch_{batch_idx:02d}_F_labels.h5"
        
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['batch_idx'] = batch_idx
            h5f.attrs['total_videos'] = len(batch_data)
            h5f.attrs['created_at'] = datetime.now().isoformat()
            
            for video_data in batch_data:
                if not video_data or not video_data['morpheme_data']:
                    continue
                    
                sen_id = video_data['sen_id']
                group_name = f"video_{sen_id:04d}"
                video_group = h5f.create_group(group_name)
                
                morpheme_data = video_data['morpheme_data']
                
                # 메타데이터 저장
                if 'metaData' in morpheme_data:
                    meta = morpheme_data['metaData']
                    video_group.attrs['url'] = meta.get('url', '')
                    video_group.attrs['duration'] = meta.get('duration', 0.0)
                    video_group.attrs['exported_on'] = meta.get('exportedOn', '')
                
                # 형태소 데이터 저장
                if 'data' in morpheme_data and morpheme_data['data']:
                    morpheme_info = morpheme_data['data'][0]  # 첫 번째 형태소 정보
                    
                    video_group.attrs['start_time'] = morpheme_info.get('start', 0.0)
                    video_group.attrs['end_time'] = morpheme_info.get('end', 0.0)
                    
                    # 속성(형태소) 저장
                    if 'attributes' in morpheme_info and morpheme_info['attributes']:
                        morpheme_text = morpheme_info['attributes'][0].get('name', '')
                        video_group.attrs['morpheme_text'] = morpheme_text
                        
                        # 텍스트를 UTF-8 바이트로 저장 (HDF5 문자열 지원)
                        video_group.create_dataset('morpheme_text_data', 
                                                 data=morpheme_text.encode('utf-8'))
                
                # 프레임-시간 매핑 계산 (비디오 FPS 기반)
                video_stats = video_data['video_info']['stats']
                if video_stats and 'video_info' in video_stats:
                    fps = video_stats['video_info'].get('fps', 30.0)
                    total_frames = video_stats['processing_info'].get('processed_frames', 0)
                    
                    # 시간 배열 생성
                    if total_frames > 0:
                        time_array = np.linspace(0, total_frames / fps, total_frames)
                        video_group.create_dataset('frame_time_mapping', data=time_array)
                
                video_group.attrs['sen_id'] = sen_id
        
        self.logger.info(f"✅ 라벨 배치 저장: {h5_path}")
    
    def process_all_batches(self):
        """모든 배치 처리"""
        self.logger.info(f"🚀 F 방향 HDF5 배치 처리 시작")
        self.logger.info(f"📊 총 {len(self.f_videos)}개 영상, {self.total_batches}개 배치")
        
        # 배치 정보 저장
        batch_info = {
            'total_videos': len(self.f_videos),
            'total_batches': self.total_batches,
            'videos_per_batch': self.videos_per_batch,
            'direction': 'F',
            'created_at': datetime.now().isoformat(),
            'batches': []
        }
        
        for batch_idx in range(self.total_batches):
            start_idx = batch_idx * self.videos_per_batch
            end_idx = min(start_idx + self.videos_per_batch, len(self.f_videos))
            batch_videos = self.f_videos[start_idx:end_idx]
            
            self.logger.info(f"\n📦 배치 {batch_idx:02d} 처리 중 ({len(batch_videos)}개 영상)")
            self.logger.info(f"   범위: SEN{batch_videos[0]['sen_id']:04d} ~ SEN{batch_videos[-1]['sen_id']:04d}")
            
            # 배치 영상들 처리
            batch_data = []
            for i, video_info in enumerate(batch_videos):
                self.logger.info(f"   [{i+1}/{len(batch_videos)}] 처리 중...")
                video_result = self._process_single_video(video_info)
                if video_result:
                    batch_data.append(video_result)
            
            if batch_data:
                # HDF5 파일들 저장
                self._save_batch_frames(batch_idx, batch_data)
                self._save_batch_poses(batch_idx, batch_data)
                self._save_batch_labels(batch_idx, batch_data)
                
                # 배치 정보 업데이트
                batch_info['batches'].append({
                    'batch_idx': batch_idx,
                    'processed_videos': len(batch_data),
                    'sen_range': f"SEN{batch_videos[0]['sen_id']:04d}-SEN{batch_videos[-1]['sen_id']:04d}",
                    'files': {
                        'frames': f"batch_{batch_idx:02d}_F_frames.h5",
                        'poses': f"batch_{batch_idx:02d}_F_poses.h5", 
                        'labels': f"batch_{batch_idx:02d}_F_labels.h5"
                    }
                })
                
                self.logger.info(f"✅ 배치 {batch_idx:02d} 완료 ({len(batch_data)}개 영상 처리)")
            else:
                self.logger.warning(f"⚠️ 배치 {batch_idx:02d} 처리 실패")
        
        # 전체 배치 정보 저장
        batch_info_path = self.metadata_dir / "batch_info.json"
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n🎉 모든 배치 처리 완료!")
        self.logger.info(f"📁 출력 디렉토리: {self.output_root}")
        self.logger.info(f"📊 배치 정보: {batch_info_path}")

    def process_test_batch(self, num_videos: int = 5):
        """테스트용으로 소수의 영상만 처리"""
        self.logger.info(f"🧪 테스트 배치 처리 시작 ({num_videos}개 영상)")
        
        if not self.f_videos:
            self.logger.error("F 방향 영상이 없습니다.")
            return
        
        # 처음 몇 개 영상만 처리
        test_videos = self.f_videos[:num_videos]
        
        # 테스트 배치 처리
        batch_data = []
        for i, video_info in enumerate(test_videos):
            self.logger.info(f"   [{i+1}/{len(test_videos)}] 테스트 처리 중...")
            video_result = self._process_single_video(video_info)
            if video_result:
                batch_data.append(video_result)
        
        if batch_data:
            # 테스트 HDF5 파일들 저장
            self._save_batch_frames(99, batch_data)  # batch_99로 저장
            self._save_batch_poses(99, batch_data)
            self._save_batch_labels(99, batch_data)
            
            self.logger.info(f"✅ 테스트 배치 완료 ({len(batch_data)}개 영상 처리)")
            return True
        else:
            self.logger.warning(f"⚠️ 테스트 배치 처리 실패")
            return False

def main():
    """메인 함수"""
    # 설정
    data_root = "/home/ty/rtmw/02/mmpose/jy/data"
    rtmw_config = "../configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py"
    rtmw_checkpoint = "../models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
    output_root = "sign_language_dataset"
    
    # 처리기 생성
    processor = SignLanguageHDF5BatchProcessor(
        data_root=data_root,
        rtmw_config=rtmw_config,
        rtmw_checkpoint=rtmw_checkpoint,
        output_root=output_root,
        videos_per_batch=250
    )
    
    # 사용자 선택
    print("\n처리 모드를 선택하세요:")
    print("1. 테스트 처리 (5개 영상)")
    print("2. 전체 배치 처리 (모든 영상)")
    print("3. 영상 목록만 확인")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == '1':
        print("🧪 테스트 처리를 시작합니다...")
        success = processor.process_test_batch(num_videos=5)
        if success:
            print("✅ 테스트 처리가 완료되었습니다!")
        else:
            print("❌ 테스트 처리가 실패했습니다.")
    elif choice == '2':
        print("🚀 전체 배치 처리를 시작합니다...")
        processor.process_all_batches()
    elif choice == '3':
        print("📝 영상 목록:")
        for i, video in enumerate(processor.f_videos[:10]):  # 처음 10개만 표시
            print(f"   {i+1}. SEN{video['sen_id']:04d} - {video['filename']}")
        if len(processor.f_videos) > 10:
            print(f"   ... 총 {len(processor.f_videos)}개")
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == "__main__":
    main()
