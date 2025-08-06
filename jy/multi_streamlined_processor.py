#!/usr/bin/env python3
"""
병렬 스트림라인 비디오 처리기 - 배치별 멀티프로세싱 지원
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import GPUtil
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

# 기존 BatchProcessor 클래스를 상속
from streamlined_processor import BatchProcessor, StreamlinedVideoProcessor

class ParallelBatchProcessor(BatchProcessor):
    """병렬 처리를 지원하는 배치 처리기"""
    
    def __init__(self, 
                 data_root: str = "data/1.Training",
                 output_dir: str = "sign_language_dataset",
                 batch_size: int = 250,
                 rtmw_model_name: str = "rtmw-x",
                 direction: str = "F",
                 item_types: List[str] = ["WORD"],
                 max_workers: int = None,
                 gpu_per_worker: float = 0.3):  # 각 워커당 GPU 메모리 비율
        
        super().__init__(data_root, output_dir, batch_size, rtmw_model_name, direction, item_types)
        
        # 병렬 처리 설정
        self.max_workers = self._determine_optimal_workers(max_workers, gpu_per_worker)
        self.gpu_per_worker = gpu_per_worker
        
        self.logger.info(f"🚀 병렬 처리 설정: {self.max_workers}개 워커, GPU 메모리/워커: {gpu_per_worker*100:.0f}%")

    def _determine_optimal_workers(self, max_workers: Optional[int], gpu_per_worker: float) -> int:
        """최적 워커 수 결정"""
        
        # 1. CPU 기반 제한
        cpu_count = mp.cpu_count()
        cpu_workers = min(cpu_count - 1, 4)  # CPU 코어 수 -1, 최대 4개
        
        # 2. 메모리 기반 제한 (각 워커당 4GB 필요 가정)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        ram_workers = max(1, int(available_ram_gb / 4))
        
        # 3. GPU 메모리 기반 제한
        gpu_workers = 1
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # XPU의 경우 정확한 메모리 정보를 얻기 어려우므로 보수적으로 설정
                gpu_workers = max(1, int(1.0 / gpu_per_worker))  # 예: 0.3이면 최대 3개
            else:
                # GPU 정보를 얻을 수 없는 경우 (XPU 등)
                gpu_workers = max(1, int(1.0 / gpu_per_worker))
        except:
            gpu_workers = 2  # 기본값
        
        # 4. 사용자 지정 값 또는 최소값 선택
        if max_workers:
            optimal_workers = min(max_workers, cpu_workers, ram_workers, gpu_workers)
        else:
            optimal_workers = min(cpu_workers, ram_workers, gpu_workers)
        
        self.logger.info(f"📊 워커 수 결정: CPU={cpu_workers}, RAM={ram_workers}, GPU={gpu_workers}")
        return max(1, optimal_workers)

    def process_single_batch_worker(self, batch_info_and_config: Tuple[Dict, Dict]) -> Tuple[int, bool, str]:
        """
        단일 배치를 처리하는 워커 함수 (멀티프로세싱용)
        
        Args:
            batch_info_and_config: (batch_info, processor_config) 튜플
            
        Returns:
            Tuple[int, bool, str]: (batch_id, success, message)
        """
        batch_info, processor_config = batch_info_and_config
        batch_id = batch_info['batch_id']
        
        try:
            # 각 프로세스에서 독립적으로 처리기 초기화
            processor = StreamlinedVideoProcessor(rtmw_model_name=processor_config['rtmw_model_name'])
            
            # 출력 디렉토리 설정
            video_output_dir = Path(processor_config['video_output_dir'])
            hdf5_output_dir = Path(processor_config['hdf5_output_dir'])
            
            # 로깅 설정 (각 프로세스별)
            logger = logging.getLogger(f"batch_{batch_id}")
            
            logger.info(f"🔄 배치 {batch_id} 처리 시작 (PID: {mp.current_process().pid})")
            
            # 1. 비디오 처리
            successful_data = {}
            batch_data = batch_info['data']
            
            for item_type, item_id, video_path in batch_data:
                success, crop_images = processor.process_video(item_type, item_id, video_path, video_output_dir)
                if success:
                    key = f"{item_type}{item_id:04d}"
                    successful_data[key] = {
                        'crop_images': crop_images,
                        'item_type': item_type,
                        'item_id': item_id
                    }
            
            if not successful_data:
                return batch_id, False, "처리된 비디오 없음"
            
            # 2. HDF5 생성
            self._create_hdf5_single_batch(successful_data, batch_info, processor_config, logger)
            
            # 3. 중간 파일 정리 (설정에 따라)
            if processor_config.get('cleanup_intermediate', False):
                for key in successful_data.keys():
                    item_dir = video_output_dir / key
                    if item_dir.exists():
                        import shutil
                        shutil.rmtree(item_dir)
            
            success_count = len(successful_data)
            total_count = len(batch_data)
            
            return batch_id, True, f"{success_count}/{total_count}개 성공"
            
        except Exception as e:
            import traceback
            error_msg = f"배치 {batch_id} 처리 실패: {str(e)}\n{traceback.format_exc()}"
            return batch_id, False, error_msg

    def _create_hdf5_single_batch(self, successful_data: Dict, batch_info: Dict, config: Dict, logger):
        """단일 배치의 HDF5 파일 생성 (워커용)"""
        import h5py
        import cv2
        from datetime import datetime
        
        batch_id = batch_info['batch_id']
        folder_name = batch_info['folder_name']
        folder_batch_idx = batch_info['folder_batch_idx']
        item_range = batch_info['item_range']
        
        hdf5_output_dir = Path(config['hdf5_output_dir'])
        video_output_dir = Path(config['video_output_dir'])
        
        # HDF5 파일 경로
        item_types = config['item_types']
        direction = config['direction']
        types_str = "_".join(item_types)
        
        frames_h5_path = hdf5_output_dir / f"batch_{types_str}_{folder_name}_{folder_batch_idx:02d}_{direction}_frames.h5"
        poses_h5_path = hdf5_output_dir / f"batch_{types_str}_{folder_name}_{folder_batch_idx:02d}_{direction}_poses.h5"
        
        # JPEG 인코딩된 데이터를 위한 가변 길이 타입
        jpeg_vlen_dtype = h5py.vlen_dtype(np.uint8)

        with h5py.File(frames_h5_path, 'w') as f_frames, \
             h5py.File(poses_h5_path, 'w') as f_poses:
            
            # 배치 메타데이터
            batch_metadata = {
                'folder_name': folder_name,
                'folder_batch_idx': folder_batch_idx,
                'item_range': item_range,
                'item_types': item_types,
                'direction': direction,
                'video_count': len(successful_data),
                'creation_time': str(datetime.now()),
                'processed_by_pid': mp.current_process().pid
            }
            
            f_frames.attrs.update(batch_metadata)
            f_poses.attrs.update(batch_metadata)
            
            # 데이터 저장
            keys = sorted(successful_data.keys())
            for key in keys:
                data = successful_data[key]
                item_type = data['item_type']
                item_id = data['item_id']
                crop_images = data['crop_images']
                
                # 저장된 데이터 로드
                item_dir = video_output_dir / f"{item_type}{item_id:04d}"
                keypoints_scaled = np.load(item_dir / "keypoints_scaled.npy")
                scores = np.load(item_dir / "scores.npy")
                
                with open(item_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                video_group = f"video_{item_type.lower()}{item_id:04d}"
                
                # 프레임 데이터 저장
                frame_group = f_frames.create_group(video_group)
                jpeg_frames = [cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1] 
                              for img in crop_images]
                
                frame_group.create_dataset("frames_jpeg", 
                                         data=jpeg_frames, 
                                         dtype=jpeg_vlen_dtype, 
                                         compression='lzf')
                f_frames.create_dataset(f"{video_group}/metadata", data=json.dumps(metadata))
                
                # 포즈 데이터 저장
                pose_group = f_poses.create_group(video_group)
                pose_group.create_dataset("keypoints_scaled", data=keypoints_scaled, compression='lzf')
                pose_group.create_dataset("scores", data=scores, compression='lzf')
        
        logger.info(f"✅ 배치 {batch_id} HDF5 생성 완료")

    def process_all_batches_parallel(self, cleanup_intermediate: bool = False):
        """병렬로 모든 배치 처리"""
        folder_video_data = self.collect_videos_by_folder()
        if not folder_video_data:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
        
        all_batches = self.create_batches_by_folder(folder_video_data)
        
        if not all_batches:
            self.logger.error("❌ 생성된 배치가 없습니다")
            return
        
        self.logger.info(f"🚀 병렬 배치 처리 시작: {len(all_batches)}개 배치, {self.max_workers}개 워커")
        
        # 프로세서 설정 준비
        processor_config = {
            'rtmw_model_name': self.rtmw_model_name,
            'video_output_dir': str(self.video_output_dir),
            'hdf5_output_dir': str(self.hdf5_output_dir),
            'item_types': self.item_types,
            'direction': self.direction,
            'cleanup_intermediate': cleanup_intermediate
        }
        
        # 배치와 설정을 튜플로 묶기
        batch_tasks = [(batch_info, processor_config) for batch_info in all_batches]
        
        start_time = time.time()
        completed_batches = 0
        failed_batches = 0
        
        # 병렬 처리 실행
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 작업 제출
            future_to_batch = {
                executor.submit(self.process_single_batch_worker, task): task[0]['batch_id'] 
                for task in batch_tasks
            }
            
            # 완료되는 대로 결과 처리
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result_batch_id, success, message = future.result()
                    
                    if success:
                        completed_batches += 1
                        self.logger.info(f"✅ 배치 {result_batch_id} 완료: {message}")
                    else:
                        failed_batches += 1
                        self.logger.error(f"❌ 배치 {result_batch_id} 실패: {message}")
                    
                    # 진행률 표시
                    total_processed = completed_batches + failed_batches
                    progress = (total_processed / len(all_batches)) * 100
                    elapsed = time.time() - start_time
                    
                    if total_processed > 0:
                        eta = (elapsed / total_processed) * (len(all_batches) - total_processed)
                        self.logger.info(f"📊 진행률: {progress:.1f}% ({total_processed}/{len(all_batches)}), "
                                      f"경과: {elapsed/60:.1f}분, 예상 완료: {eta/60:.1f}분")
                    
                except Exception as e:
                    failed_batches += 1
                    self.logger.error(f"❌ 배치 {batch_id} 예외 발생: {e}")
        
        total_time = time.time() - start_time
        
        self.logger.info("🎉 병렬 배치 처리 완료!")
        self.logger.info(f"📊 최종 결과: 성공 {completed_batches}개, 실패 {failed_batches}개")
        self.logger.info(f"⏱️ 총 소요 시간: {total_time/60:.1f}분")
        
        if completed_batches > 0:
            avg_time_per_batch = total_time / len(all_batches)
            self.logger.info(f"📈 평균 배치당 처리 시간: {avg_time_per_batch:.1f}초")

    def process_test_parallel(self, folder_name: str = None, test_count: int = 10, test_workers: int = 2):
        """병렬 처리 테스트 (소규모)"""
        folder_video_data = self.collect_videos_by_folder()
        
        if not folder_video_data:
            self.logger.error("❌ 처리할 영상이 없습니다")
            return
        
        # 테스트할 폴더 선택
        if folder_name and folder_name in folder_video_data:
            test_folder = folder_name
        else:
            test_folder = list(folder_video_data.keys())[0]
        
        video_data = folder_video_data[test_folder][:test_count]
        
        # 작은 배치들로 나누기 (테스트용)
        small_batch_size = max(1, test_count // test_workers)
        test_batches = []
        
        for i in range(0, len(video_data), small_batch_size):
            batch_data = video_data[i:i + small_batch_size]
            batch_info = {
                'batch_id': 900 + i // small_batch_size,  # 테스트용 배치 번호
                'folder_name': test_folder,
                'folder_batch_idx': i // small_batch_size,
                'data': batch_data,
                'item_range': f"{batch_data[0][0]}{batch_data[0][1]:04d}~{batch_data[-1][0]}{batch_data[-1][1]:04d}"
            }
            test_batches.append(batch_info)
        
        self.logger.info(f"🧪 병렬 테스트 시작: {len(test_batches)}개 배치, {test_workers}개 워커")
        
        # 임시로 워커 수 조정
        original_workers = self.max_workers
        self.max_workers = test_workers
        
        # 프로세서 설정
        processor_config = {
            'rtmw_model_name': self.rtmw_model_name,
            'video_output_dir': str(self.video_output_dir),
            'hdf5_output_dir': str(self.hdf5_output_dir),
            'item_types': self.item_types,
            'direction': self.direction,
            'cleanup_intermediate': False
        }
        
        batch_tasks = [(batch_info, processor_config) for batch_info in test_batches]
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=test_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_single_batch_worker, task): task[0]['batch_id'] 
                for task in batch_tasks
            }
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result_batch_id, success, message = future.result()
                    if success:
                        self.logger.info(f"✅ 테스트 배치 {result_batch_id} 완료: {message}")
                    else:
                        self.logger.error(f"❌ 테스트 배치 {result_batch_id} 실패: {message}")
                except Exception as e:
                    self.logger.error(f"❌ 테스트 배치 {batch_id} 예외: {e}")
        
        test_time = time.time() - start_time
        self.logger.info(f"🧪 병렬 테스트 완료: {test_time:.1f}초 소요")
        
        # 원래 워커 수 복원
        self.max_workers = original_workers


def main():
    """병렬 처리 메인 함수"""
    print("🚀 병렬 스트림라인 배치 처리기 (WORD + SEN 지원)")
    print("=" * 60)
    
    # 기본 설정은 기존과 동일...
    # (타입, 모델, 방향 선택 코드 생략)
    
    # 병렬 처리 설정
    print("\n병렬 처리 설정:")
    print(f"1. 자동 설정 (권장)")
    print(f"2. 수동 설정")
    
    parallel_choice = input("선택 (1-2, 기본값: 1): ").strip()
    
    max_workers = None
    gpu_per_worker = 0.3
    
    if parallel_choice == '2':
        try:
            max_workers = int(input(f"워커 수 (1-8, 기본값: 자동): ").strip())
            gpu_mem = float(input(f"워커당 GPU 메모리 비율 (0.1-0.8, 기본값: 0.3): ").strip())
            if 0.1 <= gpu_mem <= 0.8:
                gpu_per_worker = gpu_mem
        except:
            print("잘못된 입력, 자동 설정 사용")
    
    try:
        # 기본 설정으로 초기화 (실제 사용 시 선택된 값들 사용)
        batch_processor = ParallelBatchProcessor(
            rtmw_model_name='rtmw-x',
            direction='F', 
            item_types=['WORD'],
            max_workers=max_workers,
            gpu_per_worker=gpu_per_worker
        )
        print("✅ 병렬 처리기 초기화 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    while True:
        print("\n처리 모드를 선택하세요:")
        print("1. 병렬 테스트 (10개 영상, 2개 워커)")
        print("2. 전체 병렬 배치 처리")
        print("3. 전체 병렬 배치 처리 + 중간파일 정리")
        print("4. 성능 비교 테스트")
        print("0. 종료")
        
        choice = input("선택 (0-4): ").strip()
        
        if choice == '1':
            batch_processor.process_test_parallel(test_count=10, test_workers=2)
        elif choice == '2':
            batch_processor.process_all_batches_parallel(cleanup_intermediate=False)
        elif choice == '3':
            batch_processor.process_all_batches_parallel(cleanup_intermediate=True)
        elif choice == '4':
            # 성능 비교 테스트 (순차 vs 병렬)
            print("🔬 성능 비교 테스트는 별도로 구현 필요")
        elif choice == '0':
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()