
from models import IntraModel, VideoModel_cnn, DMC
from utils import get_state_dict, replicate_pad, rgb2ycbcr, ycbcr420_to_444_np, \
    np_image_to_tensor, ycbcr2rgb, YUV420Reader
from layers import CodeBookReSort, RSQBitStream, CodebookChannel
from test_config import TestConfig

import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import time
import numpy as np
import os
import concurrent.futures
import pandas as pd
from pathlib import Path
import math
import json
from typing import Dict, List, Tuple, Optional
import logging
from piq import psnr, ssim, multi_scale_ssim

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

inet, pnet = None, None

class MetricsCalculator:
    """指标计算器"""
    import lpips
    lpips_fn = lpips.LPIPS(net='alex')
    
    @staticmethod
    def calculate_psnr(original, reconstructed) -> float:
        """计算PSNR指标"""
        return psnr(original, reconstructed, data_range=1.).item()
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """计算SSIM指标"""
        return ssim(original, reconstructed, data_range=1.).item()
    
    @staticmethod
    def calculate_msssim(original, reconstructed) -> float:
        """计算MS-SSIM指标"""
        return multi_scale_ssim(original, reconstructed, data_range=1.).item()

    def calculate_lpips(self, original, reconstructed) -> float:
        """计算LPIPS指标"""
        return self.lpips_fn.to(original.device)(original, reconstructed).item()
    
    def calculate_all_metrics(self, original, reconstructed) -> Dict[str, float]:
        """计算所有指标，返回字典格式"""
        metrics = {
            'psnr': self.calculate_psnr(original, reconstructed),
            'ssim': self.calculate_ssim(original, reconstructed),
            'ms_ssim': self.calculate_msssim(original, reconstructed),
            'lpips': self.calculate_lpips(original, reconstructed)
        }
        return metrics
    
    


class VideoTester:
    """视频测试器"""
    
    def __init__(self, model_config: Dict, test_config: TestConfig, device: str = "cuda"):
        self.device = device
        self.model_config = model_config
        self.test_config = test_config
        self.metrics_calculator = MetricsCalculator()
        self.load_models()
    
    def load_models(self):
        """加载模型"""
        global inet, pnet
        inet_path = self.model_config.get('inet_path')
        pnet_path = self.model_config.get('pnet_path')
        
        if inet_path and os.path.exists(inet_path):
            inet = IntraModel(device=self.device, codebook_num=10)
            inet = inet.to(self.device)
            inet.load_state_dict(get_state_dict(inet_path), strict=True)   
            inet.eval()
        
        if pnet_path and os.path.exists(pnet_path):
            # pnet = VideoModel_cnn(device=self.device)           # CNN
            pnet = DMC(device=self.device)                        # CWF
            pnet = pnet.to(self.device)
            ckpt = get_state_dict(pnet_path)
            pnet.load_state_dict(ckpt, strict=True)
            pnet.eval()
            
        # Codebook Reordering
        # from layers.codebook_reorder import multiple_codebook_reorder
        # multiple_codebook_reorder(inet.codebook, minPts=self.test_config.minPts, metric="cosine")
        # multiple_codebook_reorder(pnet.codebook, minPts=self.test_config.minPts, metric="cosine")
        
    
    def read_yuv_frame(self, src_reader, device):
        """读取YUV帧"""
        y, uv = src_reader.read_one_frame()
        if y is None: 
            return None
        yuv = ycbcr420_to_444_np(y, uv)
        x = np_image_to_tensor(yuv, device)
        return x
    
    @torch.no_grad()
    def test_single_video(self, yuv_path: str, width: int, height: int, 
                         frame_num: int = 96, pad: int = 64) -> Dict:
        """测试单个视频文件"""
        logger.info(f"开始测试视频: {yuv_path}")
        torch.cuda.synchronize()
        
        start_time = time.time()
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        
        # 初始化配置
        # cfg = TestConfig()
        cfg = self.test_config
        cfg.yuv_path = yuv_path
        cfg.width = width
        cfg.height = height
        cfg.frame_num = frame_num
        cfg.pad = pad
        # channel = CodebookChannel(channel_type="awgn", bit_width=5)
        
        src_reader = YUV420Reader(yuv_path, width=width, height=height, skip_frame=0)
        
        frame_metrics = []
        total_encode_time = 0
        total_decode_time = 0
        
        for frame_idx in range(frame_num):
            
            x = self.read_yuv_frame(src_reader, self.device)
            if x is None:
                break
            
            _, _, H_c, W_c = x.size()
            x = replicate_pad(x, pad=pad)
            
            encode_start = time.time()
            
            if frame_idx == 0 or (cfg.p_num > 0 and frame_idx % cfg.p_num == 0):
                # intra frame
                encoded = inet.encode(x, qp=0)
                
                indices_list, shape = encoded["indices_list"], encoded["shape"]
                
                # -------------------------------------------------------------------- #
                # binary channel 
                indices_list = [v - (2**4) for v in indices_list]   # TODO: symmetry = True
                
                bit_stream = RSQBitStream(index_list=indices_list, shape=shape, 
                                        bit_width=5, gray_code=cfg.gray_code, num_bits_per_symbol=cfg.order)
                temp = bit_stream.forward(symmetry=cfg.symmetry, snr=cfg.snr)
                
                indices_list = [v + (2**4) for v in temp["index_list"]]
                # -------------------------------------------------------------------- #
                # indices_list = channel(indices_list, snr_db=cfg.snr)
                
                decode_start = time.time()
                out = inet.decode(indices_list, shape, qp=0)
                decode_time = time.time() - decode_start
                
                pnet.clear_dpb()
                pnet.add_ref_frame(None, out["x_hat"])
            else:
                fa_idx = index_map[frame_idx % 8]
                curr_qp = pnet.shift_qp(63, fa_idx)
                if cfg.interval > 0 and frame_idx % cfg.interval == 1:
                    pnet.reset_ref_feature()
                
                temp = pnet.encode(x, curr_qp)
                indices_list, shape = temp["indices_list"], temp["shape"]
                
                # -------------------------------------------------------------------- #
                indices_list = [v - (2**4) for v in indices_list]
                
                bit_stream = RSQBitStream(index_list=indices_list, shape=shape, 
                                        bit_width=5, gray_code=cfg.gray_code, num_bits_per_symbol=cfg.order)
                temp = bit_stream.forward(symmetry=cfg.symmetry, snr=cfg.snr)
                
                indices_list = [v + (2**4) for v in temp["index_list"]]
                # -------------------------------------------------------------------- #
                # indices_list = channel(indices_list, snr_db=cfg.snr)
                
                decode_start = time.time()
                out = pnet.decode(indices_list, shape, curr_qp)
            
                decode_time = time.time() - decode_start
            
            encode_time = time.time() - encode_start - decode_time
            total_encode_time += encode_time
            total_decode_time += decode_time
            
            # TODO: 计算指标
            org = ycbcr2rgb(x)[:, :, :H_c, :W_c]
            recon = ycbcr2rgb(out["x_hat"])[:, :, :H_c, :W_c]
            
            # org_np = org.cpu().numpy().transpose(1, 2, 0) * 255
            # recon_np = recon.cpu().numpy().transpose(1, 2, 0) * 255
            # psnr = self.metrics_calculator.calculate_psnr(org, recon)
            # ssim = self.metrics_calculator.calculate_ssim(org, recon)
            # msssim = self.metrics_calculator.calculate_msssim(org, recon)
            metrics = self.metrics_calculator.calculate_all_metrics(org, recon)
            
            frame_metrics.append({
                'frame_idx': frame_idx,
                'encode_time': encode_time,
                'decode_time': decode_time,
                **metrics
            })
            
            if frame_idx % 10 == 0:
                logger.info(f"处理帧 {frame_idx}: \n{metrics}")
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # 计算统计信息
        if frame_metrics:
            df = pd.DataFrame(frame_metrics)
            video_stats = {
                'video_name': Path(yuv_path).stem,
                'total_frames': len(frame_metrics),
                'total_encode_time': total_encode_time,
                'total_decode_time': total_decode_time,
                'total_time': total_time,
                'avg_encode_time_per_frame': total_encode_time / len(frame_metrics),
                'avg_decode_time_per_frame': total_decode_time / len(frame_metrics),
                'frame_metrics': frame_metrics
            }
            tmp1 = {"avg_" + key: df[key].mean() for key in metrics.keys() if key not in ['frame_idx']}
            video_stats.update(tmp1)
        else:
            video_stats = {'video_name': Path(yuv_path).stem, 'error': 'No frames processed'}
        
        logger.info(f"完成测试视频: {yuv_path}")
        return video_stats


class MultiProcessVideoTester:
    """多进程视频测试器"""
    def __init__(self, model_config: Dict, dataset_path: str, 
                 output_dir: str = "results", num_processes: int = 4, test_config=None):
        self.model_config = model_config
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.test_cfg = test_config
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def find_yuv_files(self) -> List[Tuple[str, int, int]]:
        """查找YUV文件并解析分辨率"""
        yuv_files = []
        dataset_path = Path(self.dataset_path)
        
        # 支持多种分辨率格式
        resolution_patterns = [
            ('_1920x1080_', (1920, 1080)),
            ('_1280x720_', (1280, 720)),
            ('_3840x2160_', (3840, 2160)),
            ('_416x240_', (416, 240)),
            ('_832x480_', (832, 480)),
        ]
        
        for yuv_file in dataset_path.rglob("*.yuv"):
            width, height = None, None
            
            # 尝试从文件名解析分辨率
            for pattern, (w, h) in resolution_patterns:
                if pattern in yuv_file.stem:
                    width, height = w, h
                    break
            
            # 如果无法解析，使用默认分辨率
            if width is None:
                width, height = 1920, 1080  # 默认分辨率
                logger.warning(f"无法从文件名解析分辨率，使用默认值 {width}x{height}: {yuv_file}")
            
            yuv_files.append((str(yuv_file), width, height))
        
        return yuv_files
    
    def test_single_video_wrapper(self, args: Tuple) -> Dict:
        """包装单个视频测试函数用于多进程"""
        yuv_path, width, height = args
        try:
            tester = VideoTester(self.model_config, 
                                test_config=self.test_cfg,
                                device=DEVICE_)
            return tester.test_single_video(yuv_path, width, height)
        except Exception as e:
            logger.error(f"测试视频失败 {yuv_path}: {e}")
            return {'video_name': Path(yuv_path).stem, 'error': str(e)}
    
    def run_test(self) -> pd.DataFrame:
        """运行多进程测试"""
        yuv_files = self.find_yuv_files()
        logger.info(f"找到 {len(yuv_files)} 个YUV文件")
        
        if not yuv_files:
            logger.error("未找到YUV文件")
            return pd.DataFrame()
        
        # 多进程测试
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
        #     results = list(executor.map(self.test_single_video_wrapper, yuv_files))
        results = []
        for yuv_file in yuv_files:
            results.append(self.test_single_video_wrapper(yuv_file))
        
        # 处理结果
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        if error_results:
            logger.warning(f"{len(error_results)} 个视频测试失败")
        
        # 生成详细结果表格
        detailed_results = self.generate_detailed_results(valid_results)
        
        # 生成汇总统计
        summary_results = self.generate_summary_statistics(valid_results)
        
        # 保存结果
        self.save_results(detailed_results, summary_results, error_results)
        
        return detailed_results
    
    def generate_detailed_results(self, results: List[Dict]) -> pd.DataFrame:
        """生成详细结果表格"""
        detailed_data = []
        
        for video_result in results:
            video_name = video_result['video_name']
            
            # 视频级别统计
            tmp1 = {key: video_result[key] for key in video_result.keys() if "avg" in key}
            detailed_data.append({
                'video_name': video_name,
                'metric_type': 'video_average',
                'total_encode_time': video_result['total_encode_time'],
                'total_decode_time': video_result['total_decode_time'],
                'total_time': video_result['total_time'],
                'frame_count': video_result['total_frames'],
                **tmp1
            })
            
            # 帧级别数据（抽样显示，避免表格过大）
            frame_metrics = video_result.get('frame_metrics', [])
            for i, frame_metric in enumerate(frame_metrics):
                if i % 10 == 0:  # 每10帧显示一次
                    tmp1 = {key: frame_metric[key] for key in frame_metric.keys() if key not in ['frame_idx']}
                    detailed_data.append({
                        'video_name': video_name,
                        'metric_type': f'frame_{frame_metric["frame_idx"]:03d}',
                        'total_time': frame_metric['encode_time'] + frame_metric['decode_time'],
                        'frame_count': 1,
                        **tmp1
                    })
        
        return pd.DataFrame(detailed_data)
    
    def generate_summary_statistics(self, results: List[Dict]) -> Dict:
        """生成汇总统计"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        tmp1 = {key: df[key].mean() for key in df.keys() if "avg" in key}
        summary = {
            'total_videos': len(results),
            'total_frames': df['total_frames'].sum(),
            **tmp1,
            'total_encode_time': df['total_encode_time'].sum(),
            'total_decode_time': df['total_decode_time'].sum(),
            'total_processing_time': df['total_time'].sum(),
            'avg_encode_time_per_frame': df['total_encode_time'].sum() / df['total_frames'].sum(),
            'avg_decode_time_per_frame': df['total_decode_time'].sum() / df['total_frames'].sum()
        }
        
        # 将numpy数据类型转换为Python标准类型，确保JSON序列化
        for key, value in summary.items():
            if hasattr(value, 'item'):  # 如果是numpy类型
                summary[key] = value.item()
            else:
                summary[key] = value
        
        return summary
    
    def save_results(self, detailed_results: pd.DataFrame, 
                    summary_results: Dict, error_results: List[Dict]):
        """保存结果到文件"""
        
        # 保存详细结果到Excel
        excel_path = os.path.join(self.output_dir, f"video_test_results.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            detailed_results.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # 创建汇总表
            summary_df = pd.DataFrame([summary_results])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 错误信息表
            if error_results:
                error_df = pd.DataFrame(error_results)
                error_df.to_excel(writer, sheet_name='Errors', index=False)
        
        # 保存汇总统计到JSON
        json_path = os.path.join(self.output_dir, f"summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"结果已保存到: {excel_path}, {json_path}")


def main():
    """主函数"""
    
    # 模型配置  
    model_config = {
        'inet_path': '/home/yangxiao/DCVC-RT-Training/code/video_semantic_com/rrsq_digital_semantic_com/checkpoints/intra_model_best_codebook10.pth.tar',
        # 'pnet_path': '/home/yangxiao/DCVC-RT-Training/code/video_semantic_com/rrsq_digital_semantic_com/checkpoints/cnn_pnet.pth.tar',
        'pnet_path': '/home/yangxiao/DCVC-RT-Training/code/video_semantic_com/rrsq_digital_semantic_com/checkpoints/pnet_cwf_codebook_best.pth.tar'
    }
    
    # 测试配置
    dataset_paths = [
                    "/home/yangxiao/DCVC-RT-Training/video_test/JCT-VC-HEVC/HEVCceshishipin/ClassD",
                    # "/home/yangxiao/DCVC-RT-Training/video_test/JCT-VC-HEVC/HEVCceshishipin/ClassB",
                    "/home/yangxiao/DCVC-RT-Training/video_test/JCT-VC-HEVC/HEVCceshishipin/ClassC",
                    "/home/yangxiao/DCVC-RT-Training/video_test/JCT-VC-HEVC/HEVCceshishipin/ClassE",
                    "/home/yangxiao/DCVC-RT-Training/video_test/MCL-JCV",
                    "/home/yangxiao/DCVC-RT-Training/video_test/UVG/yuv"
                    ] 
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Model Tester")
    parser.add_argument("--snr", type=int, default=13, help="SNR value")
    parser.add_argument("--minPts", type=int, default=2, help="minPts value")
    parser.add_argument("--order", type=int, default=1, help="order value")
    args = parser.parse_args()
    
    test_cfg = TestConfig()
    test_cfg.order = args.order
    test_cfg.snr = args.snr
    test_cfg.minPts = args.minPts
    print(f"SNR: {test_cfg.snr}")
    for  dataset_path in dataset_paths:
        basename = os.path.basename(dataset_path) if os.path.basename(dataset_path) != "yuv" else "UVG"
        output_dir = os.path.join("/home/yangxiao/DCVC-RT-Training/code/video_semantic_com/rrsq_digital_semantic_com/Ours_CWF", 
                                  basename + f"_snr{test_cfg.snr}_{'BPSK' if test_cfg.order == 1 else f'{2**test_cfg.order}QAM'}"
                                  )
        num_processes = 1  # 根据CPU核心数调整
        
        # 创建测试器并运行测试
        tester = MultiProcessVideoTester(
            model_config=model_config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_processes=num_processes,
            test_config=test_cfg
        )
        
        logger.info("开始视频模型测试...")
        results = tester.run_test()
        logger.info("测试完成！")
        
        # 打印汇总信息
        if not results.empty:
            print("\n=== 测试汇总 ===")
            print(f"处理视频数量: {len(results['video_name'].unique())}")
            video_avg = results[results['metric_type'] == 'video_average']
            print(f"平均PSNR: {video_avg['psnr'].mean():.2f} dB")
            print(f"平均SSIM: {video_avg['ssim'].mean():.4f}")
            print(f"平均MS-SSIM: {video_avg['ms_ssim'].mean():.4f}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    DEVICE_ = "cuda:0"
    main()
    
# 