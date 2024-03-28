#ifndef EXECUTOR_H
#define EXECUTOR_H
#include <opencv2/opencv.hpp>
#include <elas.h>
#include <iostream>

namespace Inference
{

	class executor
	{

	public:
		executor(executor &e) = delete;
		executor &operator=(executor &e) = delete;

	public:
		executor()
		{
			param.disp_min = 0;					// 最小视差
			param.disp_max = 256;				// 最大视差
			param.support_threshold = 0.85;		// 比率测试：最低match VS 次低match
			param.support_texture = 10;			// 支持点的最小纹理
			param.candidate_stepsize = 5;		// 用于支持点的sobel特征匹配的邻域半径
			param.incon_window_size = 5;		// 不连续性窗口的尺寸
			param.incon_threshold = 5;			// 不连续性窗口内的视差范围阈值
			param.incon_min_support = 5;		// 不连续性窗口内的最低支持点数量
			param.add_corners = true;			// 是否添加角点
			param.grid_size = 20;				// 网格尺寸
			param.beta = 0.02;					// 图像相似性度量的参数
			param.gamma = 3;					// 先验概率常数
			param.sigma = 1;					// 先验概率的标准差
			param.sradius = 3;					// 标准差半径
			param.match_texture = 1;			// 最低纹理
			param.lr_threshold = 1;				// 左右一致性检验阈值
			param.speckle_sim_threshold = 1;	// 连通域判断阈值
			param.speckle_size = 200;			// 连通域噪声尺寸判断阈值
			param.ipol_gap_width = 3;			// 空洞宽
			param.filter_median = false;		// 是否中值滤波
			param.filter_adaptive_mean = true;	// 是否自适应中值滤波
			param.postprocess_only_left = true; // 是否只对左视差图后处理，设置为True可以节省时间
			param.subsampling = false;			// 每个两个像素进行视差计算，设置为True可以节省时间，但是传入的D1和D2的分辨率必须为(w/2) x (h/2)
			model = std::make_unique<Elas>(param);
		}


		executor(const std::string &param_file);
		cv::Mat infer(const cv::Mat &_left, const cv::Mat &_right);
		cv::Mat infer_rectified(const cv::Mat &_left, const cv::Mat &_right);
		std::pair<cv::Mat,cv::Mat> rectify(const cv::Mat &_left, const cv::Mat &_right);



		cv::Mat Q;

	private:
		cv::Mat maplx, maply, maprx, mapry;

		Elas::parameters param;
		std::unique_ptr<Elas> model;
	};
}

#endif