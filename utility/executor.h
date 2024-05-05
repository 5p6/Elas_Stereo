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
		executor();



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