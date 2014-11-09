#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <iostream>
#include <vector>
#include <ctime>

class TrackerGpu
{
private:
	// the higher the hessian, the fewer points will surf detect
	int minHessian;
	cv::VideoCapture capture;
	cv::Mat frame;
	cv::Mat greyFrame;
	std::vector<cv::KeyPoint> keypoints;
	cv::gpu::SURF_GPU detector;
	cv::gpu::GpuMat gpuframe;
	cv::gpu::GpuMat gpukeypoints;
	cv::gpu::GpuMat gpudescriptors;
public:
	TrackerGpu(std::string videoFile, std::string imageFile , int hessian);
	~TrackerGpu() { capture.release(); };
	void track();
};