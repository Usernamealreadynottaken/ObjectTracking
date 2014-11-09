#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <vector>
#include <ctime>

class TrackerGpu
{
private:
	// the higher the hessian, the fewer points will surf detect
	int minHessian;
	cv::VideoCapture capture;
	cv::Mat image;
	cv::Mat frame;
	cv::Mat greyFrame;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::KeyPoint> keypointsimage;
	std::vector<float> descriptorsimage;
	std::vector<float> descriptors;
	cv::gpu::SURF_GPU detector;
	cv::gpu::GpuMat gpuframe;
	cv::gpu::GpuMat gpukeypoints;
	cv::gpu::GpuMat gpukeypointsimage;
	cv::gpu::GpuMat gpudescriptors;
	cv::gpu::GpuMat gpudescriptorsimage;
	cv::gpu::BruteForceMatcher_GPU_base bfmatcher;
	cv::Mat homography;
public:
	TrackerGpu(std::string videoFile, std::string imageFile , int hessian);
	~TrackerGpu() { capture.release(); };
	void track();
	void calculateKeypointsImage();
};