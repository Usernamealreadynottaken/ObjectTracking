#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <ctime>
#include <iostream>

class Tracker
{
private:
	// the higher the hessian, the fewer points will surf detect
	int minHessian;
	cv::VideoCapture capture;
	cv::Mat image;
	cv::Mat frame;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::KeyPoint> keypointsimage;
	cv::SurfFeatureDetector detector;
public:
	Tracker(std::string videoFile, std::string imageFile , int hessian);
	~Tracker() { capture.release(); };
	void track();
	void calculateKeypointsImage();
};