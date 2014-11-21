#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <iostream>
#include <vector>
#include "tracker.h"
#include "trackergpu.h"

int main()
{
	std::vector<std::string> imagefiles;
	imagefiles.push_back("chocs-front.jpg");
	imagefiles.push_back("chocs-back.jpg");
	imagefiles.push_back("chocs-bottom.jpg");
	imagefiles.push_back("chocs-top.jpg");
	imagefiles.push_back("chocs-left.jpg");
	imagefiles.push_back("chocs-right.jpg");

	// set number of threads during parallel computation
	// doesnt work with TrackerGpu
	//cv::setNumThreads(4);
	//Tracker tracker("chocs-test.avi", 400, "chocs-front.jpg");
	//tracker.track();

	TrackerGpu tracker("chocs-test.avi", 400, imagefiles);
	tracker.track();
	
	std::cin.get();
	std::cin.get();
	return 0;
}