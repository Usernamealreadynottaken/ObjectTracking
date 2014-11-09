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
	// set number of threads during parallel computation
	// doesnt work with TrackerGpu
	//cv::setNumThreads(1);
	//Tracker tracker("Wildlife.wmv", "chocs-front.jpg", 600);
	//tracker.track();

	TrackerGpu tracker("Wildlife.wmv", "chocs-front.jpg", 600);
	tracker.track();
	
	std::cin.get();
	std::cin.get();
	return 0;
}