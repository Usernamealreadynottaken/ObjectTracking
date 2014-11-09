#include "tracker.h"

Tracker::Tracker(std::string videoFile, std::string imageFile , int hessian)
	: minHessian(hessian)
{
	capture.open(videoFile);
	cv::namedWindow("Video");
	cv::Mat image = cv::imread(imageFile);
	cv::namedWindow("Image");
	cv::imshow("Image", image);
}

void Tracker::track()
{
	time_t start, end;
	double fps;
	int counter = 0;
	double sec;
	time(&start);
	while (capture.isOpened())
	{
		bool success = capture.read(frame);

		if (!success)
		{
			break;
		}

		// detect keypoints
		cv::SurfFeatureDetector detector(minHessian);
		detector.detect(frame, keypoints);

		// show next frame
		cv::imshow("Video", frame);

		// calculate and display fps
		time(&end);
		sec = difftime(end, start);
		fps = ++counter / sec;
		std::cout << fps << '\n';

		// break after pressing esc
		if (cv::waitKey(30) == 27)
		{
			cv::destroyAllWindows();
			break;
		}
	}
}