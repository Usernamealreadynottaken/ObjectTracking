#include "tracker.h"

Tracker::Tracker(std::string videoFile, int hessian, std::string imageFile)
	: minHessian(hessian)
{
	detector = cv::SurfFeatureDetector(minHessian);
	capture.open(videoFile);
	cv::namedWindow("Video");
	image = cv::imread(imageFile);
	cv::namedWindow("Image");
}

void Tracker::track()
{
	calculateKeypointsImage();
	time_t start, end;
	double fps;
	int counter = 0;
	double sec;
	time(&start);
	while (capture.isOpened())
	{
		bool success = capture.read(frame);
		//cv::cvtColor(frame, frame, CV_BGR2GRAY);

		if (!success)
		{
			break;
		}

		// detect keypoints
		detector.detect(frame, keypoints);
		//cv::drawKeypoints(frame, keypoints, frame);

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

void Tracker::calculateKeypointsImage()
{
	detector.detect(image, keypointsimage);

	// one time operation, unnecessary
	//cv::drawKeypoints(image, keypointsimage, image);
	cv::imshow("Image", image);
}