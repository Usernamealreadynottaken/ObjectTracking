#include "trackergpu.h"

TrackerGpu::TrackerGpu(std::string videoFile, std::string imageFile , int hessian)
	: minHessian(hessian)
{
	capture.open(videoFile);
	cv::namedWindow("Video");
	cv::Mat image = cv::imread(imageFile);
	cv::namedWindow("Image");
	cv::imshow("Image", image);
}

void TrackerGpu::track()
{
	time_t start, end;
	double fps;
	int counter = 0;
	double sec;
	time(&start);
	while (capture.isOpened())
	{
		bool success = capture.read(frame);
		cv::cvtColor(frame, greyFrame, CV_BGR2GRAY);
		gpuframe.upload(greyFrame);

		if (!success)
		{
			break;
		}

		// detect keypoints
		detector(gpuframe, cv::gpu::GpuMat(), gpukeypoints, gpudescriptors);
		detector.downloadKeypoints(gpukeypoints, keypoints);
		// drawing dramatically decreases performance 
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