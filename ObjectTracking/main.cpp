#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main()
{
	cv::VideoCapture capture;
	capture.open("Wildlife.wmv");
	cv::Mat frame;
	//cv::namedWindow("Video", CV_WINDOW_AUTOSIZE);
	while (capture.isOpened())
	{
		bool success = capture.read(frame);
		if (!success)
		{
			break;
		}
		// show next frame
		cv::imshow("Video", frame);

		// break after pressing esc
		if (cv::waitKey(30) == 27)
		{
			cv::destroyAllWindows();
			break;
		}
	}
	capture.release();
	
	std::cin.get();
	std::cin.get();
	return 0;
}