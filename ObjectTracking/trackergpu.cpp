#include "trackergpu.h"


TrackerGpu::TrackerGpu(std::string videoFile, std::string imageFile , int hessian)
	: minHessian(hessian)
{
	capture.open(videoFile);
	cv::namedWindow("Video");
	image = cv::imread(imageFile);
	cv::namedWindow("Image");
	bfmatcher = cv::gpu::BruteForceMatcher_GPU_base();
}

void TrackerGpu::track()
{
	std::vector<cv::DMatch> matches;
	cv::Mat img_matches;

	calculateKeypointsImage();
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
		cv::gpu::GpuMat trainIdx, distance;
		bfmatcher.match(gpudescriptors, gpudescriptorsimage, matches);

		detector.downloadKeypoints(gpukeypoints, keypoints);
		detector.downloadDescriptors(gpudescriptors, descriptors);
		detector.downloadKeypoints(gpukeypointsimage, keypointsimage);
		detector.downloadDescriptors(gpudescriptorsimage, descriptorsimage);

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < gpudescriptors.rows; i++ )
		{ 
			double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}

		std::vector< cv::DMatch > good_matches;

		for( int i = 0; i < gpudescriptors.rows; i++ )
		{ 
			if( matches[i].distance < 3*min_dist )
			{ 
				good_matches.push_back( matches[i]); 
			}
		}

		//-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for( int i = 0; i < good_matches.size(); i++ )
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypoints[ good_matches[i].queryIdx ].pt );
			scene.push_back( keypointsimage[ good_matches[i].trainIdx ].pt );
		}
		
		if (obj.size() > 4)
			homography = cv::findHomography( cv::Mat(obj), cv::Mat(scene) );
		

		//cv::drawMatches(cv::Mat(frame), keypoints, cv::Mat(image), keypointsimage, matches, img_matches);

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

void TrackerGpu::calculateKeypointsImage()
{
	cv::cvtColor(image, greyFrame, CV_BGR2GRAY);
	gpuframe.upload(greyFrame);
	detector(gpuframe, cv::gpu::GpuMat(), gpukeypointsimage, gpudescriptorsimage);

	// one time operation, unnecessary
	//cv::drawKeypoints(image, keypointsimage, image);
	cv::imshow("Image", image);
}