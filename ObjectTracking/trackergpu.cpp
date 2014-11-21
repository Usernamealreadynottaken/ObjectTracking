#include "trackergpu.h"


TrackerGpu::TrackerGpu(std::string videoFile, int hessian, std::vector<std::string> imagefiles)
	: minHessian(hessian)
{
	detector = cv::gpu::SURF_GPU(minHessian);
	capture.open(videoFile);
	cv::namedWindow("Video");
	for (size_t i = 0; i < imagefiles.size(); ++i)
	{
		images.push_back(cv::Mat());
		greyframes.push_back(cv::Mat());
		gpuimages.push_back(cv::gpu::GpuMat());
		images[i] = cv::imread(imagefiles[i]);
		//cv::namedWindow("Image" + std::to_string(i));
	}
	bfmatcher = cv::gpu::BruteForceMatcher_GPU_base();
	color = cv::Scalar(0, 255, 0);
	width = 4;
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

		if (!success)
		{
			break;
		}
		cv::cvtColor(frame, greyFrame, CV_BGR2GRAY);
		gpuframe.upload(greyFrame);

		// detect keypoints
		detector(gpuframe, cv::gpu::GpuMat(), gpukeypoints, gpudescriptors);
		cv::gpu::GpuMat trainIdx, distance;
		bfmatcher.match(gpudescriptors, gpudescriptorsimage, matches);

		detector.downloadKeypoints(gpukeypoints, keypoints);
		//detector.downloadDescriptors(gpudescriptors, descriptors);
		
		//cv::drawMatches(cv::Mat(frame), keypoints, cv::Mat(image), keypointsimage, matches, img_matches);

		//double max_dist = 0; double min_dist = 100;

		////-- Quick calculation of max and min distances between keypoints
		//for( int i = 0; i < gpudescriptors.rows; i++ )
		//{ 
		//	double dist = matches[i].distance;
		//	if( dist < min_dist ) min_dist = dist;
		//	if( dist > max_dist ) max_dist = dist;
		//}

		//std::vector< cv::DMatch > good_matches;

		//for( int i = 0; i < gpudescriptors.rows; i++ )
		//{ 
		//	if( matches[i].distance < 3*min_dist )
		//	{ 
		//		good_matches.push_back( matches[i]); 
		//	}
		//}

		//-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for(size_t i = 0; i < matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypointsimage[ matches[i].trainIdx ].pt );
			scene.push_back( keypoints[ matches[i].queryIdx ].pt );
		}
		
		//if (obj.size() > 4)
		//{
		//	homography = cv::findHomography( cv::Mat(obj), cv::Mat(scene), CV_RANSAC, 10);
		//	//-- Get the corners from the image_1 ( the object to be "detected" )
		//	std::vector<cv::Point2f> obj_corners(4);
		//	obj_corners[0] = cvPoint(0,0);
		//	obj_corners[1] = cvPoint( image.cols, 0 );
		//	obj_corners[2] = cvPoint( image.cols, image.rows );
		//	obj_corners[3] = cvPoint( 0, image.rows );
		//	std::vector<cv::Point2f> scene_corners(4);

		//	cv::perspectiveTransform(obj_corners, scene_corners, homography);

		//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		//	line( frame, scene_corners[0], scene_corners[1], color, width );
		//	line( frame, scene_corners[1], scene_corners[2], color, width );
		//	line( frame, scene_corners[2], scene_corners[3], color, width );
		//	line( frame, scene_corners[3], scene_corners[0], color, width );
		//}

		// drawing dramatically decreases performance 
		//cv::drawKeypoints(frame, keypoints, frame);
		//cv::drawMatches(frame, keypoints, image, keypointsimage, matches, img_matches);

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
	for (size_t i = 0; i < images.size(); ++i)
	{
		cv::cvtColor(images[i], greyframes[i], CV_BGR2GRAY);
		gpuimages[i].upload(greyframes[i]);
	}
	detector(gpuframe, cv::gpu::GpuMat(), gpukeypointsimage, gpudescriptorsimage);
	detector.downloadKeypoints(gpukeypointsimage, keypointsimage);
	detector.downloadDescriptors(gpudescriptorsimage, descriptorsimage);

	// one time operation, unnecessary
	//cv::drawKeypoints(image, keypointsimage, image);
	/*for (size_t i = 0; i < images.size(); ++i)
	{
		cv::imshow("Image" + std::to_string(i), images[i]);
	}*/
}