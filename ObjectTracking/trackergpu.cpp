#include "trackergpu.h"


TrackerGpu::TrackerGpu(std::string videoFile, int hessian, std::vector<std::string> imagefiles)
	: minHessian(hessian)
{
	detector = cv::gpu::SURF_GPU(minHessian);
	CONFIDENCE = 0.95;
	INLIER_RATIO = 0.18;
	INLIER_THRESHOLD = 3.0;
	capture.open(videoFile);
	cv::namedWindow("Video");
	for (size_t i = 0; i < imagefiles.size(); ++i)
	{
		images.push_back(cv::Mat());
		greyframes.push_back(cv::Mat());
		gpuimages.push_back(cv::gpu::GpuMat());
		keypointsimage.push_back(std::vector<cv::KeyPoint>());
		descriptorsimage.push_back(std::vector<float>());
		gpukeypointsimage.push_back(cv::gpu::GpuMat());
		gpudescriptorsimage.push_back(cv::gpu::GpuMat());

		images[i] = cv::imread(imagefiles[i]);
		//cv::namedWindow("Image" + std::to_string(i));
	}
	bfmatcher = cv::gpu::BruteForceMatcher_GPU< cv::L2<float> >();
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
	bool success = capture.read(frame);
	writer = cv::VideoWriter("out.avi", CV_FOURCC('M','J','P','G'), 5, cv::Size(frame.cols, frame.rows), true);
	while (capture.isOpened())
	{
		if (!success)
		{
			break;
		}
		cv::cvtColor(frame, greyFrame, CV_BGR2GRAY);
		gpuframe.upload(greyFrame);
		
		// detect keypoints
		detector(gpuframe, cv::gpu::GpuMat(), gpukeypoints, gpudescriptors);
		detector.downloadKeypoints(gpukeypoints, keypoints);
		//detector.downloadDescriptors(gpudescriptors, descriptors);

		// match keypoints
		bfmatcher.match(gpudescriptors, gpudescriptorsimage[0], matches);
		//bfmatcher.knnMatchSingle(gpudescriptors, gpudescriptorsimage[0], gpu_ret_idx, gpu_ret_dist, gpu_all_dist, 3);

		//gpu_ret_idx.download(ret_idx);
  //      gpu_ret_dist.download(ret_dist);

		//float ratio = 0.7f;
  //      float min_val = FLT_MAX;
  //      float max_val = 0.0f;
  //      for(int i=0; i < ret_idx.rows; i++) {
  //          if(ret_dist.at<float>(i,0) < ret_dist.at<float>(i,1)*ratio) {
  //              int idx = ret_idx.at<int>(i,0);

  //              Point2Df a, b;
  //              a.x = keypoints[i].pt.x;
  //              a.y = keypoints[i].pt.y;

  //              b.x = keypointsimage[0][idx].pt.x;
  //              b.y = keypointsimage[0][idx].pt.y;

  //              src.push_back(b);
  //              dst.push_back(a);
  //              match_score.push_back(ret_dist.at<float>(i,0));

  //              if(ret_dist.at<float>(i,0) < min_val) {
  //                  min_val = ret_dist.at<float>(i,0);
  //              }

  //              if(ret_dist.at<float>(i,0) > max_val) {
  //                  max_val = ret_dist.at<float>(i,0);
  //              }
  //          }
  //      }

  //      // Flip score
  //      for(unsigned int i=0; i < match_score.size(); i++) {
  //          match_score[i] = max_val - match_score[i] + min_val;
  //      }

		//
		//int K = (int)(log(1.0 - CONFIDENCE) / log(1.0 - pow(INLIER_RATIO, 4.0)));
		//if (src.size() > 4)
		//{
		//	CUDA_RANSAC_Homography(src, dst, match_score, INLIER_THRESHOLD, K, &best_inliers, best_H, &inlier_mask);
		//	
		//	for(int i=0; i < 9; i++) {
		//		best_H[i] /= best_H[8];
		//	}

		//	cv::Point2f upperleft(best_H[2], best_H[5]);
		//	cv::Point2f upperright((images.at(0).cols * best_H[0] + best_H[2]) / (images.at(0).cols * best_H[6] + best_H[8]),
		//			(images.at(0).cols * best_H[3] + best_H[5]) / (images.at(0).cols * best_H[6] + best_H[8]));
		//	cv::Point2f lowerleft((images[0].rows * best_H[1] + best_H[2]) / (images[0].rows * best_H[7] + best_H[8]),
		//			(images[0].rows * best_H[4] + best_H[5]) / (images[0].rows * best_H[7] + best_H[8]));
		//	cv::Point2f lowerright((images.at(0).cols * best_H[0] + images.at(0).rows * best_H[1] + best_H[2]) / 
		//			(images.at(0).cols * best_H[6] + images[0].rows * best_H[7] + best_H[8]),
		//			(images.at(0).cols * best_H[3] + images[0].rows * best_H[4] + best_H[5]) / 
		//			(images.at(0).cols * best_H[6] + images[0].rows * best_H[7] + best_H[8]));

		//	line(frame, upperright, upperleft, color, width);
		//	line(frame, upperright, lowerright, color, width);
		//	line(frame, upperleft, lowerleft, color, width);
		//	line(frame, lowerleft, lowerright, color, width);
		//}

		//-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for(size_t i = 0; i < matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypointsimage[0][ matches[i].trainIdx ].pt );
			scene.push_back( keypoints[ matches[i].queryIdx ].pt );
		}
		
		if (obj.size() > 4)
		{
			homography = cv::findHomography( obj, scene, CV_RANSAC, 10);
			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<cv::Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0);
			obj_corners[1] = cvPoint( images.at(0).cols, 0 );
			obj_corners[2] = cvPoint( images.at(0).cols, images.at(0).rows );
			obj_corners[3] = cvPoint( 0, images.at(0).rows );
			std::vector<cv::Point2f> scene_corners(4);

			cv::perspectiveTransform(obj_corners, scene_corners, homography);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line( frame, scene_corners[0], scene_corners[1], color, width );
			line( frame, scene_corners[1], scene_corners[2], color, width );
			line( frame, scene_corners[2], scene_corners[3], color, width );
			line( frame, scene_corners[3], scene_corners[0], color, width );
		}

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

		src.clear();
		dst.clear();
		match_score.clear();
		
		// break after pressing esc
		if (cv::waitKey(30) == 27)
		{
			cv::destroyAllWindows();
			break;
		}
		//writer << frame;
		success = capture.read(frame);
	}
}

void TrackerGpu::calculateKeypointsImage()
{
	for (size_t i = 0; i < images.size(); ++i)
	{
		cv::cvtColor(images[i], greyframes[i], CV_BGR2GRAY);
		gpuimages[i].upload(greyframes[i]);
		detector(gpuimages[i], cv::gpu::GpuMat(), gpukeypointsimage[i], gpudescriptorsimage[i]);
		detector.downloadKeypoints(gpukeypointsimage[i], keypointsimage[i]);
		detector.downloadDescriptors(gpudescriptorsimage[i], descriptorsimage[i]);
	}

	// one time operation, unnecessary
	//cv::drawKeypoints(image, keypointsimage, image);
	/*for (size_t i = 0; i < images.size(); ++i)
	{
		cv::imshow("Image" + std::to_string(i), images[i]);
	}*/
}