#include "trackergpu.h"


TrackerGpu::TrackerGpu(std::string videoFile, int hessian, std::vector<std::string> imagefiles)
	: minHessian(hessian)
{
	detector = cv::gpu::SURF_GPU(minHessian);
	CONFIDENCE = 0.95f;
	INLIER_RATIO = 0.18f;
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
		matches.push_back(std::vector<cv::DMatch>());
		obj.push_back(std::vector<cv::Point2f>());
		scene.push_back(std::vector<cv::Point2f>());

		images[i] = cv::imread(imagefiles[i]);
		//cv::namedWindow("Image" + std::to_string(i));
	}
	bfmatcher = cv::gpu::BruteForceMatcher_GPU< cv::L2<float> >();
	color = cv::Scalar(0, 255, 0);
	width = 4;
}

void TrackerGpu::ransac_opencv()
{
	std::vector<cv::Point2f> obj_corners(4);
	std::vector<cv::Point2f> scene_corners(4);
	cv::Mat img_matches;
	for (size_t img = 0; img < images.size(); ++img)
	{
		// match keypoints
		bfmatcher.match(gpudescriptors, gpudescriptorsimage[img], matches[img]);

		//-- Localize the object
		obj[img].clear();
		scene[img].clear();
		for(size_t i = 0; i < matches[img].size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj[img].push_back( keypointsimage[img][ matches[img][i].trainIdx ].pt );
			scene[img].push_back( keypoints[ matches[img][i].queryIdx ].pt );
		}
		if (scene[img].size() > 4)
		{
			homography = cv::findHomography( obj[img], scene[img], CV_RANSAC, 10.0);
			//-- Get the corners from the image_1 ( the object to be "detected" )
			obj_corners[0] = cvPoint(0,0);
			obj_corners[1] = cvPoint( images.at(img).cols, 0 );
			obj_corners[2] = cvPoint( images.at(img).cols, images.at(img).rows );
			obj_corners[3] = cvPoint( 0, images.at(img).rows );

			cv::perspectiveTransform(obj_corners, scene_corners, homography);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line( frame, scene_corners[0], scene_corners[1], color, width );
			line( frame, scene_corners[1], scene_corners[2], color, width );
			line( frame, scene_corners[2], scene_corners[3], color, width );
			line( frame, scene_corners[3], scene_corners[0], color, width );
		}
	}
}

void TrackerGpu::ransac_gpu()
{
	for (size_t img = 0; img < images.size(); ++img)
	{
		src.clear();
		dst.clear();
		match_score.clear();
		bfmatcher.knnMatchSingle(gpudescriptors, gpudescriptorsimage[img], gpu_ret_idx, gpu_ret_dist, gpu_all_dist, 3);

		gpu_ret_idx.download(ret_idx);
		gpu_ret_dist.download(ret_dist);

		float ratio = 0.7f;
		float min_val = FLT_MAX;
		float max_val = 0.0f;
		for(int i=0; i < ret_idx.rows; i++) {
			if(ret_dist.at<float>(i,0) < ret_dist.at<float>(i,1)*ratio) {
				int idx = ret_idx.at<int>(i,0);

				Point2Df a, b;
				a.x = keypoints[i].pt.x;
				a.y = keypoints[i].pt.y;

				b.x = keypointsimage[img][idx].pt.x;
				b.y = keypointsimage[img][idx].pt.y;

				src.push_back(b);
				dst.push_back(a);
				match_score.push_back(ret_dist.at<float>(i,0));

				if(ret_dist.at<float>(i,0) < min_val) {
					min_val = ret_dist.at<float>(i,0);
				}

				if(ret_dist.at<float>(i,0) > max_val) {
					max_val = ret_dist.at<float>(i,0);
				}
			}
		}

		// Flip score
		for(unsigned int i=0; i < match_score.size(); i++) {
			match_score[i] = max_val - match_score[i] + min_val;
		}

		
		int K = (int)(log(1.0 - CONFIDENCE) / log(1.0 - pow(INLIER_RATIO, 4.0)));
		if (src.size() > 4)
		{
			CUDA_RANSAC_Homography(src, dst, match_score, INLIER_THRESHOLD, K, &best_inliers, best_H, &inlier_mask);
			
			for(int i=0; i < 9; i++) {
				best_H[i] /= best_H[8];
			}

			cv::Point2f upperleft(best_H[2], best_H[5]);
			cv::Point2f upperright((images.at(0).cols * best_H[0] + best_H[2]) / (images.at(0).cols * best_H[6] + best_H[8]),
					(images.at(0).cols * best_H[3] + best_H[5]) / (images.at(0).cols * best_H[6] + best_H[8]));
			cv::Point2f lowerleft((images[0].rows * best_H[1] + best_H[2]) / (images[0].rows * best_H[7] + best_H[8]),
					(images[0].rows * best_H[4] + best_H[5]) / (images[0].rows * best_H[7] + best_H[8]));
			cv::Point2f lowerright((images.at(0).cols * best_H[0] + images.at(0).rows * best_H[1] + best_H[2]) / 
					(images.at(0).cols * best_H[6] + images[0].rows * best_H[7] + best_H[8]),
					(images.at(0).cols * best_H[3] + images[0].rows * best_H[4] + best_H[5]) / 
					(images.at(0).cols * best_H[6] + images[0].rows * best_H[7] + best_H[8]));

			line(frame, upperright, upperleft, color, width);
			line(frame, upperright, lowerright, color, width);
			line(frame, upperleft, lowerleft, color, width);
			line(frame, lowerleft, lowerright, color, width);
		}
	}
}

void TrackerGpu::ransac()
{
	bool found = false;
	inliers = 0;
	best_inliers = 0;
	int i1, i2, i3, i4;
	cv::Point2f s1, s2, s3, s4, o1, o2, o3, o4;
	cv::Mat matrix(8, 8, CV_64F);
	cv::Mat vector(8, 1, CV_64F);
	cv::Mat perspectiveTransform;

	for (size_t side = 0; side < images.size() && !found; ++side) {
		bfmatcher.match(gpudescriptors, gpudescriptorsimage[side], matches[side]);
		obj[side].clear();
		scene[side].clear();
		for(size_t i = 0; i < matches[side].size(); i++)
		{
			if (matches[side][i].distance < 0.20f) {
				obj[side].push_back( keypointsimage[side][ matches[side][i].trainIdx ].pt );
				scene[side].push_back( keypoints[ matches[side][i].queryIdx ].pt );
			}
		}
		int size = scene[side].size();
		if (size > 10) {
			found = true;
		
			for (int iterations = 0; iterations < 200; ++iterations) {
				inliers = 0;

				i1 = rand() % size;
				do {
					i2 = rand() % size;
				} while (i2 == i1);
				do {
					i3 = rand() % size;
				} while (i3 == i2 || i3 == i1);
				do {
					i4 = rand() % size;
				} while (i4 == i3 || i4 == i2 || i4 == i1);
				s1 = scene[side][i1];
				s2 = scene[side][i2];
				s3 = scene[side][i3];
				s4 = scene[side][i4];
				o1 = obj[side][i1];
				o2 = obj[side][i2];
				o3 = obj[side][i3];
				o4 = obj[side][i4];

				matrix.at<double>(0, 0) = o1.x;
				matrix.at<double>(0, 1) = o1.y;
				matrix.at<double>(0, 2) = 1;
				matrix.at<double>(0, 3) = 0;
				matrix.at<double>(0, 4) = 0;
				matrix.at<double>(0, 5) = 0;
				matrix.at<double>(0, 6) = -s1.x * o1.x;
				matrix.at<double>(0, 7) = -s1.x * o1.y;
		
				matrix.at<double>(1, 0) = o2.x;
				matrix.at<double>(1, 1) = o2.y;
				matrix.at<double>(1, 2) = 1;
				matrix.at<double>(1, 3) = 0;
				matrix.at<double>(1, 4) = 0;
				matrix.at<double>(1, 5) = 0;
				matrix.at<double>(1, 6) = -s2.x * o2.x;
				matrix.at<double>(1, 7) = -s2.x * o2.y;

				matrix.at<double>(2, 0) = o3.x;
				matrix.at<double>(2, 1) = o3.y;
				matrix.at<double>(2, 2) = 1;
				matrix.at<double>(2, 3) = 0;
				matrix.at<double>(2, 4) = 0;
				matrix.at<double>(2, 5) = 0;
				matrix.at<double>(2, 6) = -s3.x * o3.x;
				matrix.at<double>(2, 7) = -s3.x * o3.y;
		
				matrix.at<double>(3, 0) = o4.x;
				matrix.at<double>(3, 1) = o4.y;
				matrix.at<double>(3, 2) = 1;
				matrix.at<double>(3, 3) = 0;
				matrix.at<double>(3, 4) = 0;
				matrix.at<double>(3, 5) = 0;
				matrix.at<double>(3, 6) = -s4.x * o4.x;
				matrix.at<double>(3, 7) = -s4.x * o4.y;
		
				matrix.at<double>(4, 0) = 0;
				matrix.at<double>(4, 1) = 0;
				matrix.at<double>(4, 2) = 0;
				matrix.at<double>(4, 3) = o1.x;
				matrix.at<double>(4, 4) = o1.y;
				matrix.at<double>(4, 5) = 1;
				matrix.at<double>(4, 6) = -s1.y * o1.x;
				matrix.at<double>(4, 7) = -s1.y * o1.y;
		
				matrix.at<double>(5, 0) = 0;
				matrix.at<double>(5, 1) = 0;
				matrix.at<double>(5, 2) = 0;
				matrix.at<double>(5, 3) = o2.x;
				matrix.at<double>(5, 4) = o2.y;
				matrix.at<double>(5, 5) = 1;
				matrix.at<double>(5, 6) = -s2.y * o2.x;
				matrix.at<double>(5, 7) = -s2.y * o2.y;

				matrix.at<double>(6, 0) = 0;
				matrix.at<double>(6, 1) = 0;
				matrix.at<double>(6, 2) = 0;
				matrix.at<double>(6, 3) = o3.x;
				matrix.at<double>(6, 4) = o3.y;
				matrix.at<double>(6, 5) = 1;
				matrix.at<double>(6, 6) = -s3.y * o3.x;
				matrix.at<double>(6, 7) = -s3.y * o3.y;
		
				matrix.at<double>(7, 0) = 0;
				matrix.at<double>(7, 1) = 0;
				matrix.at<double>(7, 2) = 0;
				matrix.at<double>(7, 3) = o4.x;
				matrix.at<double>(7, 4) = o4.y;
				matrix.at<double>(7, 5) = 1;
				matrix.at<double>(7, 6) = -s4.y * o4.x;
				matrix.at<double>(7, 7) = -s4.y * o4.y;

				matrix = matrix.inv();

				vector.at<double>(0, 0) = s1.x;
				vector.at<double>(1, 0) = s2.x;
				vector.at<double>(2, 0) = s3.x;
				vector.at<double>(3, 0) = s4.x;
				vector.at<double>(4, 0) = s1.y;
				vector.at<double>(5, 0) = s2.y;
				vector.at<double>(6, 0) = s3.y;
				vector.at<double>(7, 0) = s4.y;

				perspectiveTransform = matrix * vector;
			
				for (int i = 0; i < size; ++i) {
					objptr = &obj[side][i];
					scnptr = &scene[side][i];
					float denominator = (perspectiveTransform.at<double>(6, 0) * objptr->x) +
						(perspectiveTransform.at<double>(7, 0) * objptr->y) + 1.0f;
					resultPoint.x = (perspectiveTransform.at<double>(0, 0) * objptr->x + perspectiveTransform.at<double>(1, 0) * objptr->y 
						+ perspectiveTransform.at<double>(2, 0)) / denominator;
					resultPoint.y = (perspectiveTransform.at<double>(3, 0) * objptr->x + perspectiveTransform.at<double>(4, 0) * objptr->y 
						+ perspectiveTransform.at<double>(5, 0)) / denominator;
					float distance = abs(pow(resultPoint.x - scnptr->x, 2) + pow(resultPoint.y - scnptr->y, 2));
					if (distance < 1000.0f) {
						++inliers;
					}
				}

				if (inliers > best_inliers) {
					best_H[0] = perspectiveTransform.at<double>(0, 0);
					best_H[1] = perspectiveTransform.at<double>(1, 0);
					best_H[2] = perspectiveTransform.at<double>(2, 0);
					best_H[3] = perspectiveTransform.at<double>(3, 0);
					best_H[4] = perspectiveTransform.at<double>(4, 0);
					best_H[5] = perspectiveTransform.at<double>(5, 0);
					best_H[6] = perspectiveTransform.at<double>(6, 0);
					best_H[7] = perspectiveTransform.at<double>(7, 0);
					best_H[8] = 1.0f;
					best_inliers = inliers;
				}
			}
			/*for (int i = 0; i < 9; ++i) {
				std::cout << best_H[i] << " ";
				std::cout << '\n';
			}*/
		
			cv::Point2f upperleftfront(best_H[2], best_H[5]);
			cv::Point2f upperrightfront((images.at(side).cols * best_H[0] + best_H[2]) / (images.at(side).cols * best_H[6] + best_H[8]),
					(images.at(side).cols * best_H[3] + best_H[5]) / (images.at(side).cols * best_H[6] + best_H[8]));
			cv::Point2f lowerleftfront((images[side].rows * best_H[1] + best_H[2]) / (images[side].rows * best_H[7] + best_H[8]),
					(images[side].rows * best_H[4] + best_H[5]) / (images[side].rows * best_H[7] + best_H[8]));
			cv::Point2f lowerrightfront((images.at(side).cols * best_H[0] + images.at(side).rows * best_H[1] + best_H[2]) / 
					(images.at(side).cols * best_H[6] + images[side].rows * best_H[7] + best_H[8]),
					(images.at(side).cols * best_H[3] + images[side].rows * best_H[4] + best_H[5]) / 
					(images.at(side).cols * best_H[6] + images[side].rows * best_H[7] + best_H[8]));

			float left = sqrt(pow(upperleftfront.x - lowerleftfront.x, 2) + pow(upperleftfront.y - lowerleftfront.y, 2));
			float right = sqrt(pow(upperrightfront.x - lowerrightfront.x, 2) + pow(upperrightfront.y - lowerrightfront.y, 2));
			float top = sqrt(pow(upperleftfront.x - upperrightfront.x, 2) + pow(upperleftfront.y - upperrightfront.y, 2));
			float bottom = sqrt(pow(lowerleftfront.x - lowerrightfront.x, 2) + pow(lowerleftfront.y - lowerrightfront.y, 2));
			float anglex = 1 - left / right;
			float angley = 1 - top / bottom;

			cv::Point2f upperleftback(upperleftfront.x + anglex * left, upperleftfront.y + angley * top);
			cv::Point2f upperrightback(upperrightfront.x + anglex * right, upperrightfront.y + angley * top);
			cv::Point2f lowerleftback(lowerleftfront.x + anglex * left, lowerleftfront.y + angley * bottom);
			cv::Point2f lowerrightback(lowerrightfront.x + anglex * right, lowerrightfront.y + angley * bottom);

			line(frame, upperleftback, upperleftfront, color, width);
			line(frame, upperrightback, upperrightfront, color, width);
			line(frame, lowerleftback, lowerleftfront, color, width);
			line(frame, lowerrightback, lowerrightfront, color, width);
		
			/*std::cout << "best inliers: " << best_inliers << '\n';
			std::cout << upperright.x << " " << upperright.y << '\n';
			std::cout << upperleft.x << " " << upperleft.y << '\n';
			std::cout << lowerright.x << " " << lowerright.y << '\n';
			std::cout << lowerleft.x << " " << lowerleft.y << '\n';*/

			line(frame, upperrightfront, upperleftfront, color, width);
			line(frame, upperrightfront, lowerrightfront, color, width);
			line(frame, upperleftfront, lowerleftfront, color, width);
			line(frame, lowerleftfront, lowerrightfront, color, width);
		}
	}
}

void TrackerGpu::track()
{
	calculateKeypointsImage();
	time_t start, end;
	double fps;
	int counter = 0;
	double sec;
	time(&start);
	bool success = capture.read(frame);
	writer = cv::VideoWriter("out.avi", CV_FOURCC('M','J','P','G'), 12, cv::Size(frame.cols, frame.rows), true);
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

		//ransac_opencv();
		//ransac_gpu();
		ransac();

		// drawing dramatically decreases performance 
		//cv::drawKeypoints(frame, keypoints, frame);
		
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
		//detector.downloadDescriptors(gpudescriptorsimage[i], descriptorsimage[i]);
	}

	// one time operation, unnecessary
	//cv::drawKeypoints(image, keypointsimage, image);
	/*for (size_t i = 0; i < images.size(); ++i)
	{
		cv::imshow("Image" + std::to_string(i), images[i]);
	}*/
}