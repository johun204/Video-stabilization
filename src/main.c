#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Windows.h>

#define STABILIZATION_MARGIN 50

#define VECTOR_ADD(X,Y) sqrtf(((X)*(X)) + ((Y)*(Y))) // 벡터 합 연산
#define DEG(RAD) (((RAD) + ((RAD) < 0 ? PI : 0)) / PI * 180) // Rad to Deg 함수 
#define K 0.04
#define THRESHOLD 0.01
#define HISTOGRAMSIZE 16 // Corner 좌표 주변으로 Histogram 할 사이즈
#define HISTOGRAMINTV 8
#define WINDOWSIZE 5 // R 계산을 위한 window의 크기
#define PI 3.141593

using namespace cv;

// Histogram of Oriented Gradient 구조체 및  선언
typedef struct _HOG {
	float d[9] = { 0 };
} HOG;
HOG plusHOG(HOG a, HOG b) {
	HOG result;
	for (int k = 0; k < 9; k++)
		result.d[k] = a.d[k] + b.d[k];
	return result;
}HOG minusHOG(HOG a, HOG b) {
	HOG result;
	for (int k = 0; k < 9; k++)
		result.d[k] = a.d[k] - b.d[k];
	return result;
}


// 가우시안 필터 함수
void GaussianFilter(Mat &Img, Mat &OutputImg, int filterSize, float sigma) {
	float *Filter = (float*)malloc(sizeof(float) * filterSize * filterSize);
	int halfSize = filterSize / 2;
	int inputWidth = Img.cols;
	int inputHeight = Img.rows;

	for (int i = 0; i <= halfSize; i++)
		for (int j = 0; j <= halfSize; j++)
			Filter[(filterSize - i - 1) * filterSize + (filterSize - j - 1)] = Filter[(filterSize - i - 1) * filterSize + j] = Filter[i * filterSize + (filterSize - j - 1)] = Filter[i * filterSize + j] = exp(-(abs(halfSize - i) + abs(halfSize - j)) / (2 * sigma * sigma)) / (sqrtf(2 * PI) * sigma);

	for (int i = 0; i < inputHeight; i++) {
		for (int j = 0; j < inputWidth; j++) {
			float sum = 0, FilterSum = 0;

			// 필터 크기만큼 반복문 수행 
			for (int x = 0; x < filterSize; x++) {
				if (i + x - halfSize < 0 || i + x - halfSize >= inputHeight)
					continue;
				for (int y = 0; y < filterSize; y++) {
					if (j + y - halfSize < 0 || j + y - halfSize >= inputWidth)
						continue;

					// 가우시안 필터 연산시 필터값의 합으로 나누기 때문에 필터값의 합 연산
					FilterSum += Filter[x * filterSize + y];

					// 필터와 이미지를 연산하여 sum 합 연산
					sum += Filter[x * filterSize + y] * Img.at<uchar>(i + x - halfSize, j + y - halfSize);
				}
			}
			// 필터의 연산값의 합을 필터값의 합으로 나누어줌
			OutputImg.at<uchar>(i, j) = sum / FilterSum;
		}
	}
	free(Filter);
}

// Harris Corner Detection 함수
int HarrisCornerDetect(Mat &Image, HOG *Hg, Point2d *POS) {
	Mat Img;
	cvtColor(Image, Img, CV_RGB2GRAY);
	int inputHeight = Img.rows;
	int inputWidth = Img.cols;

	float *ImgMag = (float*)malloc(sizeof(float) * inputHeight * inputWidth);
	float *ImgPhase = (float*)malloc(sizeof(float) * inputHeight * inputWidth);

	// 가우시안 필터 적용 (sigma : 1.0, size : 5)
	GaussianFilter(Img, Img, 5, 1.0);

	// Ix, Iy값을 저장하기 위한 포인터 변수
	float *Ix = (float*)malloc(sizeof(float) * inputHeight * inputWidth);
	float *Iy = (float*)malloc(sizeof(float) * inputHeight * inputWidth);

	// SOBEL EDGE FILTER 처리
	int Filter[3] = { -1, 0, 1 };

	// Magnitude Normalize 를 위해 최대값 최소값 계산
	int Mag_max = INT_MAX, Mag_min = INT_MIN;
	for (int i = 0; i < inputHeight; i++)
		for (int j = 0; j < inputWidth; j++) {
			int fx = 0, fy = 0;
			for (int x = 0; x < 3; x++) {

				// Boundary 예외 처리
				if (i + x - 1 < 0 || i + x > inputHeight || i + 1 - x < 0 || i + 2 - x > inputHeight)
					continue;
				for (int y = 0; y < 3; y++) {

					// Boundary 예외 처리
					if (j + y - 1 < 0 || j + y > inputWidth || j + 1 - y < 0 || j + 2 - y > inputWidth)
						continue;

					// SOBEL FILTER 연산
					if (y != 1) fx += Filter[y] * Img.at<uchar>(i + x - 1, j + y - 1);
					if (x != 1) fy += Filter[x] * Img.at<uchar>(i + x - 1, j + y - 1);
				}
			}

			// 벡터 합 연산을 이용하여  Magnitude 계산
			ImgMag[i * inputWidth + j] = VECTOR_ADD(fx, fy);

			// 최대, 최소값 구하기 (Magnitude Normalize를 위해)
			Mag_max = (Mag_max < ImgMag[i * inputWidth + j]) ? ImgMag[i * inputWidth + j] : Mag_max; 
			Mag_min = (Mag_min > ImgMag[i * inputWidth + j]) ? ImgMag[i * inputWidth + j] : Mag_min;

			// arc tangent 함수를 이용하여 Phase 계산
			ImgPhase[i * inputWidth + j] = atan2(fy, fx);

			// Ix와 Iy를 포인터 변수에 저장
			Ix[i * inputWidth + j] = fx;
			Iy[i * inputWidth + j] = fy;
		}

	// Magnitude Normalize
	for (int i = 0; i < inputHeight; i++)
		for (int j = 0; j < inputWidth; j++)
			ImgMag[i * inputWidth + j] = (ImgMag[i * inputWidth + j] - Mag_min) / (Mag_max - Mag_min) * 255;

	// 모든 픽셀에 대해 Histogram 계산 (Integral Image)
	for (int i = 0; i < inputHeight; i++)
		for (int j = 0; j < inputWidth; j++) {
			HOG prevX, prevY, prevXY, tmp;
			if (i > 0)prevY = Hg[(i - 1) * inputWidth + j];
			if (j > 0)prevX = Hg[i * inputWidth + j - 1];
			if (i > 0 && j > 0)prevXY = Hg[(i - 1) * inputWidth + j - 1];

			tmp.d[(int)(DEG(ImgPhase[i * inputWidth + j]) / 20)] += ImgMag[i * inputWidth + j];
			Hg[i * inputWidth + j] = plusHOG(minusHOG(plusHOG(prevX, prevY), prevXY), tmp);
		}

	free(ImgMag);
	free(ImgPhase);

	// R값을 저장하기 위한 포인터 변수 선언
	double *R = (double*)malloc(sizeof(double) * inputHeight * inputWidth);

	// R값의 최대값을 구하기 위한 변수
	double max = INT_MAX;

	for (int i = 0 + (WINDOWSIZE / 2); i < inputHeight - (WINDOWSIZE / 2); i++) {
		for (int j = 0 + (WINDOWSIZE / 2); j < inputWidth - (WINDOWSIZE / 2); j++) {
			double IxIx = 0, IyIy = 0, IxIy = 0;
			
			// 윈도우 크기만큼 반복문을 돌리며 Ix, Iy 연산
			for (int y = i - (WINDOWSIZE / 2); y < i + (WINDOWSIZE / 2) + 1; y++) {
				for (int x = j - (WINDOWSIZE / 2); x < j + (WINDOWSIZE / 2) + 1; x++) {
					IxIx += Ix[y * inputWidth + x] * Ix[y * inputWidth + x];
					IxIy += Ix[y * inputWidth + x] * Iy[y * inputWidth + x];
					IyIy += Iy[y * inputWidth + x] * Iy[y * inputWidth + x];
				}
			}
			
			// det와 trace 연산
			double det = IxIx * IyIy - IxIy * IxIy;
			double trace = IxIx + IyIy;

			// R값의 포인터 변수에 연산하여 저장 및 최대값 계산
			R[i * inputWidth + j] = det - K * (trace * trace);
			max = (max < R[i * inputWidth + j]) ? R[i * inputWidth + j] : max;
		}
	}
	free(Ix);
	free(Iy);
	int cnt = 0;

	// 이미지를 반복문으로 돌며 R값이 Threshold보다 클 경우 해당 Corner의 좌표 추가
	for (int i = 0; i < inputHeight; i++)
		for (int j = 0; j < inputWidth; j++) {
			int flag = 1;

			// 각 좌표의 1픽셀 테두리를 체크하며 해당 값이 중앙의 값보다 작을때만 Corner로 취급
			for (int m = i - 1; m <= i + 1 && flag == 1; m++) {
				if (m < 0 || m >= inputHeight)
					continue;
				for (int n = j - 1; n <= j + 1 && flag == 1; n++) {
					if (j < 0 || j >= inputWidth)
						continue;
					if (i == m && j == n)
						continue;
					if (R[m * inputWidth + n] >= R[i * inputWidth + j])
						flag = 0;
				}
			}// R값이 R의 최대값 * Threshold 보다 클 때만 Corner로 취급
			if (flag == 1 && R[i * inputWidth + j] > max * THRESHOLD) {
				POS[cnt++] = Point2d(j, i);
			}
		}
	free(R);

	// Corner의 개수를 반환
	return cnt;
}

// Histogram 연산 함수
void Histogram(Mat &Img, Point2d p, HOG *Hg, int BinIndex, float *Bin) {

	int inputHeight = Img.rows;
	int inputWidth = Img.cols;

	for (int k = 0; k < 36; k++) Bin[BinIndex * 36 + k] = 0;

	// 좌표 p를 중심으로 HISTOGRAMINTV 만큼 움직여가며 blkSize 크기로 4번 Histogram 한다.
	int blkSize = HISTOGRAMSIZE + HISTOGRAMINTV;
	for (int t = 0; t < 4; t++) {
		Point2d start = Point2d(p.x - (blkSize / 2) + (t % 2) * HISTOGRAMINTV, p.y - (blkSize / 2) + (t / 2) * HISTOGRAMINTV), end = Point2d(p.x - (blkSize / 2) + (t % 2) * HISTOGRAMINTV + HISTOGRAMSIZE, p.y - (blkSize / 2) + (t / 2) * HISTOGRAMINTV + HISTOGRAMSIZE);

		// Boundary Check
		start.x = (start.x < 0) ? 0 : (start.x > inputWidth) ? (inputWidth) : start.x;
		start.y = (start.y < 0) ? 0 : (start.y > inputHeight) ? (inputHeight) : start.y;
		end.x = (end.x < 0) ? 0 : (end.x > inputWidth) ? (inputWidth) : end.x;
		end.y = (end.y < 0) ? 0 : (end.y > inputHeight) ? (inputHeight) : end.y;

		HOG prevX, prevY, prevXY, tmp;
		if (end.y - 1 > 0 && end.x - 1 > 0)tmp = Hg[(int)((end.y - 1) * inputWidth + end.x - 1)];
		if (end.y - 1 > 0 && start.x - 1 > 0)prevX = Hg[(int)((end.y - 1) * inputWidth + start.x - 1)];
		if (start.y - 1 > 0 && end.x - 1 > 0)prevY = Hg[(int)((start.y - 1) * inputWidth + end.x - 1)];
		if (start.y - 1 > 0 && start.x - 1 > 0)prevXY = Hg[(int)((start.y - 1) * inputWidth + start.x - 1)];

		tmp = plusHOG(minusHOG(tmp, plusHOG(prevX, prevY)), prevXY);

		int Index = BinIndex * 36 + t * 9;
		for (int k = 0; k < 9; k++) Bin[Index + k] += tmp.d[k];
	}
}

// VideoStabilization을 위한 함수
void VideoStabilization(Mat &prevImg, Mat &currentImg, Mat &stabilizationImg, Point2d *prevP, int *prevCont, float *prevBin, int &prevcnt, int &prevMatchCnt, Point2d &stabilization)
{
	// 처음 실행했을때만 Corner를 찾아 Histogram 한다. (그 이후부터는 현재 프레임 값을 이전 프레임에 덮어씌우기)
	if (prevcnt < 0) {
		HOG *prevHog = (HOG*)malloc(sizeof(HOG) * prevImg.cols * prevImg.rows);
		prevcnt = HarrisCornerDetect(prevImg, prevHog, prevP);
		for (int i = 0; i < prevcnt; i++) Histogram(prevImg, prevP[i], prevHog, i, prevBin);
		free(prevHog);
	}


	// Corner들의 좌표를 저장하는 포인터 변수
	Point2d *currentP = (Point2d*)malloc(sizeof(Point2d) * currentImg.cols * currentImg.rows);
	int *currentCont = (int*)malloc(sizeof(int*) * currentImg.cols * currentImg.rows);

	// 각 이미지마다 HarrisCornerDetect 함수를 포인터배열의 주소를 매개변수로 호출하여
	// 함수에서 포인터 변수에 값을 할당하며, 함수는 Corner의 개수를 반환한다.
	HOG *currentHog = (HOG*)malloc(sizeof(HOG) * currentImg.cols * currentImg.rows);

	// Corner의 개수를 저장
	int currentcnt = HarrisCornerDetect(currentImg, currentHog, currentP);

	// Histogram Bin 배열 선언 (16*16 블락을 중심좌표 기준으로 8픽셀씩 4번 이동시켜가며 Histogram, Corner 좌표 하나당 36개 배열 사용)
	float *currentBin = (float*)malloc(currentcnt * 36 * sizeof(float));


	// 각각 이미지마다 좌표의 개수만큼 반복문을 돌며 Histogram 함수 실행
	for (int i = 0; i < currentcnt; i++) Histogram(currentImg, currentP[i], currentHog, i, currentBin);
	free(currentHog);


	// Histogram이 가장 유사한 좌표를 저장하기 위한 배열
	int *prevBestMatch = (int*)malloc(prevcnt * sizeof(int));
	int *currentbestMatch = (int*)malloc(currentcnt * sizeof(int));

	// Histogram의 유사도를 계산할때 유사도의 값을 저장하기 위한 배열 (값이 작을수록 유사함)
	int *prevMin = (int*)malloc(prevcnt * sizeof(int));
	int *currentMin = (int*)malloc(currentcnt * sizeof(int));

	// 유사도 값을 비교하기 위한 최대 최소 변수
	float Rmin = INT_MAX, Rmax = INT_MIN;
	float Tmin = INT_MAX, Tmax = INT_MIN;

	// prev 이미지에서 찾은 Corner의 개수만큼 반복문을 실행
	for (int i = 0; i < prevcnt; i++) {
		prevMin[i] = INT_MAX;
		prevBestMatch[i] = -1;

		// current 이미지의 Corner 개수만큼 반복문을 돌며 비교
		for (int j = 0; j < currentcnt; j++) {
			// 각 Histogram 단계별 값을 유클리드 거리를 이용해 계산
			float avg = 0;
			for (int u = 0; u < 36; u++)
				avg += (prevBin[i * 36 + u] - currentBin[j * 36 + u]) * (prevBin[i * 36 + u] - currentBin[j * 36 + u]);

			// 유사도를 계산한 값이 최소가 되는 주소를 배열에 저장
			if (prevMin[i] > avg) {
				prevMin[i] = avg;
				prevBestMatch[i] = j;
			}
		}

		// 유사도의 최소값과 최대값을 구함
		Rmax = (Rmax < prevMin[i]) ? prevMin[i] : Rmax;
		Rmin = (Rmin > prevMin[i]) ? prevMin[i] : Rmin;
	}
	for (int i = 0; i < currentcnt; i++) {
		currentMin[i] = INT_MAX;
		currentbestMatch[i] = -1;

		// prev 이미지의 Corner 개수만큼 반복문을 돌며 비교
		for (int j = 0; j < prevcnt; j++) {
			// 각 Histogram 단계별 값을 유클리드 거리를 이용해 계산
			float avg = 0; 
			for (int u = 0; u < 36; u++)
				avg += (currentBin[i * 36 + u] - prevBin[j * 36 + u]) * (currentBin[i * 36 + u] - prevBin[j * 36 + u]);

			// 유사도를 계산한 값이 최소가 되는 주소를 배열에 저장
			if (currentMin[i] > avg) {
				currentMin[i] = avg;
				currentbestMatch[i] = j;
			}
		}

		// 유사도의 최소값과 최대값을 구함
		Tmax = (Tmax < currentMin[i]) ? currentMin[i] : Tmax;
		Tmin = (Tmin > currentMin[i]) ? currentMin[i] : Tmin;
	}

	// 유사도의 오차 값이 일정 수치 이상인 경우 Corner로 취급하지 않음 (임의로 지정)
	for (int i = 0; i < prevcnt; i++)
		if (prevMin[i] > (Rmax + Rmin) * 0.7)
			prevBestMatch[i] = -1;

	for (int i = 0; i < currentcnt; i++)
		if (currentMin[i] > (Tmax + Tmin) * 0.7)
			currentbestMatch[i] = -1;



	Mat resultImg(currentImg.rows * 2, (currentImg.cols + currentImg.cols), CV_8UC3);
	Mat GrayImg;
	cvtColor(currentImg, GrayImg, CV_RGB2GRAY);
	for (int i = 0; i < GrayImg.rows; i++)
		for (int j = 0; j < GrayImg.cols; j++)
			GrayImg.at<uchar>(i, j) /= 3;

	cvtColor(GrayImg, GrayImg, CV_GRAY2RGB);
	GrayImg.copyTo(resultImg(Rect(currentImg.cols, 0, GrayImg.cols, GrayImg.rows)));
	currentImg.copyTo(resultImg(Rect(0, 0, currentImg.cols, currentImg.rows)));


	Point2d moveP = Point2d(0, 0);
	int cnt = 0;

	int *bestmoveX = (int*)malloc(sizeof(int) * currentImg.cols * currentImg.rows);
	int *bestmoveY = (int*)malloc(sizeof(int) * currentImg.cols * currentImg.rows);
	int *bestmoveXCnt = (int*)calloc(currentImg.cols * currentImg.rows, sizeof(int));
	int *bestmoveYCnt = (int*)calloc(currentImg.cols * currentImg.rows, sizeof(int));
	int cnt2 = 0, maxX = 0, maxY = 0;


	int currentMatchCnt = 0;

	// current의 좌표와 prev 좌표의 가장 유사한 좌표가 서로 일치할 경우에 Line Drawing
	for (int i = 0; i < currentcnt; i++) {
		if (currentbestMatch[i] != -1 && prevBestMatch[currentbestMatch[i]] == i) {

			// 해당 포인트가 연속으로 여러번 연결 될 경우 초록색으로 표시
			currentCont[i] = (prevCont[currentbestMatch[i]] < 10) ? (prevCont[currentbestMatch[i]] + 1) : 10;
			if (currentCont[i] > 2) currentMatchCnt++;

			bool flag = true;
			for (int j = 0; j < cnt; j++) {
				if (bestmoveX[j] == currentP[i].x - prevP[currentbestMatch[i]].x) {
					flag = false;
					bestmoveXCnt[j] ++;
					maxX = (bestmoveXCnt[maxX] > bestmoveXCnt[j]) ? maxX : j;
				}
			}
			if (flag) {
				bestmoveX[cnt] = currentP[i].x - prevP[currentbestMatch[i]].x;
				bestmoveXCnt[cnt] = 1;
				cnt++;
			}flag = true;
			for (int j = 0; j < cnt2; j++) {
				if (bestmoveY[j] == currentP[i].y - prevP[currentbestMatch[i]].y) {
					flag = false;
					bestmoveYCnt[j] ++;
					maxY = (bestmoveYCnt[maxY] > bestmoveYCnt[j]) ? maxY : j;
				}
			}
			if (flag) {
				bestmoveY[cnt2] = currentP[i].y - prevP[currentbestMatch[i]].y;
				bestmoveYCnt[cnt2] = 1;
				cnt2++;
			}
		}
		else currentCont[i] = 0;
	}
	moveP = Point2d(bestmoveX[maxX], bestmoveY[maxY]);
	free(bestmoveX);
	free(bestmoveY);
	free(bestmoveXCnt);
	free(bestmoveYCnt);


	stabilization += moveP;
	
	//화면이 움직일때
	if (prevMatchCnt * 0.5 > currentMatchCnt) {
		stabilization.x = (STABILIZATION_MARGIN + stabilization.x) / 2;
		stabilization.y = (STABILIZATION_MARGIN + stabilization.y) / 2;
	}


	// Boundary Check
	if (stabilization.x > 2 * STABILIZATION_MARGIN)stabilization.x = 2 * STABILIZATION_MARGIN;
	if (stabilization.y > 2 * STABILIZATION_MARGIN)stabilization.y = 2 * STABILIZATION_MARGIN;
	if (stabilization.x < 0)stabilization.x = 0;
	if (stabilization.y < 0)stabilization.y = 0;


	currentImg(Rect(stabilization, Size(currentImg.cols - 2 * STABILIZATION_MARGIN, currentImg.rows - 2 * STABILIZATION_MARGIN))).copyTo(stabilizationImg);
	stabilizationImg.copyTo(resultImg(Rect(stabilization + Point2d(currentImg.cols, 0), Size(stabilizationImg.cols, stabilizationImg.rows))));
	stabilizationImg.copyTo(resultImg(Rect(Point2d(currentImg.cols / 2, currentImg.rows + STABILIZATION_MARGIN), Size(stabilizationImg.cols, stabilizationImg.rows))));
	putText(resultImg, "Original Video", Point(currentImg.cols / 2, currentImg.rows) + Point(-90, 20), 0.7, 0.7, cv::Scalar(255, 255, 255));
	putText(resultImg, "Stabilization Rectangle", Point(currentImg.cols / 2, currentImg.rows) + Point(currentImg.cols, 0) + Point(-120, 20), 0.7, 0.7, cv::Scalar(255, 255, 255));
	putText(resultImg, "Stabilized Video", Point(currentImg.cols, currentImg.rows * 2) + Point(-120, -10), 0.7, 0.7, cv::Scalar(255, 255, 255));
	rectangle(resultImg, stabilization + Point2d(currentImg.cols, 0), stabilization + Point2d(currentImg.cols, 0) + Point2d(stabilizationImg.cols, stabilizationImg.rows), Scalar(0, 255, 0), 2, 8, 0);


	// 현재 프레임의 값을 이전 프레임 배열에 복사
	prevcnt = currentcnt;
	prevMatchCnt = currentMatchCnt;
	for (int i = 0; i < prevcnt; i++) {
		prevP[i] = currentP[i];
		prevCont[i] = currentCont[i];
	}for (int i = 0; i < prevcnt * 36; i++) prevBin[i] = currentBin[i];

	free(currentCont);
	free(currentP);
	free(currentBin);
	free(prevBestMatch);
	free(currentbestMatch);
	free(prevMin);
	free(currentMin);

	stabilizationImg = resultImg;
}
int main(int argc, const char* argv[]){

	String FilePath;

	if (argc > 1)
		FilePath = argv[1];

	VideoCapture capture(FilePath);
	Mat frame, prevframe;
	capture >> prevframe;

	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);


	int *prevCont = (int*)calloc(prevframe.cols * prevframe.rows, sizeof(int*));
	Point2d *prevP = (Point2d*)malloc(sizeof(Point2d*) * prevframe.cols * prevframe.rows);
	float *prevBin = (float*)malloc(sizeof(Point2d*) * prevframe.cols * prevframe.rows * 36);
	int prevcnt = -1, prevMatchCnt = 0;
	Point2d stabilization = Point2d(STABILIZATION_MARGIN, STABILIZATION_MARGIN);
	while (capture.isOpened()) {


		capture >> frame;
		if (frame.empty())
			break;
		Mat outframe;
		VideoStabilization(prevframe, frame, outframe, prevP, prevCont, prevBin, prevcnt, prevMatchCnt, stabilization);
		cv::imshow("Video Stabilization", outframe);


		frame.copyTo(prevframe);
		int key = waitKey(1);

		// ESC to break
		if (key == 27) break;
	}
	capture.release();
	destroyAllWindows();
	free(prevP);
	free(prevBin);
	free(prevCont);
	return 0;
}
