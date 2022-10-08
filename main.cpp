// #include <Eigen/Dense>
// #include <Eigen/Core>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include<cmath>
#include <iostream>

// #include "opencv2/core/eigen.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/ml.hpp>
using namespace std;
using namespace cv;

Mat img; Mat crops(0, 0, CV_8UC1);
Mat old_frame, old_gray, new_frame, new_gray;
Mat cropsRef(0, 0, CV_8UC1);
int errno;

void getdir(string dir, vector<string> &files);
Mat img2cropsRef(Mat cropsRef, Mat crops, Mat img, int window_w, int window_h, int w, int h, int inital_w, int inital_h, int NumOfCrops, int NumElementsCrop, int XrefC, int YrefC);

Mat img2cropsTrack(Mat crops, Mat img, int window_w, int window_h, int w, int h, int inital_w, int inital_h, int NumOfCrops, int NumElementsCrop);

void DoPca(const Mat &_data, int dim, Mat &eigenvalues, Mat &eigenvectors);  

int main(){
    int x1,y1,x2,y2;
    int x1_org,y1_org,x2_org,y2_org;
    int window_w,window_h,h,w;
    int NumElementsCrop,NumOfCrops;
    int inital_w; int inital_h;
    int XrefC,YrefC;
    x1=70; y1=125; x2=110; y2=200; // divid 2
    x1_org=x1; x2_org=x2; y1_org=y1; y2_org=y2;
    window_w = x2-x1; window_h = y2-y1;
    XrefC = (x2+x1)/2; YrefC = (y2+y1)/2;

    string dir = string("Kangaroo");
    vector<string> files = vector<string>();
    getdir(dir, files);
    int filenum = sizeof(files);

    // Create some random colors
    vector<Scalar> colors; RNG rng;
    for(int i = 0; i < 100; i++){
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));}

    vector<Point2f> p0, p1;
    // Take first frame and find corners in it
    old_frame = imread(dir + "\\" + files[2]); // ref frame
    



    h=old_frame.cols; w=old_frame.rows; // frame shape
    inital_w = w; inital_h = h;
    NumOfCrops = (w-window_w-1)*(h-window_h-1);
    NumElementsCrop = window_w*window_h*3; // 3: RGB
    // cropping
    crops = img2cropsRef(cropsRef,crops, old_frame, window_w, window_h, w, h, inital_w, inital_h, NumOfCrops, NumElementsCrop, XrefC,YrefC); // crops size: (33916,[1,57600])
    int dataNum = crops.cols; int elementNum =  crops.rows; // 33916 57600
    // cout << "finish cropping" << endl;
    cout << " Crops number and elements: "<< dataNum << "," << elementNum << endl;
    // for(int i=0;i<dataNum;i++){
    //     cropsRef.push_back(cropRef);}
    // cout << cropsRef.cols << ","<< cropsRef.rows << endl;


    // --------------------- P  C  A --------------------- //
    // PCA pca(crops, noArray(), CV_PCA_DATA_AS_ROW, 128); // 128 features each
    // Mat eigenvalues_ref, eigenvectors_ref;
    // DoPca(crops, 3, eigenvalues_ref, eigenvectors_ref); 
    // cout << "eigenvalues:" << eigenvalues_ref << endl; 

    // -------------------  Track loop ------------------- //
    for(int frame_index = 3; frame_index<filenum; frame_index++){
        Mat track_crops(0, 0, CV_8UC1); Mat errorMap, error1D;
        double minVal; double maxVal; 
        int track_x; int track_y;
        Point minLoc; Point maxLoc;

        new_frame = imread(dir + "\\" + files[frame_index]);
        h=new_frame.cols; w=new_frame.rows; // frame shape
        inital_w = w; inital_h = h;
        NumOfCrops = (w-window_w-1)*(h-window_h-1);
        NumElementsCrop = window_w*window_h*3; // 3: RGB
        // cropping
        track_crops = img2cropsTrack(track_crops, new_frame, window_w, window_h, w, h, inital_w, inital_h, NumOfCrops, NumElementsCrop);
        dataNum = track_crops.cols; elementNum =  track_crops.rows;
        // cout << " Crops number and elements: "<< dataNum << "," << elementNum << endl; 
        errorMap = track_crops-cropsRef; // 57600 x 33916 (Map) 
        // ==> 57600 x 1 (1D)
        cv::reduce(errorMap, error1D, 0, CV_REDUCE_SUM, CV_32S); 
        minMaxLoc( error1D, &minVal, &maxVal, &minLoc, &maxLoc );
        track_x = (minLoc.x/(h-window_w-1))+(window_w/2);
        track_y = (minLoc.x/(w-window_h-1))+(window_h/2);
        cout << "Frame Index:" << frame_index-2 << "  -  " << track_x <<","<<track_y << endl;
        rectangle(new_frame,cvPoint(track_x-window_w,track_y-window_h),cvPoint(track_x+window_w,track_y+window_h),Scalar(255,0,0),1,1,0);
        imshow("Track result",new_frame);
        waitKey(20);
    }

    return 0;
}

void getdir(string dir, vector<string> &files){
    DIR *dp;//create pointer of dir
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL){
        cout << "Error(" << errno << ") opening " << dir << endl;
    }
    while((dirp = readdir(dp)) != NULL){//if dirent's pointer is not empty
        files.push_back(string(dirp->d_name));//putting the names in vector
    }
    closedir(dp);//close the pointer
}

Mat img2cropsRef(Mat cropsRef,Mat crops, Mat img, int window_w, int window_h, int w, int h, int inital_w, int inital_h, int NumOfCrops, int NumElementsCrop, int XrefC, int YrefC){
    Mat resized_C,resized_CRef,crop;
    int cx1Ref = XrefC; int cx2Ref = XrefC+window_w;
    int cy1Ref = YrefC; int cy2Ref = YrefC+window_h;   
    Mat refCrop = img(Range(cy1Ref,cy2Ref), Range(cx1Ref,cx2Ref)); 
    cvtColor(refCrop, refCrop, CV_BGR2GRAY);
    resize(refCrop,resized_CRef,Size(inital_w,inital_h),0,0,INTER_LINEAR);
    resized_CRef = resized_CRef.reshape(1,1);
    for(int x=0;x<(h-window_w-1);x++){ 
        for(int y=0;y<(w-window_h-1);y++){ 
            int cx1 = x; int cx2 = x+window_w;
            int cy1 = y; int cy2 = y+window_h;
            crop = img(Range(cy1,cy2), Range(cx1,cx2)); 
            cvtColor(crop, crop, CV_BGR2GRAY);
            resize(crop,resized_C,Size(inital_w,inital_h),0,0,INTER_LINEAR);
            resized_C = resized_C.reshape(1,1);
            crops.push_back(resized_C);
            cropsRef.push_back(resized_CRef);
        }} 
    return crops;}

Mat img2cropsTrack(Mat crops, Mat img, int window_w, int window_h, int w, int h, int inital_w, int inital_h, int NumOfCrops, int NumElementsCrop){
    Mat resized_C,crop;
    for(int x=0;x<(h-window_w-1);x++){ 
        for(int y=0;y<(w-window_h-1);y++){ 
            int cx1 = x; int cx2 = x+window_w;
            int cy1 = y; int cy2 = y+window_h;
            crop = img(Range(cy1,cy2), Range(cx1,cx2)); 
            cvtColor(crop, crop, CV_BGR2GRAY);
            resize(crop,resized_C,Size(inital_w,inital_h),0,0,INTER_LINEAR);
            resized_C = resized_C.reshape(1,1);
            crops.push_back(resized_C);
        }} 
    return crops;
}
void DoPca(const Mat &_data, int dim, Mat &eigenvalues, Mat &eigenvectors){  
    assert( dim>0 );  
    Mat data =  cv::Mat_<double>(_data);  
    int R = data.rows; int C = data.cols;  

    if ( dim>C )  
        dim = C;  
    Mat m = Mat::zeros( 1, C, data.type() );  
    for ( int j=0; j<C; j++ ) {  
        for ( int i=0; i<R; i++ ) {  
            m.at<double>(0,j) += data.at<double>(i,j);  }}  
  
    m = m/R;    
    Mat S =  Mat::zeros( R, C, data.type() );  
    for ( int i=0; i<R; i++ )  {  
        for ( int j=0; j<C; j++ ) {  
            S.at<double>(i,j) = data.at<double>(i,j) - m.at<double>(0,j); }}  
    Mat Average = S.t() * S /(R);  
    eigen(Average, eigenvalues, eigenvectors);}

