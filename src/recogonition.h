#ifndef RECOGONITION_H
#define RECOGONITION_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
#include <fstream>
#include "tools.h"
#include "ekho.h"

class Recogonition
{
public:
    Recogonition();
    Recogonition(const std::string& trainPath);
    void getFeature(const cv::Mat& roiImg, cv::Mat& features);
    void featureCluster(const std::vector<cv::Mat> features, int centerNu, cv::Mat& centers);

    void getRegion(const string& src, vector<cv::Mat>& regions);
    void generateClassName(const string& classPath, std::map<int, string>& className);
    void getClassName();
    void initDictionary();
    void bowTrain();
    void predict(const string& src);

private:
    std::string trainPath;
    static const string dictFile;
    static const string svmFile;
    static const string classFile;

    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::FeatureDetector> detector;

    cv::Ptr<cv::BOWKMeansTrainer> bTrain;
    cv::Ptr<cv::BOWImgDescriptorExtractor> bowDE;
    const int dictSize = 3500;
    cv::Mat dictionary;

    CvSVMParams params;
    CvSVM svm;
    map<int, string> className;
};

#endif // RECOGONITION_H
