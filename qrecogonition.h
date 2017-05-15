#ifndef QRECOGONITION_H
#define QRECOGONITION_H

#include <QThread>
#include <QImage>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
#include <fstream>
#include "src/tools.h"
#include "ekho.h"

namespace des {

enum DESCIPTOR_METHOD{
    SIFT = 1,
    SURF,
    BRIEF,
    BRISK,
    ORB,
    FREAK,
    OpponentSIFT //OpponentColorDescriptorExtractor
};
}

namespace fea {

enum FEATURE_METHOD{
    SIFT = 1, //SIFT (nonfree module)
    SURF, //SURF (nonfree module)
    BRISK, //BRISK
    ORB, //ORB
    FAST, //FastFeatureDetector
    STAR, //StarFeatureDetector
    MSER, //MSER
    GFTT, //GoodFeaturesToTrackDetector
    HARRIS, //GoodFeaturesToTrackDetector with Harris detector enabled
    Dense, //DenseFeatureDetector
    SimpleBlob, //SimpleBlobDetector
    GridFAST, //GridAdaptedFeatureDetector
    PyramidSTAR //PyramidAdaptedFeatureDetector
};
}


enum MATCH_METHOD{
    FlannBased = 1,
    BruteForce,
    BruteForceL1,
    BruteForceHamming1,
    BruteForceHamming2
};

enum CLASSIFIER{
    SVM = 1,
    BOOST,
    KNN,
    ANN
};


class QRecogonition : public QThread
{
    Q_OBJECT

public:
    //QRecogonition();
    QRecogonition(des::DESCIPTOR_METHOD featureMethod = des::SIFT, MATCH_METHOD matchMethod = MATCH_METHOD::FlannBased, CLASSIFIER classifier = CLASSIFIER::SVM);
    QRecogonition(const std::string& trainPath, des::DESCIPTOR_METHOD featureMethod = des::SIFT, MATCH_METHOD matchMethod = MATCH_METHOD::FlannBased, CLASSIFIER classifier = CLASSIFIER::SVM);


    void setTrainPath(const std::string& trainPath);
    void getDistortContour(const cv::Mat& img, std::vector< std::vector< cv::Point > >& distortContours);
    void getCorrectContour(const cv::Mat& img, std::vector< std::vector< cv::Point > >& distortContours, std::vector<std::vector<cv::Point> > &detectedSigns);
    void getRegion(const string& src, vector<cv::Mat>& regions);
    bool getClassName();
    bool initDictionary();
    bool bowTrain();
    void setPredictImg(const string& pImg);
    bool predict();

    void trainAll();

private:
    void init(des::DESCIPTOR_METHOD featureMethod, MATCH_METHOD matchMethod);

signals:
        void sendStatues(const QString& message);
        void showStatues(const QImage& img);

protected:
        void run();

        QImage cvMat2QImage(const cv::Mat& src)
        {
            cv::Mat temp;
            if (src.type() == CV_8UC1)
            {
                cv::cvtColor(src, temp, CV_GRAY2RGB);
            }else{
                cv::cvtColor(src, temp, CV_BGR2RGB);
            }
            //cv::cvtColor(src, temp,CV_BGR2RGB);
            QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
            dest.bits();
            //QImage::QImage ( const uchar * data, int width, int height, Format format )
            return dest;
        }

private:
    std::string trainPath;
    static const string dictFile;
    static const string svmFile;
    static const string boostFile;
    static const string knnFile;
    static const string annFile;
    static const string classFile;
    static const std::string logPath;
    std::string pImg;

    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::FeatureDetector> detector;

    cv::Ptr<cv::BOWKMeansTrainer> bTrain;
    cv::Ptr<cv::BOWImgDescriptorExtractor> bowDE;
    static const int dictSize = 1500;
    cv::Mat dictionary;

    CLASSIFIER classifier;

    CvSVMParams svmParams;
    CvSVM svm;

    CvGBTrees boost;
    CvGBTreesParams gbParams;

    CvKNearest knn;
    int k;

    CvANN_MLP ann;
    CvANN_MLP_TrainParams annParams;
    cv::Mat layer;

    std::map<int, std::string> className;
};

#endif // QRECOGONITION_H
