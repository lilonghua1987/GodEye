#include "qrecogonition.h"

#include <chrono>
#include <ctime>

// Eigen library
#include <Eigen/Core>

#include "src/segmentation.h"
#include "src/colorConversion.h"
#include "src/imageProcessing.h"
#include "src/smartOptimisation.h"
#include "src/math_utils.h"

const string QRecogonition::dictFile =  "config/dictionary.xml";
const string QRecogonition::svmFile = "config/classesSVM.xml";
const string QRecogonition::boostFile = "config/classesBoost.xml";
const string QRecogonition::knnFile = "config/classesKnn.xml";
const string QRecogonition::annFile = "config/classesAnn.xml";
const string QRecogonition::classFile = "config/className.txt";
const string QRecogonition::logPath = "log/";


QRecogonition::QRecogonition(des::DESCIPTOR_METHOD featureMethod, MATCH_METHOD matchMethod, CLASSIFIER classifier)
    :classifier(classifier)
{
    //moveToThread(this);
    //cv::initModule_nonfree();//if use SIFT or SURF
    //matcher = cv::DescriptorMatcher::create("FlannBased");
//    matcher = cv::DescriptorMatcher::create("BruteForce");
//    extractor = cv::DescriptorExtractor::create("SIFT");
//    detector = cv::FeatureDetector::create("SIFT");

    init(featureMethod, matchMethod);
}

QRecogonition::QRecogonition(const string &trainPath, des::DESCIPTOR_METHOD featureMethod, MATCH_METHOD matchMethod, CLASSIFIER classifier)
    :trainPath(trainPath),
      classifier(classifier)
{
    //moveToThread(this);
//    cv::initModule_nonfree();//if use SIFT or SURF
//    matcher = cv::DescriptorMatcher::create("FlannBased");
//    extractor = cv::DescriptorExtractor::create("SIFT");
//    detector = cv::FeatureDetector::create("SIFT");

    init(featureMethod, matchMethod);
}


void QRecogonition::init(des::DESCIPTOR_METHOD featureMethod, MATCH_METHOD matchMethod)
{
    std::string featureName, descriptorName;
    switch (featureMethod) {
    case des::SIFT:
        cv::initModule_nonfree();//if use SIFT or SURF
        featureName = "SIFT";
        //descriptorName = "SIFT";
        descriptorName = "OpponentSIFT";
        break;
    case des::SURF:
        cv::initModule_nonfree();//if use SIFT or SURF
        featureName = "SURF";
        descriptorName = "SURF";
        break;
    case des::BRIEF:
        featureName = "FAST";
        descriptorName = "BRIEF";
        break;
    case des::BRISK:
        featureName = "BRISK";
        descriptorName = "BRISK";
        break;
    case des::ORB:
        featureName = "ORB";
        descriptorName = "ORB";
        break;
    case des::FREAK:
        featureName = "FREAK";
        descriptorName = "FREAK";
        break;
    default:
        break;
    }

    extractor = cv::DescriptorExtractor::create(descriptorName);
    detector = cv::FeatureDetector::create(featureName);

    std::cout << "initial featureDetector" << std::endl;

    switch (matchMethod) {
    case MATCH_METHOD::FlannBased:
        matcher = cv::DescriptorMatcher::create("FlannBased");
        break;
    case MATCH_METHOD::BruteForce:
        matcher = cv::DescriptorMatcher::create("BruteForce");
        break;
    case MATCH_METHOD::BruteForceL1:
        matcher = cv::DescriptorMatcher::create("BruteForce-L1");
        break;
    case MATCH_METHOD::BruteForceHamming1:
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        break;
    case MATCH_METHOD::BruteForceHamming2:
        matcher = cv::DescriptorMatcher::create("BruteForce-HammingLUT");
        break;
    default:
        break;
    }

    cout << featureMethod << " , " << matchMethod << " , " << classifier << endl;
}


void QRecogonition::setTrainPath(const string &trainPath)
{
    this->trainPath = trainPath;
}

void QRecogonition::getDistortContour(const cv::Mat& img, std::vector<std::vector<cv::Point> > &distortContours)
{
    cv::Mat input_image = img.clone();

    if (!input_image.data) {
        std::cout << "Error to read the image. Check ''cv::imread'' function of OpenCV" << std::endl;
        return;
    }

    CV_Assert(input_image.channels() == 3);

    cv::Mat bImg;
    //cv::bilateralFilter ( input_image, bImg, 3, 3*2, 2/3 );
    cv::adaptiveBilateralFilter(input_image,bImg,cv::Size(3,3),0.5);
    input_image = bImg;
    imwrite(logPath + "filterImg.jpg",bImg);
    emit showStatues(cvMat2QImage(bImg));

    // Conversion of the rgb image in ihls color space
    cv::Mat ihls_image;
    colorconversion::convert_rgb_to_ihls(input_image, ihls_image);
    // Conversion from RGB to logarithmic chromatic red and blue
    std::vector< cv::Mat > log_image;
    colorconversion::rgb_to_log_rb(input_image, log_image);

    cv::Mat log_image_seg;
    segmentation::seg_log_chromatic(log_image, log_image_seg);

    for (char i = 0; i <= 0; ++i){
        cv::Mat nhs_image_seg_red;
        cv::Mat nhs_image_seg_blue;
        if ( i == 0){
            segmentation::seg_norm_hue(ihls_image, nhs_image_seg_red, i);
            nhs_image_seg_blue = nhs_image_seg_red.clone();
        }else{
            segmentation::seg_norm_hue(ihls_image, nhs_image_seg_blue, i);
           // nhs_image_seg_red = nhs_image_seg_blue.clone();
            segmentation::seg_norm_hue(ihls_image, nhs_image_seg_red, 0);
        }

        cv::Mat merge_image_seg_with_red = nhs_image_seg_red.clone();
        cv::Mat merge_image_seg = nhs_image_seg_blue.clone();
        cv::bitwise_or(nhs_image_seg_red, log_image_seg, merge_image_seg_with_red);
        cv::bitwise_or(nhs_image_seg_blue, merge_image_seg_with_red, merge_image_seg);

         // Filter the image using median filtering and morpho math
         cv::Mat bin_image;
         imageprocessing::filter_image(merge_image_seg, bin_image);

         switch (i) {
         case 0:
             cv::imwrite(logPath + "red_unf_seg.jpg",merge_image_seg);
             emit showStatues(cvMat2QImage(merge_image_seg));
             cv::imwrite(logPath + "red_seg.jpg", bin_image);
             emit showStatues(cvMat2QImage(bin_image));
             break;
         default:
             cv::imwrite(logPath + "blue_unf_seg.jpg",merge_image_seg);
             emit showStatues(cvMat2QImage(merge_image_seg));
             cv::imwrite(logPath + "blue_seg.jpg", bin_image);
             emit showStatues(cvMat2QImage(bin_image));
             break;
         }

        std::vector< std::vector< cv::Point > > distortedContours;
        imageprocessing::contours_extraction(bin_image, distortedContours);

        for (auto& contour : distortedContours){
            distortContours.push_back(contour);
        }
    }
}

void QRecogonition::getCorrectContour(const cv::Mat& img, std::vector<std::vector<cv::Point> > &distortContours, std::vector<std::vector<cv::Point> > &detectedSigns)
{
    // Initialisation of the variables which will be returned after the distortion. These variables are linked with the transformation applied to correct the distortion
    std::vector< cv::Mat > rotationMatrix(distortContours.size());
    std::vector< cv::Mat > scalingMatrix(distortContours.size());
    std::vector< cv::Mat > translationMatrix(distortContours.size());
    for (unsigned int contour_idx = 0; contour_idx < distortContours.size(); contour_idx++) {
        rotationMatrix[contour_idx] = cv::Mat::eye(3, 3, CV_64F);
        scalingMatrix[contour_idx] = cv::Mat::eye(3, 3, CV_64F);
        translationMatrix[contour_idx] = cv::Mat::eye(3, 3, CV_64F);
    }

    // Correct the distortion
    std::vector< std::vector< cv::Point2f > > undistortContours;
    imageprocessing::correction_distortion(distortContours, undistortContours, translationMatrix, rotationMatrix, scalingMatrix);

    // Normalise the contours to be inside a unit circle
    std::vector<double> factorVector(undistortContours.size());
    std::vector< std::vector< cv::Point2f > > normalisedContours;
    initoptimisation::normalise_all_contours(undistortContours, normalisedContours, factorVector);

    std::vector< std::vector< cv::Point2f > > detected_signs_2f(normalisedContours.size());
    detectedSigns = std::vector< std::vector< cv::Point > >(normalisedContours.size());


    // For each contours
    for (unsigned int contour_idx = 0; contour_idx < normalisedContours.size(); contour_idx++) {


        // For each type of traffic sign
        /*
     * sign_type = 0 -> nb_edges = 3;  gielis_sym = 6; radius
     * sign_type = 1 -> nb_edges = 4;  gielis_sym = 4; radius
     * sign_type = 2 -> nb_edges = 12; gielis_sym = 4; radius
     * sign_type = 3 -> nb_edges = 8;  gielis_sym = 8; radius
     * sign_type = 4 -> nb_edges = 3;  gielis_sym = 6; radius / 2
     */

        optimisation::ConfigStruct_<double> final_config;
        double best_fit = std::numeric_limits<double>::infinity();
        int type_sign_to_keep = 0;
        for (int sign_type = 0; sign_type < 5; sign_type++) {

            // Check the center mass for a contour
            cv::Point2f mass_center = initoptimisation::mass_center_discovery(img, translationMatrix[contour_idx], rotationMatrix[contour_idx], scalingMatrix[contour_idx], normalisedContours[contour_idx], factorVector[contour_idx], sign_type);

            // Find the rotation offset
            double rot_offset = initoptimisation::rotation_offset(normalisedContours[contour_idx]);

            // Declaration of the parameters of the gielis with the default parameters
            optimisation::ConfigStruct_<double> contour_config;
            // Set the number of symmetry
            int gielis_symmetry = 0;
            switch (sign_type) {
            case 0:
                gielis_symmetry = 6;
                break;
            case 1:
                gielis_symmetry = 4;
                break;
            case 2:
                gielis_symmetry = 4;
                break;
            case 3:
                gielis_symmetry = 8;
                break;
            case 4:
                gielis_symmetry = 6;
                break;
            }
            contour_config.p = gielis_symmetry;
            // Set the rotation matrix
            contour_config.theta_offset = rot_offset;
            // Set the mass center
            contour_config.x_offset = mass_center.x;
            contour_config.y_offset = mass_center.y;

            // Go for the optimisation
            Eigen::Vector4d mean_err(0,0,0,0), std_err(0,0,0,0);
            optimisation::gielis_optimisation(normalisedContours[contour_idx], contour_config, mean_err, std_err);

            mean_err = mean_err.cwiseAbs();
            double err_fit = mean_err.sum();

            if (err_fit < best_fit) {
                best_fit = err_fit;
                final_config = contour_config;
                type_sign_to_keep = sign_type;
            }
        }

        // Reconstruct the contour
        std::cout << "Contour #" << contour_idx << ":\n" << final_config << std::endl;
        std::vector< cv::Point2f > gielis_contour;
        int nb_points = 1000;
        optimisation::gielis_reconstruction(final_config, gielis_contour, nb_points);
        std::vector< cv::Point2f > denormalised_gielis_contour;
        initoptimisation::denormalise_contour(gielis_contour, denormalised_gielis_contour, factorVector[contour_idx]);
        std::vector< cv::Point2f > distorted_gielis_contour;
        imageprocessing::inverse_transformation_contour(denormalised_gielis_contour, distorted_gielis_contour, translationMatrix[contour_idx], rotationMatrix[contour_idx], scalingMatrix[contour_idx]);

        // Transform to cv::Point to show the results
        std::vector< cv::Point > distorted_gielis_contour_int(distorted_gielis_contour.size());
        for (unsigned int i = 0; i < distorted_gielis_contour.size(); i++) {
            distorted_gielis_contour_int[i].x = (int) std::round(distorted_gielis_contour[i].x);
            distorted_gielis_contour_int[i].y = (int) std::round(distorted_gielis_contour[i].y);
        }

        detected_signs_2f[contour_idx] = distorted_gielis_contour;
        detectedSigns[contour_idx] = distorted_gielis_contour_int;

    }
}


void QRecogonition::getRegion(const string &src, vector<cv::Mat> &regions)
{
    cv::Mat img = cv::imread(src);
    cv::Mat rImg = img.clone();

    std::vector< std::vector< cv::Point > > distortContours;
    getDistortContour(rImg,distortContours);

    std::vector<std::vector<cv::Point> > detectedSigns;
    getCorrectContour(rImg, distortContours, detectedSigns);

    cv::Scalar color(0,255,0);
    cv::drawContours(rImg, detectedSigns, -1, color, 2, 8);
    imwrite(logPath + "result.jpg",rImg);
    emit showStatues(cvMat2QImage(rImg));

    for (auto& contour : detectedSigns){
        cv::Rect rect = cv::boundingRect(cv::Mat(contour));

        if ( (rect.x + rect.width) >= img.cols || (rect.y + rect.height) >= img.rows )
        {
            continue;
        }

        cv::Mat reg = img(rect);
        regions.push_back(reg);
        cv::Scalar color(0,255,0);
        cv::rectangle(rImg,rect,color);
    }

    imwrite(logPath + "result_rect.jpg",rImg);
    emit showStatues(cvMat2QImage(rImg));
}


bool QRecogonition::getClassName()
{
    ifstream fs;
    fs.open(classFile, ios::in);
    if (fs.is_open()){
        string line;
        while (getline(fs,line)){
            int pos = line.find_first_of("=");
            if (pos <= 0 ){
                continue;
            }else{
                int label = std::atoi(line.substr(0, pos).c_str());
                className.insert(pair<int, string>(label, line.substr(pos + 1, line.length() - pos)));
            }
        }

        if (className.size() > 0) {
            return true;
        }else{
            return false;
        }

    }else{
        return false;
    }
}


bool QRecogonition::initDictionary()
{
    emit sendStatues(QStringLiteral("InitDictionary begin ............ !"));

    cv::TermCriteria tc(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.001);

    bTrain = new cv::BOWKMeansTrainer(dictSize, tc, 1, cv::KMEANS_PP_CENTERS);
    bowDE = new cv::BOWImgDescriptorExtractor(extractor, matcher);

    cv::FileStorage dFs(dictFile, cv::FileStorage::READ);
    if (dFs.isOpened()){
        dFs["dictionary"] >> dictionary;
        dFs.release();
    }else{
        std::vector<std::string> imgList;
        tools::listFileByDir(trainPath,"jpg",imgList);

        std::cout << "imgList size = " << imgList.size() << std::endl;

        for (auto& file : imgList){
            cv::Mat img = cv::imread(file,1);
            //std::cout << " image channels : " << img.channels() << std::endl;
            //cv::Mat img = cv::imread(file,0);
//            if (img.data && img.channels() == 1){
//                std::cout << "Read img success" << std::endl;
//            }
            std::vector<cv::KeyPoint> keyPoint;
            detector->detect(img, keyPoint);
            std::cout << "Image keyPoint size = " << keyPoint.size() << std::endl;
            if ( keyPoint.size() <= 0 ){
                continue;
            }
            cv::Mat feature;
            extractor->compute(img, keyPoint, feature);
            if ( feature.empty()){
                continue;
            }
            bTrain->add(feature);
        }

        if (bTrain->descripotorsCount() < 2){
            return false;
        }

        dictionary = bTrain->cluster();
        cv::FileStorage fs(dictFile,cv:: FileStorage::WRITE);
        fs << "dictionary" << dictionary;
        fs.release();
    }

    if (dictionary.rows > 0 && dictionary.cols > 0){
        bowDE->setVocabulary(dictionary);
        return true;
    }else{
        return false;
    }
}


bool QRecogonition::bowTrain()
{
    emit sendStatues(QStringLiteral("Training ...... !"));

    cv::Mat labels(0, 1, CV_32FC1);
    cv::Mat trainData(0, dictSize, CV_32FC1);
    std::set<std::string> classNu;

    std::vector<std::string> imgList;
    tools::listFileByDir(trainPath,"jpg",imgList);

    for (auto& file : imgList){
        cv::Mat img = cv::imread(file,1);
        std::vector<cv::KeyPoint> keyPoint;
        detector->detect(img, keyPoint);
        if ( keyPoint.size() <= 0 ){
            continue;
        }
        cv::Mat feature;
        bowDE->compute(img, keyPoint, feature);
        trainData.push_back(feature);
        labels.push_back((static_cast<float>(std::atoi(tools::fileNamePart(file).substr(0,5).c_str()))));
        classNu.insert(tools::fileNamePart(file).substr(0,5));
    }

    bool flag = false;

    if (CLASSIFIER::SVM == classifier){
        svmParams.kernel_type = CvSVM::RBF;
        svmParams.svm_type = CvSVM::C_SVC;
        svmParams.gamma = 0.50625000000000000000009;
        svmParams.C = 312.500000000000000;
        svmParams.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001);

        if (svm.train(trainData, labels, cv::Mat(), cv::Mat(), svmParams)){
            svm.save(svmFile.c_str());
            flag = true;
        }
    }else if (CLASSIFIER::BOOST == classifier){
        gbParams.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
        gbParams.weak_count = 100;
        gbParams.shrinkage = 0.1f;
        gbParams.subsample_portion = 1.0f;
        gbParams.max_depth = 2;
        gbParams.use_surrogates = false;

        cv::Mat var_types( 1, trainData.cols + 1, CV_8UC1, cv::Scalar(CV_VAR_ORDERED) );
        var_types.at<uchar>( trainData.cols ) = CV_VAR_CATEGORICAL;

        if (boost.train(trainData,CV_ROW_SAMPLE,labels, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), gbParams)){
            boost.save(boostFile.c_str());
            flag = true;
        }
    }else if (CLASSIFIER::KNN == classifier){
        std::cout << "inital knn params" << std::endl;
        k = 55;
        if (knn.train(trainData,labels,cv::Mat(),false,k)){
            std::cout << "train knn" << std::endl;
            knn.save(knnFile.c_str());
            flag = true;
        }
    }else if (CLASSIFIER::ANN == classifier){
        std::cout << "inital ann params" << std::endl;
        cv::Mat trainClass(labels.rows, classNu.size(), CV_32FC1);

        for (auto i = 0; i < trainClass.rows; ++i){
            for (auto j = 0; j < trainClass.cols; ++j){
                if (labels.at<float>(i,0) == (j + 1)){
                    trainClass.at<float>(i,j) = 1;
                }else{
                    trainClass.at<float>(i,j) = 0;
                }
            }
        }

        std::cout << "total classes : " << classNu.size() << std::endl;

        layer.create(1, 3, CV_32SC1);
        layer.at<int>(0) = trainData.cols;
        layer.at<int>(1) = 35;
        layer.at<int>(2) = classNu.size();

        cv::Mat weights( 1, trainData.cols, CV_32FC1);

        CvTermCriteria criteria;
        criteria.max_iter = 10000;
        criteria.epsilon  = 0.000001;
        criteria.type     = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;

        annParams.train_method = CvANN_MLP_TrainParams::BACKPROP;
        annParams.bp_dw_scale = 0.1;
        annParams.bp_moment_scale = 0.1;
        annParams.term_crit = criteria;
        //annParams.train_method=CvANN_MLP_TrainParams::RPROP;
        //annParams.rp_dw0 = 0.1;
        //annParams.rp_dw_plus = 1.2;
        //annParams.rp_dw_minus = 0.5;
        //annParams.rp_dw_min = FLT_EPSILON;
        //annParams.rp_dw_max = 50.;

        std::cout << "create and train ann" << std::endl;
        ann.create(layer, CvANN_MLP::SIGMOID_SYM);
//        if (ann.train(trainData,trainClass,weights)){
//            ann.save(annFile.c_str());
//            flag = true;
//        }
        if (ann.train(trainData,trainClass,cv::Mat(), cv::Mat(), annParams)){
            ann.save(annFile.c_str());
            flag = true;
        }
    }

    return flag;
}


void QRecogonition::setPredictImg(const string &pImg)
{
    this->pImg = pImg;
}


bool QRecogonition::predict()
{
    cv::FileStorage trainFile;
    bool flag = false;

    if (CLASSIFIER::SVM == classifier){
        trainFile.open(svmFile,cv::FileStorage::READ);

        if (trainFile.isOpened()){
            trainFile.release();
            svm.load(svmFile.c_str());
            flag = true;
        }
    }else if (CLASSIFIER::BOOST == classifier){
        trainFile.open(boostFile,cv::FileStorage::READ);

        if (trainFile.isOpened()){
            trainFile.release();
            boost.load(boostFile.c_str());
            flag = true;
        }
    }else if (CLASSIFIER::KNN == classifier){
        trainFile.open(knnFile,cv::FileStorage::READ);

        if (trainFile.isOpened()){
            trainFile.release();
            knn.load(knnFile.c_str());
            flag = true;
        }
    }else if (CLASSIFIER::ANN == classifier){
        trainFile.open(annFile,cv::FileStorage::READ);

        if (trainFile.isOpened()){
            trainFile.release();
            ann.load(annFile.c_str());
            flag = true;
        }
    }

    if (!flag){
        if (!bowTrain()){
            return false;
        }
    }

    if (!getClassName()){
        return false;
    }

    emit sendStatues(QStringLiteral("Get the ROI regions ....!"));

    std::vector<cv::Mat> roiImg;
    getRegion(pImg, roiImg);

    emit sendStatues(QStringLiteral("Predict the ROI region ....!"));

    ekho::Ekho say("Mandarin");
    int count = 0;
    for (auto& img : roiImg)
    {
        cv::Mat gImg;
        cv::cvtColor(img, gImg, CV_BGR2BGRA);
        std::vector<cv::KeyPoint> keyPoint;
        detector->detect(img, keyPoint);
        if ( keyPoint.size() <= 0 )
        {
            continue;
        }
        cv::Mat feature;
        bowDE->compute(img, keyPoint, feature);
        //float ret = svm.predict(feature);
        float ret = -1.f;

        if (CLASSIFIER::SVM == classifier){

            ret = svm.predict(feature);

        }else if (CLASSIFIER::BOOST == classifier){

            ret = boost.predict(feature);

        }else if (CLASSIFIER::KNN == classifier){

            ret = knn.find_nearest(feature,k);

        }else if (CLASSIFIER::ANN == classifier){

            cv::Mat outputs( 1, className.size(), CV_32FC1);
            ann.predict(feature,outputs);
            cv::Point maxLoc;
            cv::minMaxLoc(outputs, 0, 0, 0, &maxLoc);
            ret = maxLoc.x + 1;
            ret += 11000;
            std::cout << "max point ( " << maxLoc.x << " , " << maxLoc.y << " )" << std::endl;
        }

        std::cout << "pretict class :" << ret << std::endl;

        if (className.find(static_cast<int>(ret)) != className.end()){
            say.blockSpeak(className.find(static_cast<int>(ret))->second);
            std::cout << className.find(static_cast<int>(ret))->second << std::endl;
            emit sendStatues(QString::fromStdString(className.find(static_cast<int>(ret))->second));
        }else{
            say.blockSpeak("无法识别的类型");
        }
        stringstream ss;
        ss << static_cast<int>(ret) << "_" << ++count;
        cv::imwrite(logPath + ss.str() + "_predict.jpg",img);
        emit showStatues(cvMat2QImage(img));
    }

    return true;
}

void QRecogonition::trainAll()
{
    std::string featStr[] = {"SIFT","SURF","BRISK","ORB","FAST","STAR","GFTT","HARRIS","Dense","SimpleBlob"};
    std::string descStr[] = {"SIFT","SURF","BRIEF","BRISK","ORB","FREAK","OpponentSIFT","OpponentSURF"};

    cv::FileStorage timeFile("config/time.xml", cv::FileStorage::WRITE);

    for (auto& feat : featStr){
        for (auto& desc : descStr){

            // Clock for measuring the elapsed time
            std::chrono::time_point<std::chrono::system_clock> detectStart, detectEnd;
            detectStart = std::chrono::system_clock::now();

            extractor = cv::DescriptorExtractor::create(desc);
            detector = cv::FeatureDetector::create(feat);

            cv::TermCriteria tc(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.001);

            bTrain = new cv::BOWKMeansTrainer(dictSize, tc, 1, cv::KMEANS_PP_CENTERS);
            bowDE = new cv::BOWImgDescriptorExtractor(extractor, matcher);

            std::vector<std::string> imgList;
            tools::listFileByDir(trainPath,"jpg",imgList);

            std::cout << "imgList size = " << imgList.size() << std::endl;

            for (auto& file : imgList){
                cv::Mat img = cv::imread(file,1);
                std::vector<cv::KeyPoint> keyPoint;

                detector->detect(img, keyPoint);
                //cv::KeyPointsFilter::retainBest(keyPoint, 1500);

                std::cout << "Image keyPoint size = " << keyPoint.size() << std::endl;
                if ( keyPoint.empty() ){
                    continue;
                }
                cv::Mat feature;

                try{
                    extractor->compute(img, keyPoint, feature);
                }catch(cv::Exception ex){
                    std::cout << ex.msg << ex.code << ex.line << std::endl;
                    continue;
                }

                if ( feature.empty()){
                    continue;
                }

                if (feature.type() != CV_32F)
                {
                    feature.convertTo(feature, CV_32F);
                }

                bTrain->add(feature);
            }

            if (bTrain->descripotorsCount() < 2){
                continue;
            }

            dictionary = bTrain->cluster();
            std::string dictName = "config/dictionary" + feat + desc + ".xml";
            cv::FileStorage fs(dictName,cv:: FileStorage::WRITE);
            fs << "dictionary" << dictionary;
            fs.release();

            bowDE->setVocabulary(dictionary);

            //train
            cv::Mat labels(0, 1, CV_32FC1);
            cv::Mat trainData(0, dictSize, CV_32FC1);
            std::set<std::string> classNu;


            for (auto& file : imgList){
                cv::Mat img = cv::imread(file,1);
                std::vector<cv::KeyPoint> keyPoint;
                detector->detect(img, keyPoint);
                if ( keyPoint.empty() ){
                    continue;
                }
                cv::Mat feature;

                try{
                    bowDE->compute(img, keyPoint, feature);
                }catch(cv::Exception ex){
                    std::cout << ex.msg << ex.code << ex.line << std::endl;
                    continue;
                }

                trainData.push_back(feature);
                labels.push_back((static_cast<float>(std::atoi(tools::fileNamePart(file).substr(0,5).c_str()))));
                classNu.insert(tools::fileNamePart(file).substr(0,5));
            }

            detectEnd = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = detectEnd-detectStart;
            std::time_t end_time = std::chrono::system_clock::to_time_t(detectEnd);

            std::cout << "Finished computation at " << std::ctime(&end_time) << "Elapsed time: " << elapsed_seconds.count()*1000 << " ms\n";

            timeFile << "detector" << feat + desc;
            timeFile << "DictTimeCost";
            timeFile << "{" << "computation" << std::ctime(&end_time);
            timeFile << "Elapsed" << elapsed_seconds.count()*1000 << "}";

            for (int i = 1; i < 5; ++i){
                bool flag = false;

                // Clock for measuring the elapsed time
                std::chrono::time_point<std::chrono::system_clock> trainStart, trainEnd;
                trainStart = std::chrono::system_clock::now();


                if (CLASSIFIER::SVM == i){

                    std::cout << "Train svm : " << i << std::endl;

                    svmParams.kernel_type = CvSVM::RBF;
                    svmParams.svm_type = CvSVM::C_SVC;
                    svmParams.gamma = 0.50625000000000000000009;
                    svmParams.C = 312.500000000000000;
                    svmParams.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 0.000001);

                    CvSVM trainSvm;

                    if (trainSvm.train(trainData, labels, cv::Mat(), cv::Mat(), svmParams)){
                        std::string svmName = "config/" + feat + desc + "svm.xml";
                        trainSvm.save(svmName.c_str());
                        flag = true;
                    }
                }else if (CLASSIFIER::BOOST == i){

                    std::cout << "Train boost : " << i << std::endl;

                    gbParams.loss_function_type = CvGBTrees::DEVIANCE_LOSS;
                    gbParams.weak_count = 100;
                    gbParams.shrinkage = 0.1f;
                    gbParams.subsample_portion = 1.0f;
                    gbParams.max_depth = 2;
                    gbParams.use_surrogates = false;

                    cv::Mat var_types( 1, trainData.cols + 1, CV_8UC1, cv::Scalar(CV_VAR_ORDERED) );
                    var_types.at<uchar>( trainData.cols ) = CV_VAR_CATEGORICAL;

                    CvGBTrees trainBoost;

                    if (trainBoost.train(trainData,CV_ROW_SAMPLE,labels, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), gbParams)){
                        std::string boostName = "config/" + feat + desc + "boost.xml";
                        trainBoost.save(boostName.c_str());
                        flag = true;
                    }
                }else if (CLASSIFIER::KNN == i){
                    continue;
                    std::cout << "inital knn params" << std::endl;
                    k = 35;
                    CvKNearest trainKnn;
                    if (trainKnn.train(trainData,labels,cv::Mat(),false,k)){
                        std::cout << "train knn" << std::endl;
                        //std::string knnName = "config/" + feat + desc + "knn.xml";
                        //trainKnn.save(knnName.c_str());
                        flag = true;
                    }
                }else if (CLASSIFIER::ANN == i){
                    std::cout << "Train ann : " << i << std::endl;
                    std::cout << "inital ann params" << std::endl;
                    cv::Mat trainClass(labels.rows, classNu.size(), CV_32FC1);

                    for (auto i = 0; i < trainClass.rows; ++i){
                        for (auto j = 0; j < trainClass.cols; ++j){
                            if (labels.at<float>(i,0) == (j + 1)){
                                trainClass.at<float>(i,j) = 1;
                            }else{
                                trainClass.at<float>(i,j) = 0;
                            }
                        }
                    }

                    std::cout << "total classes : " << classNu.size() << std::endl;

                    cv::Mat layerSize(1, 3, CV_32SC1);
                    layerSize.at<int>(0) = trainData.cols;
                    layerSize.at<int>(1) = 35;
                    layerSize.at<int>(2) = classNu.size();

                    CvTermCriteria criteria;
                    criteria.max_iter = 10000;
                    criteria.epsilon  = 0.000001;
                    criteria.type     = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;

                    annParams.train_method = CvANN_MLP_TrainParams::BACKPROP;
                    annParams.bp_dw_scale = 0.1;
                    annParams.bp_moment_scale = 0.1;
                    annParams.term_crit = criteria;

                    std::cout << "create and train ann" << std::endl;

                    CvANN_MLP trainAnn;
                    trainAnn.create(layerSize, CvANN_MLP::SIGMOID_SYM);

                    if (trainAnn.train(trainData,trainClass,cv::Mat(), cv::Mat(), annParams)){
                        std::string annName = "config/" + feat + desc + "ann.xml";
                        trainAnn.save(annName.c_str());
                        flag = true;
                    }
                }

                trainEnd = std::chrono::system_clock::now();
                std::chrono::duration<double> train_elapsed_seconds = trainEnd-trainStart;
                std::time_t train_end_time = std::chrono::system_clock::to_time_t(trainEnd);

                //std::cout << "Finished computation at " << std::ctime(&train_end_time) << "Elapsed time: " << train_elapsed_seconds.count()*1000 << " ms\n";
                timeFile << "classifer" << i;
                timeFile << "TrainTimeCost";
                timeFile << "{" << "computation" << std::ctime(&train_end_time);
                timeFile << "Elapsed" << train_elapsed_seconds.count()*1000 << "}";
            }

            bTrain.release();
            bowDE.release();
            detector.release();
            extractor.release();
        }
    }

    timeFile.release();
}

void QRecogonition::run()
{
    if (initDictionary()){
        emit sendStatues(QStringLiteral("InitDictionary success ....!"));
        predict();
    }else {
        emit sendStatues(QStringLiteral("InitDictionary fail !"));
    }

    //trainAll();
}
