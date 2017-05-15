#include "recogonition.h"

// Eigen library
#include <Eigen/Core>

// our own code
#include "segmentation.h"
#include "colorConversion.h"
#include "imageProcessing.h"
#include "smartOptimisation.h"
#include "math_utils.h"

const string Recogonition::dictFile =  "config/dictionary.xml";
const string Recogonition::svmFile = "config/classes.xml";
const string Recogonition::classFile = "config/className.txt";

Recogonition::Recogonition()
{
    cv::initModule_nonfree();//if use SIFT or SURF
    //matcher = cv::DescriptorMatcher::create("FlannBased");
    matcher = cv::DescriptorMatcher::create("BruteForce");
    extractor = cv::DescriptorExtractor::create("SIFT");
    detector = cv::FeatureDetector::create("SIFT");
}

Recogonition::Recogonition(const string &trainPath) : trainPath(trainPath)
{
    cv::initModule_nonfree();//if use SIFT or SURF
    matcher = cv::DescriptorMatcher::create("FlannBased");
    extractor = cv::DescriptorExtractor::create("SIFT");
    detector = cv::FeatureDetector::create("SIFT");
}

void Recogonition::getFeature(const cv::Mat &roiImg, cv::Mat &features)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("SIFT");
    std::vector<cv::KeyPoint> featuresPoints;
    detector->detect(roiImg, featuresPoints);
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create( "SIFT" );
    descriptorExtractor->compute(roiImg,featuresPoints,features);
}

void Recogonition::featureCluster(const std::vector<cv::Mat> features, int centerNu, cv::Mat& centers)
{
    cv::Mat fMat,labels;
    for (auto& feature : features)
    {
        fMat.push_back(feature);
    }
    cv::kmeans(fMat,centerNu,labels,cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),3, cv::KMEANS_PP_CENTERS, centers);
}

void Recogonition::getRegion(const string &src, vector<cv::Mat>& regions){
    // Read the input image
    cv::Mat input_image = cv::imread(src);

    // Check that the image has been opened
    if (!input_image.data) {
        std::cout << "Error to read the image. Check ''cv::imread'' function of OpenCV" << std::endl;
        return;
    }
    // Check that the image read is a 3 channels image
    CV_Assert(input_image.channels() == 3);

    cv::Mat bImg;
    //cv::bilateralFilter ( input_image, bImg, 3, 3*2, 2/3 );
    cv::adaptiveBilateralFilter(input_image,bImg,cv::Size(3,3),0.5);
    input_image = bImg;
    imwrite("filterImg.jpg",bImg);

    /*
   * Conversion of the image in some specific color space
   */

    // Conversion of the rgb image in ihls color space
    cv::Mat ihls_image;
    colorconversion::convert_rgb_to_ihls(input_image, ihls_image);
    // Conversion from RGB to logarithmic chromatic red and blue
    std::vector< cv::Mat > log_image;
    colorconversion::rgb_to_log_rb(input_image, log_image);

    /*
   * Segmentation of the image using the previous transformation
   */

    // Segmentation of the IHLS and more precisely of the normalised hue channel
    // ONE PARAMETER TO CONSIDER - COLOR OF THE TRAFFIC SIGN TO DETECT - RED VS BLUE
    /*int nhs_mode = 0; // nhs_mode == 0 -> red segmentation / nhs_mode == 1 -> blue segmentation
    cv::Mat nhs_image_seg_red;
    segmentation::seg_norm_hue(ihls_image, nhs_image_seg_red, nhs_mode);
    //nhs_mode = 1; // nhs_mode == 0 -> red segmentation / nhs_mode == 1 -> blue segmentation
    //cv::Mat nhs_image_seg_blue;
    cv::Mat nhs_image_seg_blue = nhs_image_seg_red.clone();
    //segmentation::seg_norm_hue(ihls_image, nhs_image_seg_blue, nhs_mode);
    // Segmentation of the log chromatic image
    // TODO - DEFINE THE THRESHOLD FOR THE BLUE TRAFFIC SIGN. FOR NOW WE AVOID THE PROCESSING FOR BLUE SIGN AND LET ONLY THE OTHER METHOD TO TAKE CARE OF IT.
    cv::Mat log_image_seg;
    segmentation::seg_log_chromatic(log_image, log_image_seg);*/

    /*
   * Merging and filtering of the previous segmentation
   */

    // Merge the results of previous segmentation using an OR operator
    // Pre-allocation of an image by cloning a previous image
    /*cv::Mat merge_image_seg_with_red = nhs_image_seg_red.clone();
    cv::Mat merge_image_seg = nhs_image_seg_blue.clone();
    cv::bitwise_or(nhs_image_seg_red, log_image_seg, merge_image_seg_with_red);
    cv::bitwise_or(nhs_image_seg_blue, merge_image_seg_with_red, merge_image_seg);

    // Filter the image using median filtering and morpho math
    cv::Mat bin_image;
    imageprocessing::filter_image(merge_image_seg, bin_image);

    cv::imwrite("unf_seg.jpg",merge_image_seg);
    cv::imwrite("seg.jpg", bin_image);*/

    /*
   * Extract candidates (i.e., contours) and remove inconsistent candidates
   */

    /*std::vector< std::vector< cv::Point > > distorted_contours;
    imageprocessing::contours_extraction(bin_image, distorted_contours);*/

    cv::Mat log_image_seg;
    segmentation::seg_log_chromatic(log_image, log_image_seg);
    std::vector< std::vector< cv::Point > > distorted_contours;
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
             cv::imwrite("red_unf_seg.jpg",merge_image_seg);
             cv::imwrite("red_seg.jpg", bin_image);
             break;
         default:
             cv::imwrite("blue_unf_seg.jpg",merge_image_seg);
             cv::imwrite("blue_seg.jpg", bin_image);
             break;
         }

        std::vector< std::vector< cv::Point > > distortedContours;
         imageprocessing::contours_extraction(bin_image, distortedContours);

         for (auto& contour : distortedContours){
             distorted_contours.push_back(contour);
         }
    }

    /*
   * Correct the distortion for each contour
   */

    // Initialisation of the variables which will be returned after the distortion. These variables are linked with the transformation applied to correct the distortion
    std::vector< cv::Mat > rotation_matrix(distorted_contours.size());
    std::vector< cv::Mat > scaling_matrix(distorted_contours.size());
    std::vector< cv::Mat > translation_matrix(distorted_contours.size());
    for (unsigned int contour_idx = 0; contour_idx < distorted_contours.size(); contour_idx++) {
        rotation_matrix[contour_idx] = cv::Mat::eye(3, 3, CV_64F);
        scaling_matrix[contour_idx] = cv::Mat::eye(3, 3, CV_64F);
        translation_matrix[contour_idx] = cv::Mat::eye(3, 3, CV_64F);
    }

    // Correct the distortion
    std::vector< std::vector< cv::Point2f > > undistorted_contours;
    imageprocessing::correction_distortion(distorted_contours, undistorted_contours, translation_matrix, rotation_matrix, scaling_matrix);

    // Normalise the contours to be inside a unit circle
    std::vector<double> factor_vector(undistorted_contours.size());
    std::vector< std::vector< cv::Point2f > > normalised_contours;
    initoptimisation::normalise_all_contours(undistorted_contours, normalised_contours, factor_vector);

    std::vector< std::vector< cv::Point2f > > detected_signs_2f(normalised_contours.size());
    std::vector< std::vector< cv::Point > > detected_signs(normalised_contours.size());


    // For each contours
    for (unsigned int contour_idx = 0; contour_idx < normalised_contours.size(); contour_idx++) {


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
            cv::Point2f mass_center = initoptimisation::mass_center_discovery(input_image, translation_matrix[contour_idx], rotation_matrix[contour_idx], scaling_matrix[contour_idx], normalised_contours[contour_idx], factor_vector[contour_idx], sign_type);

            // Find the rotation offset
            double rot_offset = initoptimisation::rotation_offset(normalised_contours[contour_idx]);

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
            optimisation::gielis_optimisation(normalised_contours[contour_idx], contour_config, mean_err, std_err);

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
        initoptimisation::denormalise_contour(gielis_contour, denormalised_gielis_contour, factor_vector[contour_idx]);
        std::vector< cv::Point2f > distorted_gielis_contour;
        imageprocessing::inverse_transformation_contour(denormalised_gielis_contour, distorted_gielis_contour, translation_matrix[contour_idx], rotation_matrix[contour_idx], scaling_matrix[contour_idx]);

        // Transform to cv::Point to show the results
        std::vector< cv::Point > distorted_gielis_contour_int(distorted_gielis_contour.size());
        for (unsigned int i = 0; i < distorted_gielis_contour.size(); i++) {
            distorted_gielis_contour_int[i].x = (int) std::round(distorted_gielis_contour[i].x);
            distorted_gielis_contour_int[i].y = (int) std::round(distorted_gielis_contour[i].y);
        }

        detected_signs_2f[contour_idx] = distorted_gielis_contour;
        detected_signs[contour_idx] = distorted_gielis_contour_int;

    }


    cv::Mat output_image = input_image.clone();
    cv::Scalar color(0,255,0);
    cv::drawContours(output_image, detected_signs, -1, color, 2, 8);
    imwrite("result.jpg",output_image);

    for (auto& contour : detected_signs){
        cv::Rect rect = cv::boundingRect(cv::Mat(contour));
        cv::Mat reg = input_image(rect);
        regions.push_back(reg);
        cv::Scalar color(0,255,0);
        cv::rectangle(output_image,rect,color);
    }

    imwrite("result_rect.jpg",output_image);
}

void Recogonition::getClassName(){
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
    }
}

void Recogonition::initDictionary(){

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
            cv::Mat img = cv::imread(file,0);
            if (img.data && img.channels() == 1){
                std::cout << "Read img success" << std::endl;
            }
            std::vector<cv::KeyPoint> keyPoint;
            detector->detect(img, keyPoint);
            std::cout << "Image keyPoint size = " << keyPoint.size() << std::endl;
            if ( keyPoint.size() <= 0 ){
                continue;
            }
            cv::Mat feature;
            extractor->compute(img, keyPoint, feature);
            bTrain->add(feature);
        }

        dictionary = bTrain->cluster();
        cv::FileStorage fs(dictFile,cv:: FileStorage::WRITE);
        fs << "dictionary" << dictionary;
        fs.release();
    }

    bowDE->setVocabulary(dictionary);
}

void Recogonition::bowTrain()
{    
    cv::Mat labels(0, 1, CV_32FC1);
    cv::Mat trainData(0, dictSize, CV_32FC1);

    std::vector<std::string> imgList;
    tools::listFileByDir(trainPath,"jpg",imgList);

    for (auto& file : imgList){
        cv::Mat img = cv::imread(file,0);
        std::vector<cv::KeyPoint> keyPoint;
        detector->detect(img, keyPoint);
        if ( keyPoint.size() <= 0 ){
            continue;
        }
        cv::Mat feature;
        bowDE->compute(img, keyPoint, feature);
        trainData.push_back(feature);
        labels.push_back((static_cast<float>(std::atoi(tools::fileNamePart(file).substr(0,5).c_str()))));
    }

    params.kernel_type = CvSVM::RBF;
    params.svm_type = CvSVM::C_SVC;
    params.gamma = 0.50625000000000000000009;
    params.C = 312.500000000000000;
    params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);

    bool res = svm.train(trainData, labels, cv::Mat(), cv::Mat(), params);
    svm.save(svmFile.c_str());
}

void Recogonition::predict(const string &src)
{
    cv::FileStorage sFs(svmFile, cv::FileStorage::READ);
    if (sFs.isOpened()){
        sFs.release();
        svm.load(svmFile.c_str());
    }else{
        bowTrain();
    }
    getClassName();
    std::vector<cv::Mat> roiImg;
    getRegion(src, roiImg);

    ekho::Ekho say("Mandarin");
    int count = 0;
    for (auto& img : roiImg)
    {
        cv::Mat gImg;
        cv::cvtColor(img, gImg, CV_BGR2BGRA);
        std::vector<cv::KeyPoint> keyPoint;
        detector->detect(img, keyPoint);
        if ( keyPoint.size() <= 0 ){
            continue;
        }
        cv::Mat feature;
        bowDE->compute(img, keyPoint, feature);
        float ret = svm.predict(feature);

        if (className.find(static_cast<int>(ret)) != className.end()){
            say.blockSpeak(className.find(static_cast<int>(ret))->second);
            cout << className.find(static_cast<int>(ret))->second << endl;
        }else{
            say.blockSpeak("无法识别的类型");
        }
        stringstream ss;
        ss << static_cast<int>(ret) << "_" << ++count;
        cv::imwrite(ss.str() + "_predict.jpg",img);
        cv::imshow(ss.str(),img);
        cv::waitKey(0);
    }
}
