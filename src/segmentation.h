/*
* Copyright (c) 2015 
* Guillaume Lemaitre (g.lemaitre58@gmail.com)
* Yohan Fougerolle (Yohan.Fougerolle@u-bourgogne.fr)
*
* This program is free software; you can redistribute it and/or modify it
* under the terms of the GNU General Public License as published by the Free
* Software Foundation; either version 2 of the License, or (at your option)
* any later version.
*
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
* more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc., 51
* Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

// stl library
#include <vector>

// OpenCV library
#include <opencv2/opencv.hpp>

/* Definition for log segmentation */
// To segment red traffic signs
#define MINLOGRG 0.5
#define MAXLOGRG 2.1
// To segment blue traffic signs
#define MINLOGBG -0.9
#define MAXLOGBG 0.8

/* Definition for ihls segmentation */
// To segment red traffic signs
#define R_HUE_MAX 15 // R_HUE_MAX 11
#define R_HUE_MIN 240
#define R_SAT_MIN 25 // R_SAT_MIN 30
#define R_CONDITION (h < hue_max || h > hue_min) && s > sat_min
// To segment blue traffic signs
#define B_HUE_MAX 163
#define B_HUE_MIN 134
#define B_SAT_MIN 39 // B_SAT_MIN 20
#define B_CONDITION (h < hue_max && h > hue_min) && s > sat_min

namespace segmentation {

  // Segmentation of logarithmic chromatic images
  void seg_log_chromatic(const std::vector< cv::Mat >& log_image, cv::Mat& log_image_seg);

  // Segmentation of normalised hue
  void seg_norm_hue(const cv::Mat& ihls_image, cv::Mat& nhs_image, const int& colour = 0, int hue_max = R_HUE_MAX, int hue_min = R_HUE_MIN, int sat_min = R_SAT_MIN);

}

#endif // SEGMENTATION_H
