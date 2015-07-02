#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/normal_prior.h>
#include <glog/logging.h>

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
    : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T* const camera,
    const T* const point,
    T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];
    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2  * (l1 + l2  * r2);
    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
    const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
      new SnavelyReprojectionError(observed_x, observed_y)));
  }
  double observed_x;
  double observed_y;
};

void main() {

  std::vector< std::vector<cv::Point2d> > cams;
  std::vector< std::vector<cv::Point2d> > cams_pruned;
  std::vector< cv::Mat> t;// (1000, 1000, CV_8UC1);
 
  for (int i = 0; i < 2; i++) {
    std::string path;
    std::stringstream stm(path);
    stm << "x:\\Calibration\\";

    stm << "points.0" << i << ".txt";

    std::ifstream fs;
    int count;

    fs.open(stm.str());

    fs >> count;

    std::cout << count << std::endl;

    cams.push_back( std::vector<cv::Point2d>() );
    cams.back().resize(count * 2);
    t.push_back(cv::Mat(1000, 1000, CV_8UC1));

    std::vector<cv::Point2d>::iterator bg = cams.back().begin();

    if (fs.is_open())
    {
      while (!fs.eof()) {
        std::string pp;
        double x, y;
        fs >> x;

        //std::cout << pp << " ";

        fs >> y;
        //std::cout << y << std::endl;

        (*bg).x = x;
        (*bg).y = y;

        std::advance(bg, 1);
      }
   
      fs.close();
    }

    for (int i = 0; i < cams.back().size(); i++) {
	    cv::circle(t.back(), cams.back()[i], 5, cv::Scalar(255));
    }

    stm << ".png";
    cv::imwrite(stm.str(), t.back());
    //cv::imshow(stm.str(), t.back());
    //cv::waitKey(0);
  }

  cams_pruned.resize(cams.size());

  for (int i = 0; i < cams.back().size(); i++) {
    if (cams[0][i].x != -1 && cams[1][i].x != -1) {
      cams_pruned[0].push_back(cams[0][i]);
      cams_pruned[1].push_back(cams[1][i]);
    }
  }

  std::cout << cams_pruned[0].size();


  //const double* observations = bal_problem.observations();
  double* params = new double[6];
  ceres::Problem problem;
  for (int i = 0; i < cams_pruned[0].size(); ++i) {
 
    ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(cams_pruned[1][i].x, cams_pruned[1][i].y);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */, params);
  }
 
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";



  //for (int i = 0; i < cam00.size(); i++) {
	 // cv::circle(t, cam00[i], 5, cv::Scalar(255));
  //}
  //for (int i = 0; i < cam00.size(); i += 2) {
	 // if (cam00[i].x >= 0 && cam00[i + 1].x >= 0)
	 // cv::line(t, cam00[i], cam00[i+1], cv::Scalar(255));
  //}
  system("PAUSE");
}
