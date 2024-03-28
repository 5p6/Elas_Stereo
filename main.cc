#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "./utility/executor.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "./utility/utility.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr GetPointCloud(const cv::Mat &rgb, const cv::Mat &Points_image)
{
    // point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->points.resize(rgb.rows * rgb.cols);

    //
    Points_image.convertTo(Points_image, CV_32F);
    for (int i = 0; i < rgb.rows; i++)
    {
        for (int j = 0; j < rgb.cols; j++)
        {
            pcl::PointXYZRGB p;
            p.x = Points_image.ptr<float>(i)[3 * j];
            p.y = Points_image.ptr<float>(i)[3 * j + 1];
            p.z = Points_image.ptr<float>(i)[3 * j + 2];

            double r = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (r > 10 || r < 1)
                continue;

            // color
            if (rgb.channels() == 3)
            {
                p.b = rgb.ptr<uchar>(i)[3 * j];
                p.g = rgb.ptr<uchar>(i)[3 * j + 1];
                p.r = rgb.ptr<uchar>(i)[3 * j + 2];
            }
            else
            {
                p.b = 0;
                p.g = 0;
                p.r = 0;
            }
            cloud->points.emplace_back(p);
        }
    }
    return cloud;
}

void exec_img(const std::string &left_path, const std::string &right_path, const std::string &param_file)
{
    pcl::visualization::PCLVisualizer viewer("point cloud");
    cv::Mat left = cv::imread(left_path);
    cv::Mat right = cv::imread(right_path);
    Inference::executor e(param_file);

    // cv::Mat pred = e.infer(left, right);
    // Initialize SGBM parameters
    cv::Mat pred;
    // -------------------
    int window_size = 3;
    int min_disp = 0;
    int num_disp = ((left.cols / 8) + 15) - 16;
    int block_size = 3;
    int P1 = 8 * 3 * window_size * window_size;
    int P2 = 32 * 3 * window_size * window_size;

    // Create SGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        min_disp, num_disp, block_size, P1, P2, 1, 63, 10, 100, 100, cv::StereoSGBM::MODE_SGBM_3WAY);

    stereo->compute(left, right, pred);
    // ---------
    cv::Mat disp_vis;
    pred.convertTo(disp_vis, CV_8UC1);
    cv::Mat point_image;
    // 3d image
    cv::reprojectImageTo3D(pred, point_image, e.Q);

    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_INFERNO);
    cv::namedWindow("disp", cv::WINDOW_NORMAL);
    cv::imshow("disp", disp_vis);

    auto ps = GetPointCloud(left, point_image);
    viewer.addPointCloud(ps);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void exec_rectified(const std::string &left_path, const std::string &right_path)
{
    cv::Mat left = cv::imread(left_path);
    cv::Mat right = cv::imread(right_path);
    Inference::executor e;
    cv::Mat pred;
    // cv::Mat pred = e.infer_rectified(left, right);
    // -------------------
    int block_size = 8;
    int P1 = 8 * 3 * block_size * block_size;
    int P2 = 32 * 3 * block_size * block_size;

    // Create SGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0, 64, block_size, P1, P2, -1, 1, 10, 100, 100, cv::StereoSGBM::MODE_SGBM);
    
    stereo->compute(left,right,pred);
    cv::Mat disp_vis;
    pred.convertTo(disp_vis, CV_8UC1);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_INFERNO);
    cv::imshow("left",left);
    cv::namedWindow("disp", cv::WINDOW_NORMAL);
    cv::imshow("disp", disp_vis);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void exec_video(int id, int height, int width, const std::string &param_file)
{
    std::unique_ptr<pcl::visualization::CloudViewer> viewer(new pcl::visualization::CloudViewer("cloud"));
    cv::VideoCapture cap(id);
    cap.set(4, height);
    cap.set(3, width);

    cv::Mat frame;
    Inference::executor e(param_file);
    // -------------------
    int block_size = 13;
    int P1 = 8 * 3 * block_size * block_size;
    int P2 = 32 * 3 * block_size * block_size;

    // Create SGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0, 128, block_size, P1, P2, 5, 1, 10, 100, 10, cv::StereoSGBM::MODE_SGBM_3WAY);
    
    while (cap.isOpened())
    {
        cap >> frame;
        if (!frame.empty())
        {
            std::cout << "this frame is empty" << std::endl;
        }
        cv::Mat left = frame.colRange(cv::Range(0, static_cast<int>(width / 2)));
        cv::Mat right = frame.colRange(cv::Range(static_cast<int>(width / 2) + 1, width));

        cv::namedWindow("left", cv::WINDOW_NORMAL);
        cv::imshow("left", left);
        // get disp
        cv::Mat pred = e.infer(left, right);
        pred.convertTo(pred, CV_8UC1);
        // visualize the disp float
        cv::Mat disp_vis;
        cv::applyColorMap(pred, disp_vis, cv::COLORMAP_INFERNO);
        cv::namedWindow("disp", cv::WINDOW_NORMAL);
        cv::imshow("disp", disp_vis);

        cv::Mat points_image;
        // 3d
        cv::reprojectImageTo3D(pred, points_image, e.Q);
        //
        auto ps = GetPointCloud(left, points_image);
        // show
        viewer->showCloud(ps);
        if (cv::waitKey(1) == 27)
            break;
    }

    cv::destroyAllWindows();
}

void exec_sequence(const std::string &left_path, const std::string &right_path, const std::string &param_path)
{
    Inference::executor e(param_path);
    // -------------------
    int block_size = 8;
    int P1 = 8 * 3 * block_size * block_size;
    int P2 = 32 * 3 * block_size * block_size;

    // Create SGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0, 128, block_size, P1, P2, -1, 1, 10, 100, 100, cv::StereoSGBM::MODE_HH4);

    std::vector<std::string> left_paths = utility::getFilesFromPathWithoutSort(left_path);
    std::vector<std::string> right_paths = utility::getFilesFromPathWithoutSort(right_path);
    double min = 0;
    double max = 0;
    for (int i = 0; i < left_paths.size(); i++)
    {
        // get the disp image
        cv::Mat left = cv::imread(left_paths[i]);
        cv::Mat right = cv::imread(right_paths[i]);

        // auto [rectify_left,rectify_right] = e.rectify(left,right);
        cv::Mat disp = e.infer(left, right);
        // cv::Mat disp;
        // stereo->compute(rectify_left,rectify_right,disp);

        // convert
        disp.convertTo(disp, CV_32FC1);
        // min and max value location
        cv::minMaxLoc(disp, &min, &max);
        // std::cout<<"min : "<<min<<" max : "<<max<<std::endl;
        // normalization
        cv::Mat disp_norm = (disp - min) / (max - min) * 255;
        disp_norm.convertTo(disp_norm, CV_8U);
        cv::Mat disp_vis;
        cv::applyColorMap(disp_norm, disp_vis, cv::COLORMAP_INFERNO);
        // cv::namedWindow(std::string("disp") + std::to_string(i), cv::WINDOW_NORMAL);
        // cv::imshow("left", rectify_left);
        // cv::imshow("right", rectify_right);
        cv::imshow("disp", disp_vis);
        if(cv::waitKey(5)==27) break;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "the usage of this code : \n"
                  << "[Exec] [Img] [Left_Img] [Right_Img] [Param_file] \n"
                  << "[Exec] [Video] [Id] [Height] [Width] [Param_file] \n"
                  << "[Exec] [Rectified] [Left_Img] [Right_Img] \n"
                  << "[Exec] [Sequence] [Left_Sequence_Path] [Right_Sequence_Path] [Param_file]"
                  << std::endl;
        return 0;
    }
    if (std::string(argv[1]) == "Img")
    {
        exec_img(argv[2], argv[3], argv[4]);
    }
    else if (std::string(argv[1]) == "Video")
    {
        exec_video(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), argv[5]);
    }
    else if (std::string(argv[1]) == "Rectified")
    {
        exec_rectified(argv[2], argv[3]);
    }
    else if (std::string(argv[1]) == "Sequence")
    {
        exec_sequence(argv[2], argv[3], argv[4]);
    }
    return 1;
}