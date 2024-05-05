#include "executor.h"

namespace Inference
{

    std::pair<cv::Mat, cv::Mat> executor::rectify(const cv::Mat &_left, const cv::Mat &_right)
    {
        cv::Mat left = _left.clone();
        cv::Mat right = _right.clone();
        // 去畸变
        cv::remap(left, left, maplx, maply, cv::INTER_CUBIC);
        cv::remap(right, right, maplx, maply, cv::INTER_CUBIC);
        return std::pair<cv::Mat, cv::Mat>(left, right);
    }
    executor::executor()
    {
        param.disp_min = 1;                 // 最小视差
        param.disp_max = 192;                // 最大视差
        param.support_threshold = 0.85;     // 比率测试：最低match VS 次低match
        param.support_texture = 10;         // 支持点的最小纹理
        param.candidate_stepsize = 5;       // 用于支持点的sobel特征匹配的邻域半径
        param.incon_window_size = 5;        // 不连续性窗口的尺寸
        param.incon_threshold = 5;          // 不连续性窗口内的视差范围阈值
        param.incon_min_support = 5;        // 不连续性窗口内的最低支持点数量
        param.add_corners = true;           // 是否添加角点
        param.grid_size = 20;               // 网格尺寸
        param.beta = 0.02;                  // 图像相似性度量的参数
        param.gamma = 3;                    // 先验概率常数
        param.sigma = 1;                    // 先验概率的标准差
        param.sradius = 3;                  // 标准差半径
        param.match_texture = 1;            // 最低纹理
        param.lr_threshold = 1;             // 左右一致性检验阈值
        param.speckle_sim_threshold = 1;    // 连通域判断阈值
        param.speckle_size = 200;           // 连通域噪声尺寸判断阈值
        param.ipol_gap_width = 3;           // 空洞宽
        param.filter_median = false;        // 是否中值滤波
        param.filter_adaptive_mean = true;  // 是否自适应中值滤波
        param.postprocess_only_left = true; // 是否只对左视差图后处理，设置为True可以节省时间
        param.subsampling = false;          // 每个两个像素进行视差计算，设置为True可以节省时间，但是传入的D1和D2的分辨率必须为(w/2) x (h/2)
        model = std::make_unique<Elas>(param);
    }
    executor::executor(const std::string &param_file) : param(Elas::setting::ROBOTICS)
    {
        param.reset();
        param.disp_min = 0;
        param.disp_max = 192;
        // param.support_threshold = 0.7;     // 比率测试：最低match VS 次低match
        // param.support_texture = 5;         // 支持点的最小纹理
        // param.candidate_stepsize = 5;       // 用于支持点的sobel特征匹配的邻域半径
        // param.incon_window_size = 5;        // 不连续性窗口的尺寸
        // param.incon_threshold = 5;          // 不连续性窗口内的视差范围阈值
        // param.incon_min_support = 5;        // 不连续性窗口内的最低支持点数量
        // param.add_corners = true;           // 是否添加角点
        // param.grid_size = 20;               // 网格尺寸
        // param.beta = 0.02;                  // 图像相似性度量的参数
        // param.gamma = 3;                    // 先验概率常数
        // param.sigma = 1;                    // 先验概率的标准差
        // param.sradius = 5;                  // 标准差半径
        // param.match_texture = 1;            // 最低纹理
        // param.lr_threshold = 1;             // 左右一致性检验阈值
        // param.speckle_sim_threshold = 1;    // 连通域判断阈值
        // param.speckle_size = 200;           // 连通域噪声尺寸判断阈值
        // param.ipol_gap_width = 3;           // 空洞宽
        // param.filter_median = false;        // 是否中值滤波
        // param.filter_adaptive_mean = true;  // 是否自适应中值滤波
        // param.postprocess_only_left = true; // 是否只对左视差图后处理，设置为True可以节省时间
        // param.subsampling = false;          // 每个两个像素进行视差计算，设置为True可以节省时间，但是传入的D1和D2的分辨率必须为(w/2) x (h/2)

        model = std::make_unique<Elas>(param);
        cv::FileStorage file(param_file, cv::FileStorage::READ);
        cv::Mat K_l, D_l, K_r, D_r;
        cv::Mat P_l, R_l, P_r, R_r;
        if (!file.isOpened())
        {
            std::cout << "the parameters file does not open !" << std::endl;
            CV_Assert(file.isOpened());
        }

        // 参数导入
        file["K_l"] >> K_l;
        file["D_l"] >> D_l;
        cv::Size image_size = cv::Size(file["width"], file["height"]);
        file["K_r"] >> K_r;
        file["D_r"] >> D_r;
        file["R_l"] >> R_l;
        file["R_r"] >> R_r;
        file["P_l"] >> P_l;
        file["P_r"] >> P_r;
        file["Q"] >> Q;
        if (file["Camera_SensorType"].string() == "Fisheye")
        {
            // fisheye remap prepare
            cv::fisheye::initUndistortRectifyMap(K_l, D_l, R_l, P_l, image_size, CV_32FC1, maplx, maply);
            cv::fisheye::initUndistortRectifyMap(K_r, D_r, R_r, P_r, image_size, CV_32FC1, maprx, mapry);
        }
        else if (file["Camera_SensorType"].string() == "Pinhole")
        {
            // pinhole remap prepare
            cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l, image_size, CV_32FC1, maplx, maply);
            cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r, image_size, CV_32FC1, maprx, mapry);
        }
    }

    cv::Mat executor::infer_rectified(const cv::Mat &_left, const cv::Mat &_right)
    {
        cv::Mat left = _left.clone();
        cv::Mat right = _right.clone();
        cv::imshow("left image", left);
        cv::imshow("right image", right);
        // cv::waitKey(0);
        // cv::destroyAllWindows();
        // 参数调整
        int height = left.rows;
        int width = left.cols;
        int dim[3] = {width, height, width};
        if (left.channels() == 3)
            cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
        if (right.channels() == 3)
            cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);
        cv::Mat left_disp = cv::Mat::zeros(left.size(), CV_32FC1);
        cv::Mat right_disp = cv::Mat::zeros(right.size(), CV_32FC1);
        // 计算
        model->process(left.data, right.data, left_disp.ptr<float>(0), right_disp.ptr<float>(0), dim);

        return left_disp;
    }
    cv::Mat executor::infer(const cv::Mat &_left, const cv::Mat &_right)
    {
        cv::Mat left = _left.clone();
        cv::Mat right = _right.clone();
        // 去畸变
        cv::remap(left, left, maplx, maply, cv::INTER_CUBIC);
        cv::remap(right, right, maplx, maply, cv::INTER_CUBIC);
        // cv::imshow("recitfy_left", left);
        // cv::imshow("recitfy_right", right);
        // cv::waitKey(0);
        // cv::destroyAllWindows();
        // 参数调整
        int height = left.rows;
        int width = left.cols;
        int dim[3] = {width, height, width};
        if (left.channels() == 3)
            cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
        if (right.channels() == 3)
            cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);
        cv::Mat left_disp = cv::Mat::zeros(left.size(), CV_32FC1);
        cv::Mat right_disp = cv::Mat::zeros(right.size(), CV_32FC1);
        // 计算
        model->process(left.data, right.data, left_disp.ptr<float>(0), right_disp.ptr<float>(0), dim);

        return left_disp;
    }

}