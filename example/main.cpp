// 消费者示例：用 food_volume_measure 的公开 API 跑一次 IM 体积测量。
//
// 用法:
//   example <baseline.pcd> <food.pcd> [config.json] [--dump-middle[=DIR]]
//
//   baseline.pcd         空炉基线点云
//   food.pcd             待测食材点云
//   config.json          可选，MeasurementConfig 的 JSON；省略则用内置默认值
//   --dump-middle[=DIR]  可选，把各阶段中间点云写到 DIR（默认 middle_data/cpp）

#include <iomanip>
#include <iostream>
#include <string>

#include "volume_log.hpp"
#include "volume_measurement.hpp"



namespace {

constexpr const char* kDefaultMiddleDir = "middle_data/cpp";

void print_usage(const char* exe) {
    std::cout << "用法: " << exe << " <baseline.pcd> <food.pcd> [config.json] [--dump-middle[=DIR]]\n\n"
              << "  baseline.pcd        空炉基线点云\n"
              << "  food.pcd            待测食材点云\n"
              << "  config.json         可选，MeasurementConfig JSON\n"
              << "  --dump-middle[=DIR] 可选，写出各阶段中间点云（默认 " << kDefaultMiddleDir << "）\n";
}

// 打印一次测量的完整诊断信息，含逐连通块明细。
void print_estimate(const vm::VolumeEstimate& est) {
    std::cout << "\n=== 测量结果 ===\n";
    std::cout << "状态                : " << vm::status_to_string(est.status) << "\n";
    if (est.status != vm::MeasurementStatus::kSuccess) {
        std::cout << "失败原因            : " << est.message << "\n";
        return;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "体积                : " << est.volume_cm3 << " cm^3\n";
    std::cout << "  实测部分          : " << est.raw_volume_cm3 << " cm^3\n";
    std::cout << "  补洞部分          : " << est.interpolated_volume_cm3 << " cm^3\n";
    std::cout << std::defaultfloat;

    std::cout << "\n--- 点云 ---\n";
    std::cout << "输入点数            : " << est.input_points << "\n";
    std::cout << "降采样点数          : " << est.downsampled_points << "\n";
    std::cout << "baseline 帧数       : " << est.baseline_frames << "\n";
    std::cout << "baseline 网格数     : " << est.baseline_cell_count << "\n";

    std::cout << "\n--- 连通块 ---\n";
    std::cout << "DBSCAN 簇数         : " << est.cluster_count << "\n";
    std::cout << "选中块数            : " << est.component_count << "\n";
    std::cout << "选中块标签          : [";
    for (std::size_t i = 0; i < est.selected_cluster_labels.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << est.selected_cluster_labels[i];
    }
    std::cout << "]\n";
    std::cout << "选中块点数          : " << est.selected_cluster_points << "\n";

    for (std::size_t i = 0; i < est.component_estimates.size(); ++i) {
        const vm::ComponentVolumeEstimate& c = est.component_estimates[i];
        const int label = (i < est.selected_cluster_labels.size()) ? est.selected_cluster_labels[i] : -1;
        std::cout << "  块[" << label << "] 体积 " << std::fixed << std::setprecision(3) << c.volume_cm3 << " cm^3"
                  << " | 实测格 " << c.measured_cells << " 补洞格 " << c.interpolated_cells << " | 最大高度 "
                  << std::setprecision(4) << c.max_height_m << " m\n";
        std::cout << std::defaultfloat;
    }

    std::cout << "\n--- 栅格 / 几何 ---\n";
    std::cout << "顶表面格数          : " << est.top_surface_points << "\n";
    std::cout << "实测格 / 补洞格     : " << est.measured_cells << " / " << est.interpolated_cells << "\n";
    std::cout << "占用格 / 外接框格   : " << est.occupied_cells << " / " << est.bbox_cell_count << "\n";
    std::cout << "未匹配 baseline 格  : " << est.missing_baseline_cells << "\n";
    std::cout << "未补洞格数          : " << est.unfilled_hole_cells << "\n";
    std::cout << std::setprecision(6);
    std::cout << "足迹面积            : " << est.footprint_area_m2 << " m^2\n";
    std::cout << "平均 / 最大高度     : " << est.mean_height_m << " / " << est.max_height_m << " m\n";
    std::cout << std::setprecision(4);
    std::cout << "覆盖率              : " << est.coverage_ratio << "\n";
    std::cout << "AABB / OBB / 凸包   : " << est.aabb_volume_m3 << " / " << est.obb_volume_m3 << " / "
              << est.convex_hull_volume_m3 << " m^3\n";
    std::cout << std::defaultfloat;
}

} // namespace



int main(int argc, char* argv[]) {
    std::string baseline_path;
    std::string food_path;
    std::string config_path;
    std::string middle_dir;
    bool dump_middle = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg.rfind("--dump-middle", 0) == 0) {
            dump_middle = true;
            const std::size_t eq = arg.find('=');
            middle_dir = (eq == std::string::npos) ? kDefaultMiddleDir : arg.substr(eq + 1);
            continue;
        }
        if (baseline_path.empty()) {
            baseline_path = arg;
        } else if (food_path.empty()) {
            food_path = arg;
        } else if (config_path.empty()) {
            config_path = arg;
        } else {
            std::cerr << "多余的参数: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (baseline_path.empty() || food_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    vm::log_set_level(vm::LogLevel::kInfo);
    vm::log_set_console(true);
    vm::log_set_file("pcd_im_trace.txt", vm::LogFileMode::kTruncate);

    const vm::PointCloud baseline = vm::load_pcd(baseline_path);
    const vm::PointCloud food = vm::load_pcd(food_path);
    if (baseline.points.empty() || food.points.empty()) {
        std::cerr << "点云加载失败\n";
        return 1;
    }
    std::cout << "baseline 点数       : " << baseline.points.size() << "\n";
    std::cout << "food 点数           : " << food.points.size() << "\n";

    vm::FoodVolumeMeasurer measurer;
    if (!config_path.empty() && !measurer.load_config_from_json(config_path)) {
        std::cerr << "警告: 配置加载失败(" << config_path << ")，使用内置默认值\n";
    }
    if (dump_middle) {
        measurer.set_save_middle_cloud(true).set_middle_cloud_dir(middle_dir);
    }

    const vm::VolumeEstimate est = measurer.set_baseline({baseline}).set_food(food).run();
    print_estimate(est);
    if (dump_middle && est.status == vm::MeasurementStatus::kSuccess) {
        std::cout << "\n各阶段中间点云已写入 : " << middle_dir << "/\n";
    }

    vm::log_close_file();
    return est.status == vm::MeasurementStatus::kSuccess ? 0 : 1;
}
