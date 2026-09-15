// 性能基准：测 `FoodVolumeMeasurer::run()` 的端到端耗时与进程峰值内存。
//
// 用法:
//   bench <baseline.pcd> <food.pcd> [runs] [config.json]
//
// 说明:
//   runs         重复次数，默认 1；报告单次均值与总计
//   config.json  可选，MeasurementConfig 的 JSON；省略则用内置默认值
//
// 点云在计时开始前一次性载入，因此测量结果不含磁盘 I/O；进程峰值 RSS 反映
// 整个进程（含依赖库）的水线。

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <sys/resource.h>

#include "volume_log.hpp"
#include "volume_measurement.hpp"



namespace {

using Clock = std::chrono::steady_clock;

double wall_ms(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double cpu_ms(const rusage& a, const rusage& b) {
    const double u =
        (b.ru_utime.tv_sec - a.ru_utime.tv_sec) * 1000.0 + (b.ru_utime.tv_usec - a.ru_utime.tv_usec) / 1000.0;
    const double s =
        (b.ru_stime.tv_sec - a.ru_stime.tv_sec) * 1000.0 + (b.ru_stime.tv_usec - a.ru_stime.tv_usec) / 1000.0;
    return u + s;
}

double cpu_ms_since(const rusage& a) {
    rusage now;
    getrusage(RUSAGE_SELF, &now);
    return cpu_ms(a, now);
}

void print_row(const std::string& name, double wall, double cpu) {
    std::cout << std::left << std::setw(24) << name << std::right << std::setw(12) << std::fixed
              << std::setprecision(2) << wall << std::setw(12) << cpu << "\n";
}

} // namespace



int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <baseline.pcd> <food.pcd> [runs] [config.json]\n";
        return 1;
    }
    const std::string baseline_path = argv[1];
    const std::string food_path = argv[2];
    const int runs = (argc >= 4) ? std::atoi(argv[3]) : 1;
    if (runs < 1) {
        std::cerr << "runs 必须 >= 1\n";
        return 1;
    }
    const std::string config_path = (argc >= 5) ? argv[4] : "";

    // 关闭日志输出，避免 I/O 干扰计时。
    vm::log_set_level(vm::LogLevel::kError);
    vm::log_set_console(false);

    // 计时前一次性载入点云，只测量算法本身。
    const vm::PointCloud baseline = vm::load_pcd(baseline_path);
    const vm::PointCloud food = vm::load_pcd(food_path);
    if (baseline.points.empty() || food.points.empty()) {
        std::cerr << "点云加载失败\n";
        return 1;
    }

    vm::FoodVolumeMeasurer measurer;
    if (!config_path.empty() && !measurer.load_config_from_json(config_path)) {
        std::cerr << "警告: 配置加载失败(" << config_path << ")，使用内置默认值\n";
    }
    measurer.set_baseline({baseline}).set_food(food);

    //measurer.save_config_to_json("param.json");
    measurer.set_save_middle_cloud(true);
    measurer.set_middle_cloud_dir("middle_clouds");

    double total_wall = 0.0;
    double total_cpu = 0.0;
    vm::VolumeEstimate last{};
    for (int i = 0; i < runs; ++i) {
        rusage cpu0;
        getrusage(RUSAGE_SELF, &cpu0);
        const Clock::time_point t0 = Clock::now();

        last = measurer.run();

        total_wall += wall_ms(t0, Clock::now());
        total_cpu += cpu_ms_since(cpu0);
        if (i == 0 && last.status != vm::MeasurementStatus::kSuccess) {
            std::cerr << "测量失败: status=" << vm::status_to_string(last.status) << " (" << last.message << ")\n";
        }
    }

    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    const double max_rss_mb = static_cast<double>(ru.ru_maxrss) / 1024.0; // Linux 下 ru_maxrss 单位为 KB

    std::cout << "\n===== food_volume_measure 端到端基准 (runs=" << runs << ") =====\n";
    std::cout << std::left << std::setw(24) << "指标" << std::right << std::setw(12) << "wall(ms)"
              << std::setw(12) << "cpu(ms)" << "\n";
    std::cout << std::string(48, '-') << "\n";
    print_row("单次 run() 均值", total_wall / runs, total_cpu / runs);
    print_row("总计", total_wall, total_cpu);
    std::cout << std::string(48, '-') << "\n";
    std::cout << "输入点数: " << last.input_points << " | 降采样后: " << last.downsampled_points
              << " | 选中块数: " << last.component_count << "\n";
    std::cout << std::fixed << std::setprecision(3) << "体积: " << last.volume_cm3 << " cm^3\n";
    std::cout << "峰值内存 max_rss: " << std::setprecision(2) << max_rss_mb << " MB\n";

    return last.status == vm::MeasurementStatus::kSuccess ? 0 : 1;
}
