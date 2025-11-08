#ifndef ESX_ENERGY_MONITOR_HPP
#define ESX_ENERGY_MONITOR_HPP

#include "config.hpp"
#include <string>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <cstdint>
#if USE_CUDA
#include <nvml.h>
#endif

/**
 * @file energy_monitor.hpp
 * @brief Real energy monitoring using RAPL (CPU) and NVML (GPU).
 * @details Provides energy monitoring for 15-25% energy savings validation.
 */

namespace esx::runtime {

/**
 * @brief Energy monitor using RAPL for CPU and NVML for GPU.
 */
class EnergyMonitor {
public:
    EnergyMonitor() {
        initialize_rapl();
        initialize_nvml();
    }
    
    ~EnergyMonitor() {
        cleanup_nvml();
    }
    
    /**
     * @brief Get current CPU energy consumption (Joules) using RAPL.
     * @return Energy consumption in Joules, or 0.0 if RAPL not available.
     */
    double get_cpu_energy() {
        if (!rapl_available) {
            return 0.0;
        }
        
        try {
            // Try to read RAPL energy from /sys/class/powercap (Linux only)
            std::ifstream energy_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
            if (energy_file.is_open()) {
                uint64_t energy_uj = 0;
                energy_file >> energy_uj;
                energy_file.close();
                
                // Convert microjoules to joules
                double energy_j = static_cast<double>(energy_uj) / 1e6;
                
                // Calculate delta since last reading
                if (last_cpu_energy > 0) {
                    double delta = energy_j - last_cpu_energy;
                    last_cpu_energy = energy_j;
                    return delta > 0 ? delta : 0.0;
                } else {
                    last_cpu_energy = energy_j;
                    return 0.0;
                }
            }
        } catch (...) {
            // RAPL not available (Windows/Mac or no permissions), return 0
        }
        
        return 0.0;
    }
    
    /**
     * @brief Get current GPU energy consumption (Joules) using NVML.
     * @return Energy consumption in Joules, or 0.0 if NVML not available.
     */
    double get_gpu_energy() {
#if USE_CUDA
        if (!nvml_available) {
            return 0.0;
        }
        
        try {
            unsigned int device_count = 0;
            nvmlReturn_t result = nvmlDeviceGetCount(&device_count);
            if (result != NVML_SUCCESS || device_count == 0) {
                return 0.0;
            }
            
            nvmlDevice_t device = nullptr;
            result = nvmlDeviceGetHandleByIndex(0, &device);
            if (result != NVML_SUCCESS) {
                return 0.0;
            }
            
            unsigned long long energy = 0;
            result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);
            if (result == NVML_SUCCESS) {
                // Convert millijoules to joules
                double energy_j = static_cast<double>(energy) / 1000.0;
                
                // Calculate delta since last reading
                if (last_gpu_energy > 0) {
                    double delta = energy_j - last_gpu_energy;
                    last_gpu_energy = energy_j;
                    return delta > 0 ? delta : 0.0;
                } else {
                    last_gpu_energy = energy_j;
                    return 0.0;
                }
            }
        } catch (...) {
            // NVML not available
        }
#endif
        return 0.0;
    }
    
    /**
     * @brief Get total energy consumption (CPU + GPU).
     * @return Total energy in Joules.
     */
    double get_total_energy() {
        return get_cpu_energy() + get_gpu_energy();
    }
    
    /**
     * @brief Monitor energy for a duration and return average power.
     * @param duration_seconds Duration to monitor in seconds.
     * @return Average power consumption in Watts.
     */
    double monitor_power(double duration_seconds = 1.0) {
        double initial_energy = get_total_energy();
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(duration_seconds * 1000)));
        double final_energy = get_total_energy();
        return (final_energy - initial_energy) / duration_seconds;
    }
    
private:
    bool rapl_available = false;
    bool nvml_available = false;
    double last_cpu_energy = 0.0;
    double last_gpu_energy = 0.0;
    
    void initialize_rapl() {
        // Check if RAPL is available
        std::ifstream test_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
        rapl_available = test_file.is_open();
        test_file.close();
    }
    
    void initialize_nvml() {
#if USE_CUDA
        try {
            nvmlReturn_t result = nvmlInit();
            if (result == NVML_SUCCESS) {
                nvml_available = true;
            }
        } catch (const std::exception& e) {
            nvml_available = false;
        }
#endif
    }
    
    void cleanup_nvml() {
#if USE_CUDA
        if (nvml_available) {
            nvmlShutdown();
        }
#endif
    }
    
#if USE_CUDA && defined(NVML_API_VERSION)
    // NVML is available - use real API
#else
    // NVML stubs for when not available
    typedef void* nvmlDevice_t;
    typedef int nvmlReturn_t;
    enum { NVML_SUCCESS = 0 };
    
    inline nvmlReturn_t nvmlInit() { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetCount(unsigned int* count) { *count = 0; return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) { return NVML_SUCCESS; }
    inline nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) { 
        *energy = 0; return NVML_SUCCESS; 
    }
    inline void nvmlShutdown() {}
#endif
};

} // namespace esx::runtime

#endif // ESX_ENERGY_MONITOR_HPP

