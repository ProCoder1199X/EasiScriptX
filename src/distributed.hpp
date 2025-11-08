#ifndef ESX_DISTRIBUTED_HPP
#define ESX_DISTRIBUTED_HPP

#include "config.hpp"
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#if USE_SPDLOG
#include <spdlog/spdlog.h>
#endif

/**
 * @file distributed.hpp
 * @brief Distributed training support for EasiScriptX (ESX).
 * @details Provides multi-GPU (NCCL) and multi-node (MPI) distributed training
 * capabilities with fault tolerance and retry mechanisms.
 */

#if USE_MPI
#include <mpi.h>
#endif

#if USE_NCCL
#include <nccl.h>
#endif

namespace esx::runtime {

/**
 * @brief Distributed training coordinator for multi-GPU and multi-node setups.
 */
class Distributed {
public:
    static bool initialized;
    static int world_size;
    static int world_rank;
    static int local_rank;
    static int local_size;

    /**
     * @brief Initialize multi-GPU training with NCCL.
     * @param num_gpus Number of GPUs to use.
     * @return true if initialization successful, false otherwise.
     */
    static bool init_multi_gpu(int num_gpus) {
        ESX_DEBUG("DISTRIBUTED", "Initializing multi-GPU with " << num_gpus << " GPUs");
#if USE_SPDLOG
        spdlog::info("Initializing multi-GPU with {} GPUs", num_gpus);
#endif
        
#if USE_NCCL && USE_CUDA
        try {
            ncclComm_t comm;
            ncclUniqueId id;
            ncclGetUniqueId(&id);
            
            if (world_rank == 0) {
                // Broadcast NCCL ID to all processes
                for (int i = 1; i < world_size; ++i) {
                    MPI_Send(&id, sizeof(ncclUniqueId), MPI_BYTE, i, 0, MPI_COMM_WORLD);
                }
            } else {
                MPI_Recv(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            ncclCommInitRank(&comm, world_size, id, world_rank);
            comms.push_back(comm);
            initialized = true;
            
            ESX_DEBUG("DISTRIBUTED", "NCCL initialized successfully");
#if USE_SPDLOG
            spdlog::info("NCCL initialized successfully");
#endif
            return true;
        } catch (const std::exception& e) {
            ESX_DEBUG("DISTRIBUTED", "NCCL initialization failed: " << e.what());
#if USE_SPDLOG
            spdlog::error("NCCL initialization failed: {}", e.what());
#endif
            return init_cpu_fallback();
        }
#else
        ESX_DEBUG("DISTRIBUTED", "NCCL not available, using CPU fallback");
        return init_cpu_fallback();
#endif
    }

    /**
     * @brief Initialize multi-node training with MPI.
     * @param num_nodes Number of nodes to use.
     * @return true if initialization successful, false otherwise.
     */
    static bool init_multi_node(int num_nodes) {
        ESX_DEBUG("DISTRIBUTED", "Initializing multi-node with " << num_nodes << " nodes");
#if USE_SPDLOG
        spdlog::info("Initializing multi-node with {} nodes", num_nodes);
#endif
        
#if USE_MPI
        try {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            
            // Calculate local rank and size
            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
            MPI_Comm_size(local_comm, &local_size);
            MPI_Comm_rank(local_comm, &local_rank);
            
            initialized = true;
            ESX_DEBUG("DISTRIBUTED", "MPI initialized: rank " << world_rank << "/" << world_size);
#if USE_SPDLOG
            spdlog::info("MPI initialized: rank {}/{}", world_rank, world_size);
#endif
            return true;
        } catch (const std::exception& e) {
            ESX_DEBUG("DISTRIBUTED", "MPI initialization failed: " << e.what());
#if USE_SPDLOG
            spdlog::error("MPI initialization failed: {}", e.what());
#endif
            return false;
        }
#else
        ESX_DEBUG("DISTRIBUTED", "MPI not available");
        return false;
#endif
    }

    /**
     * @brief Aggregate gradients across all processes.
     * @param gradients Vector of gradient tensors to aggregate.
     * @param retry_count Number of retry attempts on failure.
     * @return true if aggregation successful, false otherwise.
     */
    static bool aggregate_gradients(std::vector<std::vector<double>>& gradients, int retry_count = 3) {
        ESX_DEBUG("DISTRIBUTED", "Aggregating gradients across " << world_size << " processes");
#if USE_SPDLOG
        spdlog::info("Aggregating gradients across {} processes", world_size);
#endif
        
        for (int retry = 0; retry < retry_count; ++retry) {
            try {
#if USE_MPI
                for (auto& grad : gradients) {
                    MPI_Allreduce(MPI_IN_PLACE, grad.data(), grad.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    // Average the gradients
                    for (auto& val : grad) {
                        val /= world_size;
                    }
                }
                ESX_DEBUG("DISTRIBUTED", "Gradient aggregation successful");
#if USE_SPDLOG
                spdlog::info("Gradient aggregation successful");
#endif
                return true;
#else
                ESX_DEBUG("DISTRIBUTED", "MPI not available, skipping gradient aggregation");
                return true;
#endif
            } catch (const std::exception& e) {
                ESX_DEBUG("DISTRIBUTED", "Gradient aggregation failed (attempt " << (retry + 1) << "): " << e.what());
#if USE_SPDLOG
                spdlog::warn("Gradient aggregation failed (attempt {}): {}", retry + 1, e.what());
#endif
                if (retry == retry_count - 1) {
                    return false;
                }
                // Wait before retry
                std::this_thread::sleep_for(std::chrono::milliseconds(100 * (retry + 1)));
            }
        }
        return false;
    }

    /**
     * @brief Synchronize all processes.
     * @return true if synchronization successful, false otherwise.
     */
    static bool synchronize() {
#if USE_MPI
        try {
            MPI_Barrier(MPI_COMM_WORLD);
#if USE_SPDLOG
            spdlog::debug("Synchronization barrier passed");
#endif
            return true;
        } catch (const std::exception& e) {
            ESX_DEBUG("DISTRIBUTED", "Synchronization failed: " << e.what());
#if USE_SPDLOG
            spdlog::warn("Synchronization failed: {}", e.what());
#endif
            return false;
        }
#else
        return true;
#endif
    }

    /**
     * @brief Finalize distributed training.
     */
    static void finalize() {
        ESX_DEBUG("DISTRIBUTED", "Finalizing distributed training");
#if USE_SPDLOG
        spdlog::info("Finalizing distributed training");
#endif
        
#if USE_NCCL
        for (auto& comm : comms) {
            ncclCommDestroy(comm);
        }
        comms.clear();
#endif

#if USE_MPI
        if (initialized) {
            MPI_Finalize();
        }
#endif
        
        initialized = false;
    }

    /**
     * @brief Get current process rank.
     * @return Process rank (0-based).
     */
    static int get_rank() { return world_rank; }

    /**
     * @brief Get total number of processes.
     * @return Total process count.
     */
    static int get_size() { return world_size; }

    /**
     * @brief Check if distributed training is initialized.
     * @return true if initialized, false otherwise.
     */
    static bool is_initialized() { return initialized; }

    /**
     * @brief Get local GPU rank within node.
     * @return Local GPU rank (0-based).
     */
    static int get_local_rank() { return local_rank; }

    /**
     * @brief Get number of GPUs on current node.
     * @return Local GPU count.
     */
    static int get_local_size() { return local_size; }

private:
    static std::vector<ncclComm_t> comms;

    /**
     * @brief CPU fallback for when GPU/NCCL is not available.
     * @return true if fallback successful, false otherwise.
     */
    static bool init_cpu_fallback() {
        ESX_DEBUG("DISTRIBUTED", "Using CPU fallback for distributed training");
        world_size = 1;
        world_rank = 0;
        local_rank = 0;
        local_size = 1;
        initialized = true;
        return true;
    }
};

// Static member definitions
bool Distributed::initialized = false;
int Distributed::world_size = 1;
int Distributed::world_rank = 0;
int Distributed::local_rank = 0;
int Distributed::local_size = 1;
std::vector<ncclComm_t> Distributed::comms;

/**
 * @brief Gradient checkpointing for memory-efficient training.
 * @param model Model tensor to checkpoint.
 * @param segments Number of segments to checkpoint.
 * @return true if checkpointing successful, false otherwise.
 */
inline bool checkpoint_gradients(const std::vector<std::vector<double>>& model, int segments) {
    ESX_DEBUG("DISTRIBUTED", "Checkpointing gradients with " << segments << " segments");
    
    if (segments <= 0) {
        ESX_DEBUG("DISTRIBUTED", "Invalid segment count: " << segments);
        return false;
    }
    
    // Mock gradient checkpointing implementation
    // In full implementation, would save intermediate activations
    // and recompute them during backward pass
    ESX_DEBUG("DISTRIBUTED", "Gradient checkpointing completed");
    return true;
}

/**
 * @brief Energy-aware training with RAPL monitoring.
 * @param max_power Maximum power consumption in watts.
 * @return Current power consumption.
 */
inline double monitor_power_consumption(double max_power) {
    ESX_DEBUG("DISTRIBUTED", "Monitoring power consumption (max: " << max_power << "W)");
    
    // Mock RAPL (Running Average Power Limit) monitoring
    // In full implementation, would read from /sys/class/powercap/
    double current_power = max_power * 0.7; // Mock 70% utilization
    
    if (current_power > max_power) {
        ESX_DEBUG("DISTRIBUTED", "Power limit exceeded: " << current_power << "W > " << max_power << "W");
    }
    
    return current_power;
}

} // namespace esx::runtime

#endif // ESX_DISTRIBUTED_HPP