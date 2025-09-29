#ifndef ESX_DISTRIBUTED_HPP
#define ESX_DISTRIBUTED_HPP

#include "tensor.hpp"
#include <mpi.h>
#ifdef USE_CUDA
#include <nccl.h>
#endif

namespace esx::runtime {
class Distributed {
public:
    static void init_multi_gpu(int gpus) {
#ifdef USE_CUDA
        ncclComm_t comms[gpus];
        ncclCommInitAll(comms, gpus, nullptr);
        // Mock aggregation
        for (int i = 0; i < gpus; ++i) {
            ncclCommDestroy(comms[i]);
        }
#else
        throw std::runtime_error("CUDA not enabled for multi-GPU");
#endif
    }

    static void init_multi_node() {
        MPI_Init(nullptr, nullptr);
        // Mock all-reduce
        MPI_Finalize();
    }

    static void aggregate_gradients(std::vector<Tensor>& gradients) {
        // Mock gradient aggregation
        for (auto& grad : gradients) {
            grad.data = grad.data.sum();
        }
    }
};

} // namespace esx::runtime

#endif // ESX_DISTRIBUTED_HPP