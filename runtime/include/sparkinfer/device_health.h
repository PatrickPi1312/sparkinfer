#pragma once
#include <atomic>
#include <cuda_runtime.h>

namespace sparkinfer {

// Has the CUDA context been lost?
//
// Some CUDA errors are STICKY: once raised, the context is permanently unusable and every
// subsequent call in the process returns the same error. cudaErrorIllegalAddress is the one that
// matters here -- a single out-of-bounds access in any kernel kills the context for good.
//
// The runtime's cu() helpers used to print and continue, which is fine for a recoverable error
// (cudaErrorMemoryAllocation, say) and catastrophic for a sticky one: the process keeps issuing
// work against a dead context, every call fails, and the code proceeds on whatever uninitialised
// or stale host memory the failed copies left behind. Observed under a 16-concurrent burst on a
// nearly-full card: 21,535 identical "illegal memory access" lines followed by
// "malloc(): unaligned tcache chunk detected" and std::length_error -- i.e. the host heap was
// eventually corrupted and the server died. A server must degrade to a clean 503, not that.
//
// So: record it once, let callers ask, and refuse new work instead of grinding on. This is
// deliberately one-way -- there is no recovery short of restarting the process, and pretending
// otherwise would just produce a subtler failure later.
inline std::atomic<bool>& device_lost_flag() {
    static std::atomic<bool> lost{false};
    return lost;
}

inline bool device_lost() { return device_lost_flag().load(std::memory_order_relaxed); }

// True for errors that leave the context unusable. Allocation failures are deliberately NOT here:
// running out of memory is a normal, recoverable capacity condition that the engine already
// reports as 503/429, and treating it as fatal would take a server down for ordinary load.
inline bool is_unrecoverable(cudaError_t e) {
    switch (e) {
        case cudaErrorIllegalAddress:
        case cudaErrorMisalignedAddress:
        case cudaErrorIllegalInstruction:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorLaunchFailure:
        case cudaErrorHardwareStackError:
        case cudaErrorECCUncorrectable:
        case cudaErrorContextIsDestroyed:
        case cudaErrorDeviceUninitialized:
            return true;
        default:
            return false;
    }
}

// Call from every cu()-style wrapper. Returns true if this error killed the context.
inline bool note_cuda_error(cudaError_t e) {
    if (e == cudaSuccess) return false;
    if (!is_unrecoverable(e)) return false;
    device_lost_flag().store(true, std::memory_order_relaxed);
    return true;
}

}  // namespace sparkinfer
