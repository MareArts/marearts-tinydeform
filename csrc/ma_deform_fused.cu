// Copyright 2026 MareArts
// SPDX-License-Identifier: Apache-2.0
//
// Fused multi-scale deformable attention for inference: value projection + bilinear sampling + weighted sum.
//
// The standard decoder computes value = memory @ W^T + b for every memory token, then samples value bilinearly at
// a few locations per (query, head). Sampling and the weighted sum are linear, so for one (query, head):
//
//   out[d] = sum_s w_s * (W[d] . memory[t_s] + b[d])  =  W[d] . (sum_s w_s * memory[t_s])  +  b[d] * sum_s w_s
//
// where s runs over levels x points x 4 bilinear corners inside the map (corners outside count 0, as zero padding
// does). fused_forward builds m = sum_s w_s * memory[t_s] (C values) and projects it, so the cost follows
// queries x samples instead of tokens. Its sample reads are C wide instead of D, so with many queries the
// project-first order wins again; sample_forward is that order (value projected by the caller, e.g. cuBLAS).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr int kMaxLevelPoints = 64;  // levels x points per head
constexpr int kMaxChannels = 8192;   // shared memory holds m[C] as float

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

// 16-byte loads converted to float, for the vector kernel
template <typename T>
struct Vec16;
template <>
struct Vec16<float> {
    static constexpr int n = 4;
    __device__ __forceinline__ static void load(const float* p, float* o) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(p));
        o[0] = v.x, o[1] = v.y, o[2] = v.z, o[3] = v.w;
    }
};
template <>
struct Vec16<at::Half> {
    static constexpr int n = 8;
    __device__ __forceinline__ static void load(const at::Half* p, float* o) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
        const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h[i]);
            o[2 * i] = f.x, o[2 * i + 1] = f.y;
        }
    }
};
template <>
struct Vec16<at::BFloat16> {
    static constexpr int n = 8;
    __device__ __forceinline__ static void load(const at::BFloat16* p, float* o) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(h[i]);
            o[2 * i] = f.x, o[2 * i + 1] = f.y;
        }
    }
};

struct Args {
    const int64_t* shapes;  // [L, 2] (height, width)
    const int64_t* starts;  // [L] first token of each level
    int Q, N, C, H, D, L, P;
};

// The 4 corner tokens and weights (attention weight x bilinear weight, align_corners=False) of sample j of row
// bqh. Corners outside the map, or outside [0, N), get weight 0 and token 0.
template <typename scalar_t>
__device__ __forceinline__ void corner4(const scalar_t* __restrict__ loc, const scalar_t* __restrict__ attn,
                                        const Args& a, size_t bqh, int j, int* tok, float* wt) {
    const int LP = a.L * a.P, l = j / a.P;
    const int hl = int(a.shapes[2 * l]), wl = int(a.shapes[2 * l + 1]), st = int(a.starts[l]);
    const float w = to_float(attn[bqh * LP + j]);
    // clamp first so far-away locations cannot overflow the int conversion; they sample nothing either way
    const float x = fminf(fmaxf(to_float(loc[(bqh * LP + j) * 2]) * wl - 0.5f, -2.f), wl + 1.f);
    const float y = fminf(fmaxf(to_float(loc[(bqh * LP + j) * 2 + 1]) * hl - 0.5f, -2.f), hl + 1.f);
    const float xf = floorf(x), yf = floorf(y);
    const int x0 = int(xf), y0 = int(yf);
    const float fx = x - xf, fy = y - yf;
    for (int k = 0; k < 4; ++k) {
        const int dx = k & 1, dy = k >> 1, xi = x0 + dx, yi = y0 + dy;
        const int t = st + yi * wl + xi;
        const bool in = xi >= 0 && xi < wl && yi >= 0 && yi < hl && t >= 0 && t < a.N;
        tok[k] = in ? t : 0;
        wt[k] = in ? w * (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy) : 0.f;
    }
}

// All corners of block row bqh into tok/wt and their weight sum into wsum. Ends after a __syncthreads().
template <typename scalar_t>
__device__ __forceinline__ void corners(const scalar_t* __restrict__ loc, const scalar_t* __restrict__ attn,
                                        const Args& a, size_t bqh, int* tok, float* wt, float* wsum) {
    const int LP = a.L * a.P;
    for (int j = threadIdx.x; j < LP; j += blockDim.x) corner4(loc, attn, a, bqh, j, tok + j * 4, wt + j * 4);
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int k = 0; k < LP * 4; ++k) s += wt[k];
        *wsum = s;
    }
}

// Any dtype, any C, any alignment: one channel per thread, one output row per warp.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads)
    fused_scalar(const scalar_t* __restrict__ memory, const scalar_t* __restrict__ weight,
                 const scalar_t* __restrict__ bias, const scalar_t* __restrict__ loc,
                 const scalar_t* __restrict__ attn, scalar_t* __restrict__ out, Args a) {
    extern __shared__ float m[];  // [C]
    __shared__ int tok[kMaxLevelPoints * 4];
    __shared__ float wt[kMaxLevelPoints * 4];
    __shared__ float wsum;
    const size_t bqh = blockIdx.x;  // (b * Q + q) * H + h
    const int h = int(bqh % a.H), b = int(bqh / a.H / a.Q), S = a.L * a.P * 4, C = a.C;
    corners(loc, attn, a, bqh, tok, wt, &wsum);

    const scalar_t* mb = memory + size_t(b) * a.N * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < S; ++s) acc = fmaf(wt[s], to_float(mb[size_t(tok[s]) * C + c]), acc);
        m[c] = acc;
    }
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int d = warp; d < a.D; d += warps) {
        const scalar_t* wr = weight + size_t(h * a.D + d) * C;
        float acc = 0.f;
        for (int c = lane; c < C; c += 32) acc = fmaf(to_float(wr[c]), m[c], acc);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            if (bias != nullptr) acc = fmaf(to_float(bias[h * a.D + d]), wsum, acc);
            out[bqh * a.D + d] = static_cast<scalar_t>(acc);
        }
    }
}

// float / half / bfloat16 with C % (16 bytes / element) == 0 and 16-byte aligned memory and weight:
// 16-byte loads, and the samples split across thread groups so more rows are in flight.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads)
    fused_vec(const scalar_t* __restrict__ memory, const scalar_t* __restrict__ weight,
              const scalar_t* __restrict__ bias, const scalar_t* __restrict__ loc, const scalar_t* __restrict__ attn,
              scalar_t* __restrict__ out, Args a) {
    constexpr int V = Vec16<scalar_t>::n;
    extern __shared__ float part[];  // [groups][C]; part[0..C) ends up holding m
    __shared__ int tok[kMaxLevelPoints * 4];
    __shared__ float wt[kMaxLevelPoints * 4];
    __shared__ float wsum;
    const size_t bqh = blockIdx.x;
    const int h = int(bqh % a.H), b = int(bqh / a.H / a.Q), S = a.L * a.P * 4, C = a.C, CV = C / V;
    const int groups = CV >= kThreads ? 1 : kThreads / CV;
    corners(loc, attn, a, bqh, tok, wt, &wsum);

    const scalar_t* mb = memory + size_t(b) * a.N * C;
    const int g = int(threadIdx.x) / CV;  // 0 for every thread when groups == 1
    if (g < groups) {
        for (int ch = int(threadIdx.x) % (groups == 1 ? kThreads : CV); ch < CV; ch += (groups == 1 ? kThreads : CV)) {
            float acc[V] = {};
            for (int s = g; s < S; s += groups) {
                float v[V];
                Vec16<scalar_t>::load(mb + size_t(tok[s]) * C + ch * V, v);
                const float w = wt[s];
#pragma unroll
                for (int i = 0; i < V; ++i) acc[i] = fmaf(w, v[i], acc[i]);
            }
#pragma unroll
            for (int i = 0; i < V; ++i) part[g * C + ch * V + i] = acc[i];
        }
    }
    __syncthreads();
    if (groups > 1) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            float s = part[c];
            for (int k = 1; k < groups; ++k) s += part[k * C + c];
            part[c] = s;
        }
        __syncthreads();
    }

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int d = warp; d < a.D; d += warps) {
        const scalar_t* wr = weight + size_t(h * a.D + d) * C;
        float acc = 0.f;
        for (int ch = lane; ch < CV; ch += 32) {
            float w[V];
            Vec16<scalar_t>::load(wr + ch * V, w);
#pragma unroll
            for (int i = 0; i < V; ++i) acc = fmaf(w[i], part[ch * V + i], acc);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            if (bias != nullptr) acc = fmaf(to_float(bias[h * a.D + d]), wsum, acc);
            out[bqh * a.D + d] = static_cast<scalar_t>(acc);
        }
    }
}

// Project-first path: value = value_proj(memory) is given as [B, N, H * D]. One warp per (batch, query, head) row,
// lanes over that head's D channels; the same sampling rules as the fused kernels.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads)
    sample_kernel(const scalar_t* __restrict__ value, const scalar_t* __restrict__ loc,
                  const scalar_t* __restrict__ attn, scalar_t* __restrict__ out, Args a, int64_t rows) {
    constexpr int kRows = kThreads / 32;
    __shared__ int tok[kRows][kMaxLevelPoints * 4];
    __shared__ float wt[kRows][kMaxLevelPoints * 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const size_t bqh = size_t(blockIdx.x) * kRows + warp;
    if (bqh >= size_t(rows)) return;  // the whole warp leaves; only __syncwarp below
    const int LP = a.L * a.P, S = LP * 4, h = int(bqh % a.H), b = int(bqh / a.H / a.Q), HD = a.H * a.D;
    for (int j = lane; j < LP; j += 32) corner4(loc, attn, a, bqh, j, tok[warp] + j * 4, wt[warp] + j * 4);
    __syncwarp();
    const scalar_t* vb = value + size_t(b) * a.N * HD + size_t(h) * a.D;
    for (int d = lane; d < a.D; d += 32) {
        float acc = 0.f;
        for (int s = 0; s < S; ++s) acc = fmaf(wt[warp][s], to_float(vb[size_t(tok[warp][s]) * HD + d]), acc);
        out[bqh * a.D + d] = static_cast<scalar_t>(acc);
    }
}

// Shape, dtype and device checks shared by both entry points. Returns B, Q, H, L, P.
std::array<int64_t, 5> check_geometry(const torch::Tensor& ref, const torch::Tensor& spatial_shapes,
                                      const torch::Tensor& level_start_index, const torch::Tensor& sampling_locations,
                                      const torch::Tensor& attention_weights) {
    TORCH_CHECK(ref.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(ref.dim() == 3, "memory / value must be [B, N, C]");
    TORCH_CHECK(ref.size(1) <= INT32_MAX, "too many memory tokens");
    TORCH_CHECK(sampling_locations.dim() == 6 && sampling_locations.size(5) == 2,
                "sampling_locations must be [B, Q, H, L, P, 2]");
    TORCH_CHECK(attention_weights.dim() == 5, "attention_weights must be [B, Q, H, L, P]");
    TORCH_CHECK(attention_weights.sizes() == sampling_locations.sizes().slice(0, 5),
                "attention_weights must be [B, Q, H, L, P] like sampling_locations");
    const int64_t B = sampling_locations.size(0), Q = sampling_locations.size(1), H = sampling_locations.size(2);
    const int64_t L = sampling_locations.size(3), P = sampling_locations.size(4);
    TORCH_CHECK(ref.size(0) == B, "batch size mismatch");
    TORCH_CHECK(H > 0, "need at least one head");
    TORCH_CHECK(L * P <= kMaxLevelPoints, "levels x points must be <= ", kMaxLevelPoints);
    TORCH_CHECK(spatial_shapes.scalar_type() == torch::kLong && spatial_shapes.dim() == 2 &&
                    spatial_shapes.size(0) == L && spatial_shapes.size(1) == 2,
                "spatial_shapes must be int64 [L, 2]");
    TORCH_CHECK(level_start_index.scalar_type() == torch::kLong && level_start_index.dim() == 1 &&
                    level_start_index.size(0) == L,
                "level_start_index must be int64 [L]");
    for (const auto* t : {&sampling_locations, &attention_weights, &spatial_shapes, &level_start_index}) {
        TORCH_CHECK(t->device() == ref.device(), "all tensors must be on the same device");
    }
    for (const auto* t : {&sampling_locations, &attention_weights}) {
        TORCH_CHECK(t->scalar_type() == ref.scalar_type(), "all float tensors must have the same dtype");
    }
    TORCH_CHECK(B * Q * H <= INT32_MAX, "too many (batch, query, head) rows");
    return {B, Q, H, L, P};
}

template <typename scalar_t>
void launch(const torch::Tensor& mem, const torch::Tensor& w, const torch::Tensor& bs, const torch::Tensor& lc,
            const torch::Tensor& aw, torch::Tensor& out, const Args& args, int64_t blocks, cudaStream_t stream) {
    const scalar_t* bp = bs.defined() ? bs.data_ptr<scalar_t>() : nullptr;
    const int64_t C = args.C;
    if constexpr (!std::is_same_v<scalar_t, double>) {
        constexpr int V = Vec16<scalar_t>::n;
        if (C % V == 0 && reinterpret_cast<uintptr_t>(mem.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0) {
            const int64_t CV = C / V, groups = CV >= kThreads ? 1 : kThreads / CV;
            fused_vec<scalar_t><<<unsigned(blocks), kThreads, size_t(groups * C) * sizeof(float), stream>>>(
                mem.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), bp, lc.data_ptr<scalar_t>(), aw.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), args);
            return;
        }
    }
    fused_scalar<scalar_t><<<unsigned(blocks), kThreads, size_t(C) * sizeof(float), stream>>>(
        mem.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), bp, lc.data_ptr<scalar_t>(), aw.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(), args);
}

}  // namespace

torch::Tensor fused_forward(const torch::Tensor& memory, const torch::Tensor& weight,
                            const c10::optional<torch::Tensor>& bias, const torch::Tensor& spatial_shapes,
                            const torch::Tensor& level_start_index, const torch::Tensor& sampling_locations,
                            const torch::Tensor& attention_weights) {
    const auto [B, Q, H, L, P] =
        check_geometry(memory, spatial_shapes, level_start_index, sampling_locations, attention_weights);
    const int64_t N = memory.size(1), C = memory.size(2);
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == C, "value_weight must be [H * D, C] with C = memory.size(2)");
    const int64_t HD = weight.size(0);
    TORCH_CHECK(HD % H == 0, "value_weight rows must be a multiple of the head count");
    TORCH_CHECK(C <= kMaxChannels, "channels must be <= ", kMaxChannels);
    TORCH_CHECK(weight.device() == memory.device() && weight.scalar_type() == memory.scalar_type(),
                "value_weight must have the device and dtype of memory");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == memory.device() && bias->scalar_type() == memory.scalar_type() &&
                        bias->dim() == 1 && bias->size(0) == HD,
                    "value_bias must be [H * D] like value_weight, same device and dtype");
    }

    const c10::cuda::CUDAGuard guard(memory.device());
    auto mem = memory.contiguous(), w = weight.contiguous(), lc = sampling_locations.contiguous();
    auto aw = attention_weights.contiguous(), sh = spatial_shapes.contiguous(), st = level_start_index.contiguous();
    torch::Tensor bs = bias.has_value() ? bias->contiguous() : torch::Tensor();
    auto out = torch::empty({B, Q, HD}, memory.options());
    const int64_t blocks = B * Q * H;
    if (blocks == 0) return out;
    const Args args{sh.data_ptr<int64_t>(), st.data_ptr<int64_t>(), int(Q), int(N), int(C),
                    int(H), int(HD / H), int(L), int(P)};
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, mem.scalar_type(), "tinydeform_fused",
                                    [&] { launch<scalar_t>(mem, w, bs, lc, aw, out, args, blocks, stream); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor sample_forward(const torch::Tensor& value, int64_t heads, const torch::Tensor& spatial_shapes,
                             const torch::Tensor& level_start_index, const torch::Tensor& sampling_locations,
                             const torch::Tensor& attention_weights) {
    const auto [B, Q, H, L, P] =
        check_geometry(value, spatial_shapes, level_start_index, sampling_locations, attention_weights);
    const int64_t N = value.size(1), HD = value.size(2);
    TORCH_CHECK(heads == H && HD % H == 0, "value must be [B, N, H * D] with H = sampling_locations.size(2)");
    const c10::cuda::CUDAGuard guard(value.device());
    auto v = value.contiguous(), lc = sampling_locations.contiguous(), aw = attention_weights.contiguous();
    auto sh = spatial_shapes.contiguous(), st = level_start_index.contiguous();
    auto out = torch::empty({B, Q, HD}, value.options());
    const int64_t rows = B * Q * H;
    if (rows == 0) return out;
    const Args args{sh.data_ptr<int64_t>(), st.data_ptr<int64_t>(), int(Q), int(N), int(HD),
                    int(H), int(HD / H), int(L), int(P)};
    constexpr int kRows = kThreads / 32;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, v.scalar_type(), "tinydeform_sample", [&] {
        sample_kernel<scalar_t><<<unsigned((rows + kRows - 1) / kRows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            v.data_ptr<scalar_t>(), lc.data_ptr<scalar_t>(), aw.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), args,
            rows);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "fused value projection + multi-scale deformable attention (CUDA)");
    m.def("sample_forward", &sample_forward, "multi-scale deformable attention on projected values (CUDA)");
}
