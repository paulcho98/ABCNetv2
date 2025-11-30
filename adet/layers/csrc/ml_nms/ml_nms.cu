// adet/layers/csrc/ml_nms/ml_nms.cu
// Copyright (c) Facebook, Inc.
// Modernized for PyTorch 2.x / CUDA 12.x (remove THC, use ATen/c10)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/ceil_div.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(const float* __restrict__ a,
                               const float* __restrict__ b) {
  if (a[5] != b[5]) {
    return 0.0f;
  }
  float left   = max(a[0], b[0]);
  float right  = min(a[2], b[2]);
  float top    = max(a[1], b[1]);
  float bottom = min(a[3], b[3]);
  float width  = max(right - left + 1.0f, 0.0f);
  float height = max(bottom - top + 1.0f, 0.0f);
  float interS = width * height;
  float Sa     = (a[2] - a[0] + 1.0f) * (a[3] - a[1] + 1.0f);
  float Sb     = (b[2] - b[0] + 1.0f) * (b[3] - b[1] + 1.0f);
  return interS / (Sa + Sb - interS);
}

__global__ void ml_nms_kernel(const int n_boxes,
                              const float nms_overlap_thresh,
                              const float* __restrict__ dev_boxes,
                              unsigned long long* __restrict__ dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 6];
  if (threadIdx.x < col_size) {
    const int base = (threadsPerBlock * col_start + threadIdx.x) * 6;
    block_boxes[threadIdx.x * 6 + 0] = dev_boxes[base + 0];
    block_boxes[threadIdx.x * 6 + 1] = dev_boxes[base + 1];
    block_boxes[threadIdx.x * 6 + 2] = dev_boxes[base + 2];
    block_boxes[threadIdx.x * 6 + 3] = dev_boxes[base + 3];
    block_boxes[threadIdx.x * 6 + 4] = dev_boxes[base + 4];
    block_boxes[threadIdx.x * 6 + 5] = dev_boxes[base + 5];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float* cur_box  = dev_boxes + cur_box_idx * 6;

    unsigned long long t = 0ULL;
    int start = (row_start == col_start) ? (threadIdx.x + 1) : 0;

    for (int i = start; i < col_size; ++i) {
      if (devIoU(cur_box, block_boxes + i * 6) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }

    const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

namespace adet {

// boxes is an [N x 6] CUDA tensor (x1,y1,x2,y2,score,category)
at::Tensor ml_nms_cuda(const at::Tensor boxes, const float nms_overlap_thresh) {
  using scalar_t = float;

  TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 6,
              "boxes must have shape [N, 6]");

  auto scores      = boxes.select(/*dim=*/1, /*index=*/4);
  auto order_t     = std::get<1>(scores.sort(/*dim=*/0, /*descending=*/true));
  auto boxes_sorted= boxes.index_select(/*dim=*/0, order_t);

  const int boxes_num = static_cast<int>(boxes.size(0));
  const int col_blocks= (boxes_num + threadsPerBlock - 1) / threadsPerBlock;

  scalar_t* boxes_dev = boxes_sorted.data_ptr<scalar_t>();

  // Device allocation via PyTorch's caching allocator (no THC)
  const size_t bytes = static_cast<size_t>(boxes_num) *
                       static_cast<size_t>(col_blocks) *
                       sizeof(unsigned long long);
  auto* mask_dev = static_cast<unsigned long long*>(
      c10::cuda::CUDACachingAllocator::raw_alloc(bytes));

  dim3 blocks((boxes_num + threadsPerBlock - 1) / threadsPerBlock,
              (boxes_num + threadsPerBlock - 1) / threadsPerBlock);
  dim3 threads(threadsPerBlock);

  ml_nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev);
  AT_CUDA_CHECK(cudaGetLastError());

  std::vector<unsigned long long> mask_host(static_cast<size_t>(boxes_num) * col_blocks);
  AT_CUDA_CHECK(cudaMemcpy(mask_host.data(),
                           mask_dev,
                           sizeof(unsigned long long) * mask_host.size(),
                           cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks, 0ULL);

  at::Tensor keep = at::empty({boxes_num},
                              boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; ++i) {
    const int nblock  = i / threadsPerBlock;
    const int inblock = i % threadsPerBlock;
    if ((remv[nblock] & (1ULL << inblock)) == 0ULL) {
      keep_out[num_to_keep++] = i;
      const unsigned long long* p = mask_host.data() + static_cast<size_t>(i) * col_blocks;
      for (int j = nblock; j < col_blocks; ++j) {
        remv[j] |= p[j];
      }
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);

  // Map kept indices (in sorted order) back to original order
  return std::get<0>(
           order_t.index({ keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
                                .to(order_t.device(), keep.scalar_type()) })
                 .sort(/*dim=*/0, /*descending=*/false));
}

} // namespace adet
