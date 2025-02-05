#include<stdio.h>
#include<torch/types.h>



__global__ void flash_attn_kernel(const float *Q, const float *K, const float *V, float *O, int bc, int br, const int n, const int d, const int tc, const int tr,
    float *m, float *l, const float softmax_scale, float *s, float *out_q, float *out_k, float *out_o)
{
    int b = blockIdx.y;
    int h = blockIdx.x;
    int b_size = gridDim.y;
    int h_size = gridDim.x;
    
    // block.dimx = n * d

    int qkv_offset = b * h_size * n * d + h * n * d;
    int lm_offset = b * h_size * n + h * n;
    int s_offset = b * h_size * n * n + h * n * n;

    extern  __shared__ float sram[];
    //div sram to q,k,v,s
    float *sq = sram;
    float *sk = sq + br * d;
    float *sv = sk + bc * d;
    float *S = sv + bc * d;
    //float *m_row_pre = S + bc * br;
    //float *l_row_pre = m_row_pre + br;
    //float m_row_pre;
    //float l_row_pre;
    
    //TODO: take more thread 
 
    

    for(int j = 0; j < tc; j++)
    {
        //load k v to sram
        
        //sk[threadIdx.y * d + threadIdx.x] = K[qkv_offset + j * bc * d + threadIdx.y * d + threadIdx.x];
        //sv[threadIdx.y * d + threadIdx.x] = V[qkv_offset + j * bc * d + threadIdx.y * d + threadIdx.x];

        //update bc 
        if(j == tc - 1 && n % bc != 0)
        {
            bc = n % bc;
        } 
        


        for(int t =0; t < (blockDim.y + bc -1)/blockDim.y ; t++)
        {
            if(threadIdx.x < d && t * blockDim.y + threadIdx.y < bc && j * bc + t * blockDim.y + threadIdx.y < n)
            {
                sk[t * blockDim.y * d + threadIdx.y * d + threadIdx.x] = K[qkv_offset + j * bc * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x];
                sv[t * blockDim.y * d + threadIdx.y * d + threadIdx.x] = V[qkv_offset + j * bc * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x];
                //out_k[qkv_offset + j * bc * d + i * blockDim.y * d + threadIdx.y * d + threadIdx.x] = sk[i * blockDim.y * d + threadIdx.y * d + threadIdx.x]; 
            }
        }
        __syncthreads();
        //continue;
        for(int i = 0; i < tr ; i++)
        {
            if(i == tr - 1 && n % br != 0)
            {
                br = n % br;
            }
            //load q, o, l, m to sram
            //sq[threadIdx.y * d + threadIdx.x] = Q[qkv_offset + i * br * d + threadIdx.y * d + threadIdx.x];
            for(int t = 0; t < (blockDim.y + br - 1)/blockDim.y ; t++)
            {
                if(threadIdx.x < d && t * blockDim.y + threadIdx.y < br && i * br + t * blockDim.y + threadIdx.y < n)
                {
                    sq[t * blockDim.y * d + threadIdx.y * d + threadIdx.x] = Q[qkv_offset + i * br * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x];
                    //out_q[qkv_offset + i * br * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x] = sq[t * blockDim.y * d + threadIdx.y * d + threadIdx.x];
                }
            }

         
            __syncthreads();
            //continue;
            //compute score = sq * sk^T            
            for(int t = 0; t < (blockDim.y + br - 1)/blockDim.y ; t++)
            {   
                
                if(t * blockDim.y + threadIdx.y < br && i * br + t * blockDim.y + threadIdx.y < n)
                {
                    if(threadIdx.x < bc && j * bc + threadIdx.x < n)
                    {
                        float score = 0;
                        for(int k = 0; k < d; k++)
                        {
                            score += sq[t * blockDim.y * d + threadIdx.y * d + k] * sk[threadIdx.x * d + k];
                        }
                        score *= softmax_scale;
                        S[t * blockDim.y * bc + threadIdx.y * bc + threadIdx.x] = score;

                        //s[s_offset + i * br * n + (t * blockDim.y + threadIdx.y ) * n + j * bc + threadIdx.x] = score;
                    }
                }
                __syncthreads();
                float row_max;
                if(t * blockDim.y + threadIdx.y < br && i * br + t * blockDim.y + threadIdx.y < n)
                {
                    
                    row_max = S[t * blockDim.y * bc + threadIdx.y * bc];
                    for(int k = 1; k < bc && j * bc + k < n; k++)
                    {
                        if(S[t * blockDim.y * bc + threadIdx.y * bc + k] > row_max)
                        {
                            row_max = S[t * blockDim.y * bc + threadIdx.y * bc + k];
                        }
                    }
                }
                if(t * blockDim.y + threadIdx.y < br && i * br + t * blockDim.y + threadIdx.y < n)
                {
                    if(threadIdx.x < bc && j * bc + threadIdx.x < n)
                        S[t * blockDim.y * bc + threadIdx.y * bc + threadIdx.x] = __expf(S[t * blockDim.y * bc + threadIdx.y * bc + threadIdx.x] - row_max);
                }
                __syncthreads();
                if(t * blockDim.y + threadIdx.y < br && i * br + t * blockDim.y + threadIdx.y < n)  
                {
                    float row_sum = 0;  
                    for(int k = 0; k < bc && j * bc + k < n; k++)
                    {
                        //row_sum += S[threadIdx.y * bc + k];
                        row_sum += S[t * blockDim.y * bc + threadIdx.y * bc + k];
                    }

                    float m_row_pre = m[lm_offset + i * br + t * blockDim.y + threadIdx.y];
                    float l_row_pre = l[lm_offset + i * br + t * blockDim.y + threadIdx.y];
                    //update m_row_new, l_row_new
                    float m_row_new = max(m_row_pre, row_max);
                    float l_row_new = __expf(m_row_pre - m_row_new) * l_row_pre + __expf(row_max -m_row_new) * row_sum;
                    
                    if(threadIdx.x < d)
                    {
                        float pv = 0;
                        for(int k =0; k < bc && j * bc + k < n; k++)
                        {
                            //pv += S[threadIdx.y * bc + k] * sv[k * d + threadIdx.x];
                            pv += S[t * blockDim.y * bc + threadIdx.y * bc + k ] * sv[k * d + threadIdx.x];
                        }
                        O[qkv_offset + i * br * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x] = (1 / l_row_new) * (l_row_pre * __expf(m_row_pre - m_row_new) \
                            * O[qkv_offset + i * br *d + t * blockDim.y *d + threadIdx.y * d + threadIdx.x ] + __expf(row_max - m_row_new) * pv );
                        //out_o[qkv_offset + i * br * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x] = i * br * d + t * blockDim.y * d + threadIdx.y * d + threadIdx.x;
                        
                       
                    }
                    l[lm_offset + i * br + t * blockDim.y + threadIdx.y] = l_row_new;
                    m[lm_offset + i * br + t * blockDim.y + threadIdx.y] = m_row_new;
                   
                }                  

                
                __syncthreads();
            }
            __syncthreads();
           
            
               
        }
        
        __syncthreads();
    }


}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, const int max_thread_num)
{
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    //const int bc = max_thread_num / dim;
    //const int br = max_thread_num / dim;
    const int b = q.size(0);
    const int h = q.size(1);
    const int n = q.size(2);
    const int d = q.size(3);
    const int bc = ceil(float(max_sram_size) / sizeof(float) / (4 * d));
    const int br = min(bc, d);
    const int tc = ceil(float(n) / bc);
    const int tr = ceil(float(n) / br);
    const float softmax_scale = 1.0 / sqrt(d);
    //init o,l,m to hbm
    torch::Tensor o = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({b, h, n});
    torch::Tensor m = torch::full({b, h, n}, -1e9);
    torch::Tensor s = torch::zeros({b, h, n, n});
    torch::Tensor out_q = torch::zeros_like(q);
    torch::Tensor out_k = torch::zeros_like(k);
    torch::Tensor out_o = torch::zeros_like(q);
    torch::Device device(torch::kCUDA);
    //o = o.to(device);
    l = l.to(device);
    m = m.to(device);
    s = s.to(device);
    printf("q: (%d %d %d %d)\n", q.size(0), q.size(1), q.size(2), q.size(3));
    const int sram_size = br * d * sizeof(float) + bc * d * 2 * sizeof(float) + bc * br * sizeof(float) + 2 * br * sizeof(float);
    printf("bc: %d, br: %d, tc: %d, tr: %d \n", bc, br, tc, tr);    
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
    printf("Max elements: %d\n", max_sram_size / sizeof(float));
    printf("Max matrix size on shared memory:(%d %d)\n", max_sram_size / sizeof(float) / d, d);

    unsigned int block_d = max(d, bc);
    unsigned int block_h = max_thread_num / block_d;
    dim3 grid(h, b);
    printf("grid: (%d %d %d)\n", grid.x, grid.y, grid.z);
    dim3 block(block_d, block_h);
    printf("block: (%d %d)\n", block.x, block.y);


    flash_attn_kernel<<<grid, block, sram_size>>>(q.data_ptr<float>(), k.data_ptr<float>(), 
        v.data_ptr<float>(), o.data_ptr<float>(), bc, br, n, d, tc, tr, m.data_ptr<float>(),
        l.data_ptr<float>(), softmax_scale, s.data_ptr<float>(), out_q.data_ptr<float>(), out_k.data_ptr<float>(), out_o.data_ptr<float>());
    cudaDeviceSynchronize();
    return o;
}

//main
int main(int argc, char **argv)
{
    torch::Tensor q = torch::randn({16, 12, 128, 32}).to(torch::kCUDA);
    torch::Tensor k = torch::randn({16, 12, 128, 32}).to(torch::kCUDA);
    torch::Tensor v = torch::randn({16, 12, 128, 32}).to(torch::kCUDA);
    torch::Tensor s = torch::zeros({16, 12, 128, 128}).to(torch::kCUDA);
    s = forward(q, k, v, 1024);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return 0;
}
