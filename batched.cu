#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define SH_PER_TB 32768

REGISTER_OP("Bspmm")
    .Attr("adjoint_a: bool")
    .Attr("adjoint_b: bool")
    .Attr("TI: list(type)")
    .Attr("TV: list(type)")
    .Input("sp_ids: TI")
    .Input("sp_values: TV")
    .Input("sp_shape: TI")
    .Input("rhs: TV")
    .Output("out: TV")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			int numTensor = c->num_outputs();
			for (int i = 0; i < numTensor; ++i) {
				::tensorflow::shape_inference::ShapeHandle sp_shape_shape;
				TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(numTensor * 2 + i, &sp_shape_shape));
				c->set_output(i, c->Matrix(c->Dim(sp_shape_shape, 0), c->Dim(c->input(numTensor * 3 + i), 1)));
			}
			return Status::OK();
		});

REGISTER_OP("Bspmdt")
    .Attr("adjoint_a: bool")
    .Attr("adjoint_b: bool")
    .Attr("TI: list(type)")
    .Attr("TV: list(type)")
    .Input("sp_ids: TI")
    .Input("sp_values: TV")
    .Input("sp_shape: TI")
    .Input("rhs: float")
    .Output("out: TV")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
			int numTensor = c->num_outputs();
			for (int i = 0; i < numTensor; ++i) {
				::tensorflow::shape_inference::ShapeHandle sp_shape_shape;
				TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(numTensor * 2 + i, &sp_shape_shape));
				c->set_output(i, c->Matrix(c->Dim(sp_shape_shape, 0), c->Dim(c->input(numTensor * 3), 1)));
			}
			return Status::OK();
		});

/* CUDA kernel: Initialize output tensors (matrices) */
template <typename idType, typename valType>
__global__ void BatchedInitOutputs(valType **d_out, const idType* __restrict__ d_outRows,
								  const idType nvector)
{
	int target = blockIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d_outRows[target] * nvector) return;
	d_out[target][i] = 0;
}

/*
  CUDA kernel: Compute sparse tensor x dense matrix of batched tensors (matrices)
  One CUDA kernel for all SpMMs
  One thread block for one SpMM
  One subTB (1, 2, ..., 32 threads) for one non-zero
*/
template <typename idType, typename valType, int subTB, int adjoint_a>
__global__ void BatchedSpMM(const idType* __restrict__ d_outNnz,
							idType **d_sp_ids, valType **d_sp_values, valType **d_rhs, valType **d_out,
							const idType nvector)
{
	idType targetTensor = blockIdx.y;
	idType i = blockIdx.x * blockDim.x + threadIdx.x;
	idType targetNnz = i / subTB;
	if (targetNnz >= d_outNnz[targetTensor]) return;
	
	idType row_id = d_sp_ids[targetTensor][2 * targetNnz + adjoint_a] * nvector;
	idType column_id = d_sp_ids[targetTensor][2 * targetNnz + (1 - adjoint_a)] * nvector;
	valType val = d_sp_values[targetTensor][targetNnz];
	for (idType localN = i & (subTB - 1); localN < nvector; localN += subTB) {
		atomicAdd(d_out[targetTensor] + row_id + localN, val * d_rhs[targetTensor][column_id + localN]);
	}
}

/*
  CUDA kernel: Compute sparse tensor x dense matrix of batched tensors (matrices)
  One CUDA kernel for all SpMMs
  One thread block for one SpMM
  One subTB (1, 2, ..., 32 threads) for one non-zero
*/
template <typename idType, typename valType, int sh_size, int subTB, int adjoint_a>
__global__ void BatchedSpMM_small(const idType* __restrict__ d_outRows,
								  const idType* __restrict__ d_nnz,
								  idType **d_sp_ids, valType **d_sp_values, valType **d_rhs,
								  valType **d_out,
								  const idType outColumn)
{
	const idType targetTensor = blockIdx.x;
	__shared__ valType tmp_out[sh_size];
	
	const idType r = d_outRows[targetTensor];
	const idType i = threadIdx.x;
	for (idType j = i; j < r * outColumn; j += blockDim.x) {
		tmp_out[j] = 0;
	}
	__syncthreads();

	for (idType targetNnz = i / subTB; targetNnz < d_nnz[targetTensor]; targetNnz += (blockDim.x / subTB)) {
		idType row_id = d_sp_ids[targetTensor][(targetNnz << 1) + adjoint_a] * outColumn;
		idType column_id = d_sp_ids[targetTensor][(targetNnz << 1) + (1 - adjoint_a)] * outColumn;
		valType val = d_sp_values[targetTensor][targetNnz];
		for (idType localN = i & (subTB - 1); localN < outColumn; localN += subTB) {
			atomicAdd(tmp_out + row_id + localN, val * d_rhs[targetTensor][column_id + localN]);
		}
	}
	__syncthreads();
	
	for (idType j = i; j < r * outColumn; j += blockDim.x) {
		d_out[targetTensor][j] = tmp_out[j];
	}
}

/*
  CUDA kernel: Compute sparse tensor x dense matrix of batched tensors (matrices)
  One CUDA kernel for all SpMMs
  One thread block for one partition
  One subTB (1, 2, ..., 32 threads) for one non-zero
*/
template <typename idType, typename valType, int max_size, int subTB, int adjoint_a>
__global__ void BatchedSpMM_partition(const idType* __restrict__ d_outRows,
									  const idType* __restrict__ d_nnz,
									  const idType partition, 
									  const idType nPartition, const idType partition_bit,
									  idType **d_sp_ids, valType **d_sp_values,
									  valType **d_rhs,
									  valType **d_out,
									  const idType nvector)
{
	__shared__ valType tmp_out[max_size];
	
	const idType targetTensor = blockIdx.x / nPartition;
	const idType targetPartition = blockIdx.x % nPartition;

	const idType offset = targetPartition << partition_bit;
	const idType nrow = d_outRows[targetTensor];
	const idType nnz = d_nnz[targetTensor];
	const idType p = (targetPartition == nPartition - 1)? (nvector - partition * (nPartition - 1)) : partition;
	
	const idType i = threadIdx.x;
	for (idType j = i; j < (nrow << partition_bit); j += blockDim.x) {
		tmp_out[j] = 0;
	}
	__syncthreads();
	
	for (idType targetNnz = i / subTB; targetNnz < nnz; targetNnz += (blockDim.x / subTB)) {
		idType row_id = d_sp_ids[targetTensor][(targetNnz << 1) + adjoint_a] << partition_bit;
		idType column_id = d_sp_ids[targetTensor][(targetNnz << 1) + (1 - adjoint_a)] * nvector + offset;
		valType val = d_sp_values[targetTensor][targetNnz];

		for (idType localN = i & (subTB - 1); localN < p; localN += subTB) {
			atomicAdd(tmp_out + row_id + localN, val * d_rhs[targetTensor][column_id + localN]);
		}
	}
	__syncthreads();
	
	for (idType j = i / subTB; j < nrow; j += (blockDim.x / subTB)) {
		for (idType k = i & (subTB -1); k < p; k += subTB) {
			d_out[targetTensor][j * nvector + offset + k] = tmp_out[(j << partition_bit) + k];
		}
	}
}

template <typename idType, typename valType, int adjoint_a>
void spmms_batched_coo_static(const idType *outRows, const idType *nnz,
							  const idType *d_outRows, const idType *d_nnz,
							  idType **d_sp_ids, valType **d_sp_values,
							  valType **d_d_x, valType **d_d_y,
							  const idType nvector, const idType batch)
{
	const idType max_size = SH_PER_TB / sizeof(valType);
	
	idType max_nrow = 0;
	for (idType i = 0; i < batch; ++i) {
		if (max_nrow < outRows[i]) {
			max_nrow = outRows[i];
		}
	}
	const int bs = 1024;
	// Any output matrix in batch can be placed on shared memory
	if (max_nrow * nvector <= max_size) {
		// Launch a kernel with appropriate subTB for nvector
		if (nvector > 16) {
			BatchedSpMM_small<idType, valType, max_size, 32, adjoint_a><<<batch, bs>>>(d_outRows, d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 8) {
			BatchedSpMM_small<idType, valType, max_size, 16, adjoint_a><<<batch, bs>>>(d_outRows, d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 4) {
			BatchedSpMM_small<idType, valType, max_size, 8, adjoint_a><<<batch, bs>>>(d_outRows, d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 2) {
			BatchedSpMM_small<idType, valType, max_size, 4, adjoint_a><<<batch, bs>>>(d_outRows, d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 1) {
			BatchedSpMM_small<idType, valType, max_size, 2, adjoint_a><<<batch, bs>>>(d_outRows, d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector == 1) {
			BatchedSpMM_small<idType, valType, max_size, 1, adjoint_a><<<batch, bs>>>(d_outRows, d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else {
			return;
		}
	}
	else if (max_nrow > max_size) {
		/* Initialization */
		idType max_thread = 0;
		for (idType i = 0; i < batch; ++i) {
			if (max_thread < outRows[i]) {
				max_thread = outRows[i];
			}
		}
		max_thread *= nvector;

		idType gs = (max_thread + bs - 1) / bs;
		BatchedInitOutputs<<<dim3(gs, batch), bs>>>(d_d_y, d_outRows, nvector);
	
		/* Batched Spmm Kernel */
		max_thread = 0;
		for (idType i = 0; i < batch; ++i) {
			if (max_thread < nnz[i]) {
				max_thread = nnz[i];
			}
		}
		if (nvector > 16) {
			max_thread *= 32;
			gs = (max_thread + bs - 1) / bs;
			BatchedSpMM<idType, valType, 32, adjoint_a><<<dim3(gs, batch), bs>>>
				(d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 8) {
			max_thread *= 16;
			gs = (max_thread + bs - 1) / bs;
			BatchedSpMM<idType, valType, 16, adjoint_a><<<dim3(gs, batch), bs>>>
				(d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 4) {
			max_thread *= 8;
			gs = (max_thread + bs - 1) / bs;
			BatchedSpMM<idType, valType, 8, adjoint_a><<<dim3(gs, batch), bs>>>
				(d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 2) {
			max_thread *= 4;
			gs = (max_thread + bs - 1) / bs;
			BatchedSpMM<idType, valType, 4, adjoint_a><<<dim3(gs, batch), bs>>>
				(d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 1) {
			max_thread *= 2;
			gs = (max_thread + bs - 1) / bs;
			BatchedSpMM<idType, valType, 2, adjoint_a><<<dim3(gs, batch), bs>>>
				(d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector == 1) {
			gs = (max_thread + bs - 1) / bs;
			BatchedSpMM<idType, valType, 1, adjoint_a><<<dim3(gs, batch), bs>>>
				(d_nnz, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else {
			return;
		}
	}
	else {
		idType p_bit = 0;
		while (((max_nrow << (p_bit + 1)) <= max_size) && ((1 << (p_bit + 1)) <= nvector)) {
			p_bit++;
		}
		idType p = 1 << p_bit;
		idType nPartition = (nvector + p - 1) / p;

		// One thread block for one partition
		if (nvector > 16) {
			BatchedSpMM_partition<idType, valType, max_size, 32, adjoint_a><<<nPartition * batch, bs>>>(d_outRows, d_nnz, p, nPartition, p_bit, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 8) {
			BatchedSpMM_partition<idType, valType, max_size, 16, adjoint_a><<<nPartition * batch, bs>>>(d_outRows, d_nnz, p, nPartition, p_bit, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 4) {
			BatchedSpMM_partition<idType, valType, max_size, 8, adjoint_a><<<nPartition * batch, bs>>>(d_outRows, d_nnz, p, nPartition, p_bit, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 2) {
			BatchedSpMM_partition<idType, valType, max_size, 4, adjoint_a><<<nPartition * batch, bs>>>(d_outRows, d_nnz, p, nPartition, p_bit, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector > 1) {
			BatchedSpMM_partition<idType, valType, max_size, 2, adjoint_a><<<nPartition * batch, bs>>>(d_outRows, d_nnz, p, nPartition, p_bit, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else if (nvector == 1) {
			BatchedSpMM_partition<idType, valType, max_size, 1, adjoint_a><<<nPartition * batch, bs>>>(d_outRows, d_nnz, p, nPartition, p_bit, d_sp_ids, d_sp_values, d_d_x, d_d_y, nvector);
		}
		else {
			return;
		}
	}
}

template <typename Device, typename idType, typename valType, bool ADJ_A, bool ADJ_B>
struct BspmmFunctor
{
    void operator()(OpKernelContext* context, Device &d, idType numTensor, idType *outRows, idType *outColumns, idType *nnz, idType **sp_ids, valType **sp_values, valType **rhs, valType **out);
};

/* Functor for CPU */
template <typename idType, typename valType, bool ADJ_A, bool ADJ_B>
struct BspmmFunctor<CPUDevice, idType, valType, ADJ_A, ADJ_B>
{
	void operator()(OpKernelContext* context, const CPUDevice &d, idType numTensor, idType *outRows, idType *outColumns, idType *nnz, idType **sp_ids, valType **sp_values, valType **rhs, valType **out)
    {
        /* Initialization */
        for (idType t = 0; t < numTensor; ++t) {
            for (idType i = 0; i < outRows[t] * outColumns[t]; ++i) {
                out[t][i] = 0;
            }
        }
		/* SpMM */
		const int sp_inc = (ADJ_A)? 1 : 0;		
        for (idType t = 0; t < numTensor; ++t) {
            for (idType i = 0; i < nnz[t]; ++i) {
                idType row_id = sp_ids[t][2 * i + sp_inc];
                idType column_id = sp_ids[t][2 * i + (1 - sp_inc)];
                valType val = sp_values[t][i];
                for (idType j = 0; j < outColumns[t]; ++j) {
					idType r = (ADJ_B)? j * outRows[t] + column_id : column_id * outColumns[t] + j;
                    out[t][row_id * outColumns[t] + j] += val * rhs[t][r];
                }
            }
        }
    }
};


/* Functor for GPU */
template <typename idType, typename valType, bool ADJ_A, bool ADJ_B>
struct BspmmFunctor<GPUDevice, idType, valType, ADJ_A, ADJ_B>
{
	void operator()(OpKernelContext* context, const GPUDevice &d, idType numTensor, idType *outRows, idType *outColumns, idType *nnz, idType **sp_ids, valType **sp_values, valType **rhs, valType **out)
    {
		/* Memory allocation and copyHtD */
		const TensorShape s({numTensor});
		Tensor d_outRows_t, d_outColumns_t, d_nnz_t, d_sp_ids_t, d_sp_values_t, d_rhs_t, d_out_t;
		context->allocate_temp(DT_INT64, s, &d_outRows_t);
		context->allocate_temp(DT_INT64, s, &d_outColumns_t);
		context->allocate_temp(DT_INT64, s, &d_nnz_t);
		context->allocate_temp(DT_INT64, s, &d_sp_ids_t);
		context->allocate_temp(DT_INT64, s, &d_sp_values_t);
		context->allocate_temp(DT_INT64, s, &d_rhs_t);
		context->allocate_temp(DT_INT64, s, &d_out_t);

        idType *d_outRows, *d_outColumns, *d_nnz;
		idType **d_sp_ids;
		valType **d_sp_values, **d_rhs, **d_out;

		d_outRows = (idType *)(d_outRows_t.vec<int64>().data());
		d_outColumns = (idType *)(d_outColumns_t.vec<int64>().data());
		d_nnz = (idType *)(d_nnz_t.vec<int64>().data());

		d_sp_ids = (idType **)(d_sp_ids_t.vec<int64>().data());
		d_sp_values = (valType **)(d_sp_values_t.vec<int64>().data());
		d_rhs = (valType **)(d_rhs_t.vec<int64>().data());
		d_out = (valType **)(d_out_t.vec<int64>().data());

		cudaMemcpyAsync(d_outRows, outRows, sizeof(idType) * numTensor, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_outColumns, outColumns, sizeof(idType) * numTensor, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_nnz, nnz, sizeof(idType) * numTensor, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_sp_ids, sp_ids, sizeof(idType*) * numTensor, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_sp_values, sp_values, sizeof(valType*) * numTensor, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_rhs, rhs, sizeof(valType*) * numTensor, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_out, out, sizeof(valType*) * numTensor, cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		if (ADJ_A == false) {
			spmms_batched_coo_static<idType, valType, 0>(outRows, nnz,
														 d_outRows, d_nnz,
														 d_sp_ids, d_sp_values,
														 d_rhs, d_out,
														 outColumns[0], numTensor);
		}
		else {
			spmms_batched_coo_static<idType, valType, 1>(outRows, nnz,
														 d_outRows, d_nnz,
														 d_sp_ids, d_sp_values,
														 d_rhs, d_out,
														 outColumns[0], numTensor);
		}

    }
};

template <typename Device, typename idType, typename valType>
class BspmmOp : public OpKernel
{
public:
    explicit BspmmOp(OpKernelConstruction* context) : OpKernel(context) {
		// Grab the attributes
		OP_REQUIRES_OK(context, context->GetAttr("adjoint_a", &adjoint_a));
		OP_REQUIRES_OK(context, context->GetAttr("adjoint_b", &adjoint_b));
	}
    
    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        OpInputList sp_shape_list(context, 0, 0);
        OpInputList sp_ids_list(context, 0, 0);
        OpInputList sp_values_list(context, 0, 0);
        OpInputList rhs_list(context, 0, 0);

		OP_REQUIRES_OK(context, context->input_list(StringPiece("sp_shape"), &sp_shape_list));
        OP_REQUIRES_OK(context, context->input_list(StringPiece("sp_ids"), &sp_ids_list));
        OP_REQUIRES_OK(context, context->input_list(StringPiece("sp_values"), &sp_values_list));
		OP_REQUIRES_OK(context, context->input_list(StringPiece("rhs"), &rhs_list));

		// int numTensor = n_tensor;
        int numTensor = rhs_list.size();

        OpOutputList olist(context, 0, numTensor);

        Tensor** output_tensor = new Tensor*[numTensor];
		idType *outColumns = new idType[numTensor];
		idType *outRows = new idType[numTensor];
        idType *nnz = new idType[numTensor];
        idType **sp_ids = new idType*[numTensor];
        valType **sp_values = new valType*[numTensor];
        valType **rhs = new valType*[numTensor];
        valType **output = new valType*[numTensor];

        for (int i = 0; i < numTensor; ++i) {
            auto sp_shape_t = (sp_shape_list[i]).vec<idType>();
            auto rhs_shape_t = (rhs_list[i]).shape();
			outRows[i] = (adjoint_a)? sp_shape_t(1) : sp_shape_t(0);
			outColumns[i] = (adjoint_b)? rhs_shape_t.dim_size(0) : rhs_shape_t.dim_size(1);
            nnz[i] = (sp_values_list[i]).shape().dim_size(0);
            /* Create an output tensor */
            TensorShape output_shape({outRows[i], outColumns[i]});
            olist.allocate(i, output_shape, output_tensor + i);
        
            sp_ids[i] = (idType *)((sp_ids_list[i]).matrix<idType>().data());
            sp_values[i] = (valType *)((sp_values_list[i]).vec<valType>().data());
            rhs[i] = (valType *)((rhs_list[i]).matrix<valType>().data());
            output[i] = (valType *)((output_tensor[i])->matrix<valType>().data());
			
        }
        // Execute Batched SpMM
#define GENERATE_ADJOOINT_PAIR(ADJ_A, ADJ_B) \
		if (adjoint_a == ADJ_A && adjoint_b == ADJ_B) { \
		 	BspmmFunctor<Device, idType, valType, ADJ_A, ADJ_B>()(context, context->eigen_device<Device>(), numTensor, outRows, outColumns, nnz, sp_ids, sp_values, rhs, output); \
		}

		GENERATE_ADJOOINT_PAIR(false, false);
		GENERATE_ADJOOINT_PAIR(true, false);
		GENERATE_ADJOOINT_PAIR(false, true);
		GENERATE_ADJOOINT_PAIR(true, true);
        
		delete[] outRows;
		delete[] outColumns;
		delete[] nnz;
        delete[] sp_ids;
        delete[] sp_values;
        delete[] rhs;
        delete[] output;
		delete[] output_tensor;
    }

private:
	bool adjoint_a;
	bool adjoint_b;
};

template <typename Device, typename idType, typename valType>
class BspmdtOp : public OpKernel
{
public:
    explicit BspmdtOp(OpKernelConstruction* context) : OpKernel(context) {
		// Grab the attributes
		OP_REQUIRES_OK(context, context->GetAttr("adjoint_a", &adjoint_a));
		OP_REQUIRES_OK(context, context->GetAttr("adjoint_b", &adjoint_b));
	}
    
    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        OpInputList sp_shape_list(context, 0, 0);
        OpInputList sp_ids_list(context, 0, 0);
        OpInputList sp_values_list(context, 0, 0);
		const Tensor *rhs_t;

		OP_REQUIRES_OK(context, context->input_list(StringPiece("sp_shape"), &sp_shape_list));
        OP_REQUIRES_OK(context, context->input_list(StringPiece("sp_ids"), &sp_ids_list));
        OP_REQUIRES_OK(context, context->input_list(StringPiece("sp_values"), &sp_values_list));
		OP_REQUIRES_OK(context, context->input(StringPiece("rhs"), &rhs_t));

		// int numTensor = n_tensor;
        int numTensor = sp_shape_list.size();

        OpOutputList olist(context, 0, numTensor);

        Tensor** output_tensor = new Tensor*[numTensor];
		idType *outColumns = new idType[numTensor];
		idType *outRows = new idType[numTensor];
        idType *nnz = new idType[numTensor];
        idType **sp_ids = new idType*[numTensor];
        valType **sp_values = new valType*[numTensor];
        valType **rhs = new valType*[numTensor];
        valType **output = new valType*[numTensor];

        for (int i = 0; i < numTensor; ++i) {
            auto sp_shape_t = (sp_shape_list[i]).vec<idType>();
            auto rhs_shape_t = rhs_t->shape();
			outRows[i] = (adjoint_a)? sp_shape_t(1) : sp_shape_t(0);
			outColumns[i] = (adjoint_b)? rhs_shape_t.dim_size(0) : rhs_shape_t.dim_size(1);
            nnz[i] = (sp_values_list[i]).shape().dim_size(0);
            /* Create an output tensor */
            TensorShape output_shape({outRows[i], outColumns[i]});
            olist.allocate(i, output_shape, output_tensor + i);
        
            sp_ids[i] = (idType *)((sp_ids_list[i]).matrix<idType>().data());
            sp_values[i] = (valType *)((sp_values_list[i]).vec<valType>().data());
			idType rhsRow = (adjoint_b)? rhs_shape_t.dim_size(1) : rhs_shape_t.dim_size(0);
			rhs[i] = (valType *)(rhs_t->matrix<valType>().data() + i * (rhsRow / numTensor) * outColumns[i]);
            output[i] = (valType *)((output_tensor[i])->matrix<valType>().data());
			
        }
        // Execute Batched SpMM
#define GENERATE_ADJOOINT_PAIR(ADJ_A, ADJ_B) \
		if (adjoint_a == ADJ_A && adjoint_b == ADJ_B) { \
		 	BspmmFunctor<Device, idType, valType, ADJ_A, ADJ_B>()(context, context->eigen_device<Device>(), numTensor, outRows, outColumns, nnz, sp_ids, sp_values, rhs, output); \
		}

		GENERATE_ADJOOINT_PAIR(false, false);
		GENERATE_ADJOOINT_PAIR(true, false);
		GENERATE_ADJOOINT_PAIR(false, true);
		GENERATE_ADJOOINT_PAIR(true, true);
        
		delete[] outRows;
		delete[] outColumns;
		delete[] nnz;
        delete[] sp_ids;
        delete[] sp_values;
        delete[] rhs;
        delete[] output;
		delete[] output_tensor;
    }

private:
	bool adjoint_a;
	bool adjoint_b;
};

#define REGISTER_BSPMM_CPU(idType, valType) \
	REGISTER_KERNEL_BUILDER(Name("Bspmm").Device(DEVICE_CPU) \
							.TypeConstraint<idType>("TI") \
							.TypeConstraint<valType>("TV") \
							.HostMemory("sp_shape"), \
							BspmmOp<CPUDevice, idType, valType>);

#define REGISTER_BSPMM_GPU(idType, valType) \
	REGISTER_KERNEL_BUILDER(Name("Bspmm").Device(DEVICE_GPU) \
							.TypeConstraint<idType>("TI") \
							.TypeConstraint<valType>("TV") \
							.HostMemory("sp_shape"),		\
							BspmmOp<GPUDevice, idType, valType>);

REGISTER_BSPMM_CPU(int32, float)
// REGISTER_BSPMM_CPU(int32, double)
REGISTER_BSPMM_CPU(int64, float)
// REGISTER_BSPMM_CPU(int64, double)
REGISTER_BSPMM_GPU(int32, float)
// REGISTER_BSPMM_GPU(int32, double)
REGISTER_BSPMM_GPU(int64, float)
// REGISTER_BSPMM_GPU(int64, double)

#define REGISTER_BSPMDT_CPU(idType, valType) \
	REGISTER_KERNEL_BUILDER(Name("Bspmdt").Device(DEVICE_CPU) \
							.TypeConstraint<idType>("TI") \
							.TypeConstraint<valType>("TV") \
							.HostMemory("sp_shape"), \
							BspmdtOp<CPUDevice, idType, valType>);

#define REGISTER_BSPMDT_GPU(idType, valType) \
	REGISTER_KERNEL_BUILDER(Name("Bspmdt").Device(DEVICE_GPU) \
							.TypeConstraint<idType>("TI") \
							.TypeConstraint<valType>("TV") \
							.HostMemory("sp_shape"),		\
							BspmdtOp<GPUDevice, idType, valType>);

REGISTER_BSPMDT_CPU(int32, float)
// REGISTER_BSPMDT_CPU(int32, double)
REGISTER_BSPMDT_CPU(int64, float)
// REGISTER_BSPMDT_CPU(int64, double)
REGISTER_BSPMDT_GPU(int32, float)
// REGISTER_BSPMDT_GPU(int32, double)
REGISTER_BSPMDT_GPU(int64, float)
// REGISTER_BSPMDT_GPU(int64, double)
