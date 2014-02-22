/*
   Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
#define TILE_WIDTH 2

namespace cuda
{
	__device__ float* get_sub_matrix(float *sq_matrix, int sq_dimension, int row, int col){
		float *sub_matrix;
		sub_matrix = &sq_matrix[sq_dimension*row*BLOCK_SIZE + BLOCK_SIZE*col];
		return sub_matrix;
	}
	void padding_reverse(float *sq_matrix, float *new_matrix, unsigned int sq_dimension, unsigned int new_dimension) {
		for(int i=0; i<sq_dimension; i++) {
			for(int j=0; j<sq_dimension; j++) {
				sq_matrix[i*sq_dimension + j] = new_matrix[i*new_dimension + j];
			}
		}
	}
	void padding(float *sq_matrix, float *new_matrix, unsigned int sq_dimension, unsigned int new_dimension) {
		for(int i=0; i<new_dimension; i++) {
			for(int j=0; j<new_dimension; j++) {
				if(i<sq_dimension && j<sq_dimension) {
					new_matrix[i*new_dimension + j] = sq_matrix[i*sq_dimension + j];
				}
				else {
					new_matrix[i*new_dimension + j] = 0;
				}
			}
		}
	}
	__global__ 
		void 
		small_matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
		{
			int row = blockIdx.y*blockDim.y+threadIdx.y;
			int col = blockIdx.x*blockDim.x+threadIdx.x;

			float sum = 0.0f;

			for(int k = 0; k < sq_dimension; k++)
			{
				sum += sq_matrix_1[row*sq_dimension + k] * sq_matrix_2[k*sq_dimension + col];
			}
			sq_matrix_result[row*sq_dimension + col] = sum;

		}
	__global__ 
		void 
		matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
		{

			int b_col = blockIdx.x;
			int b_row = blockIdx.y;
			int t_col = threadIdx.x;
			int t_row = threadIdx.y;

			float *sub_matrix = get_sub_matrix(sq_matrix_result, sq_dimension, b_row, b_col);

			float sum = 0.0f;

			for(int i = 0; i < sq_dimension/BLOCK_SIZE; i++)
			{
				float *sub_1 = get_sub_matrix(sq_matrix_1, sq_dimension, b_row, i);
				float *sub_2 = get_sub_matrix(sq_matrix_2, sq_dimension, i, b_col);
				__shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
				__shared__ float B[BLOCK_SIZE][BLOCK_SIZE];

				A[t_row][t_col] = sub_1[t_row*sq_dimension+t_col];
				B[t_row][t_col] = sub_2[t_row*sq_dimension+t_col]; 

				__syncthreads();

				for (int j = 0; j < BLOCK_SIZE ; ++j){
					sum += A[t_row][j] * B[j][t_col]; 
				}

				__syncthreads();


			}
			sub_matrix[t_row*sq_dimension + t_col] = sum;

		}

	void 
		matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension)
		{

			/* padded dimension*/
			unsigned int new_dimension = (sq_dimension%BLOCK_SIZE == 0? sq_dimension:(sq_dimension/BLOCK_SIZE+1)*BLOCK_SIZE);
			unsigned int size = new_dimension * new_dimension * sizeof(float);
			float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;
			/***************************************************
			  0st Part: padding, if necessary
			 ****************************************************/
			float *padded_matrix_1, *padded_matrix_2, *padded_matrix_result;
			if(sq_dimension ^ new_dimension){
				padded_matrix_1 = (float *)malloc(size);
				padded_matrix_2 = (float *)malloc(size);
				padded_matrix_result = (float *)malloc(size);

				padding(sq_matrix_1, padded_matrix_1, sq_dimension, new_dimension);
				padding(sq_matrix_2, padded_matrix_2,  sq_dimension, new_dimension);
			}else{
				padded_matrix_1 = sq_matrix_1;
				padded_matrix_2 = sq_matrix_2;
				padded_matrix_result = sq_matrix_result;
			}
			/***************************************************
			  1st Part: Allocation of memory on device memory  
			 ****************************************************/

			/* copy sq_matrix_1 and sq_matrix_2 to device memory */
			cudaMalloc((void**) &sq_matrix_1_d, size);
			cudaMemcpy(sq_matrix_1_d, padded_matrix_1, size, cudaMemcpyHostToDevice);
			cudaMalloc((void**) &sq_matrix_2_d, size);
			cudaMemcpy(sq_matrix_2_d, padded_matrix_2, size, cudaMemcpyHostToDevice);

			/*allocate sq_matrix_result on host */
			cudaMalloc((void**) &sq_matrix_result_d, size);

			/***************************************************
			  2nd Part: Inovke kernel 
			 ****************************************************/
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
			dim3 dimGrid(new_dimension / dimBlock.x, new_dimension / dimBlock.y);
			if(sq_dimension > BLOCK_SIZE){
				matrix_mul_kernel<<<dimGrid, dimBlock, dimBlock.x * dimBlock.x * sizeof(float)>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, new_dimension);
			}
			else{
				small_matrix_mul_kernel<<<dimGrid, dimBlock, dimBlock.x * dimBlock.x * sizeof(float)>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, new_dimension);
			}
			/***************************************************
			  3rd Part: Transfer result from device to host 
			 ****************************************************/
			cudaMemcpy(padded_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);

			/***************************************************
			  4th Part: free the padded matrices, if padding is necessary
			 ****************************************************/
			if(sq_dimension ^ new_dimension){
				padding_reverse(sq_matrix_result, padded_matrix_result, sq_dimension, new_dimension);
				free(padded_matrix_1);
				free(padded_matrix_2);
				free(padded_matrix_result);
			}
			cudaFree(sq_matrix_1_d);
			cudaFree(sq_matrix_2_d);
			cudaFree(sq_matrix_result_d);
		}  
} // namespace cuda
