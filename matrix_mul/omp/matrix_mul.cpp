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

#include <omp.h>
#include <iostream>
#include <emmintrin.h>
#include "matrix_mul.h"

namespace omp
{

	void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, 
			float *sq_matrix_result, unsigned int sq_dimension )
	{
		float *sq_matrix_2_tmp = new float[sq_dimension * sq_dimension];
#pragma omp parallel for 
		for(unsigned int i = 0; i < sq_dimension; i++) {
			for(unsigned int j = 0; j < sq_dimension; j++){
				sq_matrix_2_tmp[i*sq_dimension+j] = sq_matrix_2[j*sq_dimension+i];
			}
		}

		if(sq_dimension%4 == 0)
#pragma omp parallel for 
			for (unsigned int i = 0; i < sq_dimension; i++) 
			{
				int index_i = i*sq_dimension;			
				for(unsigned int j = 0; j < sq_dimension; j++) 
				{   
					int index_j = j*sq_dimension;
					__m128 dest1 = (__m128)_mm_setzero_si128();
					__m128 dest2 = (__m128)_mm_setzero_si128();
					__m128 dest3 = (__m128)_mm_setzero_si128();
					__m128 dest4 = (__m128)_mm_setzero_si128();
					sq_matrix_result[i*sq_dimension + j] = 0;
					_mm_prefetch(&sq_matrix_1[i*sq_dimension], _MM_HINT_NTA);
					_mm_prefetch(&sq_matrix_2_tmp[j*sq_dimension], _MM_HINT_NTA);
					for (unsigned int k = 0; k < (sq_dimension>>4)*16; k+=16) {
						__m128* src11 = (__m128*)&sq_matrix_1[index_i+k];
						__m128* src12 = (__m128*)&sq_matrix_2_tmp[index_j+k];
						dest1 = _mm_add_ps(dest1, _mm_mul_ps(*src11, *src12));
						__m128* src21 = (__m128*)&sq_matrix_1[index_i+k+4];
						__m128* src22 = (__m128*)&sq_matrix_2_tmp[index_j+k+4];
						dest2 = _mm_add_ps(dest2, _mm_mul_ps(*src21, *src22));
						__m128* src31 = (__m128*)&sq_matrix_1[index_i+k+8];
						__m128* src32 = (__m128*)&sq_matrix_2_tmp[index_j+k+8];
						dest3 = _mm_add_ps(dest3, _mm_mul_ps(*src31, *src32));
						__m128* src41 = (__m128*)&sq_matrix_1[index_i+k+12];
						__m128* src42 = (__m128*)&sq_matrix_2_tmp[index_j+k+12];
						dest4 = _mm_add_ps(dest4, _mm_mul_ps(*src41, *src42));
					}
					float *dest_f1 = (float *)&dest1, * dest_f2 = (float *)&dest2, * dest_f3 = (float *)&dest3, * dest_f4 = (float *)&dest4;
					for(int p=0; p<4; p++) {
						sq_matrix_result[i*sq_dimension + j] += (*dest_f1++ + *dest_f2++ + *dest_f3++ + *dest_f4++);
					}
					for (unsigned int k = (sq_dimension>>4)*16; k < sq_dimension; k++) {
						sq_matrix_result[index_i + j] += sq_matrix_1[index_i + k] * 
							sq_matrix_2_tmp[index_j + k];
					}
				}
			}
		else
#pragma omp parallel for 
			for (unsigned int i = 0; i < sq_dimension; i++) 
			{
				for(unsigned int j = 0; j < sq_dimension; j++) 
				{       
					__m128 dest = _mm_setzero_ps();
					sq_matrix_result[i*sq_dimension + j] = 0;
					for (unsigned int k = 0; k < (sq_dimension>>2)*4; k+=4) {
						__m128 src1 = _mm_loadu_ps(&sq_matrix_1[i*sq_dimension + k]);
						__m128 src2 = _mm_loadu_ps(&sq_matrix_2_tmp[j*sq_dimension + k]);
						dest = _mm_add_ps(dest, _mm_mul_ps(src1, src2));
					}
					float * dest_f = (float *)&dest;
					for(int p=0; p<4; p++) {
						sq_matrix_result[i*sq_dimension + j] += (*dest_f++);
					}
					for (unsigned int k = (sq_dimension>>2)*4; k < sq_dimension; k++) {
						sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
							sq_matrix_2_tmp[j*sq_dimension + k];
					}
				}
			}
	}

}  //namespace omp
