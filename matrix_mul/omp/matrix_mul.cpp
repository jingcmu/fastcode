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
			for(unsigned int j = 0; j < sq_dimension; j++) 
			{       
				__m128 dest1 = (__m128)_mm_setzero_si128();
				__m128 dest2 = (__m128)_mm_setzero_si128();
				//__m128 dest3 = (__m128)_mm_setzero_si128();
				//__m128 dest4 = (__m128)_mm_setzero_si128();
				sq_matrix_result[i*sq_dimension + j] = 0;
				for (unsigned int k = 0; k < (sq_dimension>>3)*8; k+=8) {
					__m128* src1 = (__m128*)&sq_matrix_1[i*sq_dimension + k];
					__m128* src2 = (__m128*)&sq_matrix_2_tmp[j*sq_dimension + k];
					dest1 = _mm_add_ps(dest1, _mm_mul_ps(*src1, *src2));
					__m128* src3 = (__m128*)&sq_matrix_1[i*sq_dimension + k+4];
					__m128* src4 = (__m128*)&sq_matrix_2_tmp[j*sq_dimension + k+4];
					dest2 = _mm_add_ps(dest2, _mm_mul_ps(*src3, *src4));
					//__m128* src5 = (__m128*)&sq_matrix_1[i*sq_dimension + k+8];
					//__m128* src6 = (__m128*)&sq_matrix_2_tmp[j*sq_dimension + k+8];
					//dest3 = _mm_add_ps(dest3, _mm_mul_ps(*src5, *src6));
					//__m128* src7 = (__m128*)&sq_matrix_1[i*sq_dimension + k+12];
					//__m128* src8 = (__m128*)&sq_matrix_2_tmp[j*sq_dimension + k+12];
					//dest4 = _mm_add_ps(dest4, _mm_mul_ps(*src7, *src8));
				}
				float * dest_pf1 = (float *)&dest1, dest_f = 0.0f;
				float * dest_pf2 = (float *)&dest2;
				//float * dest_pf3 = (float *)&dest3;
				//float * dest_pf4 = (float *)&dest4;
				for(int p=0; p<4; p++) {
					 dest_f += (*dest_pf1++ + *dest_pf2++);
				}
				sq_matrix_result[i*sq_dimension + j] = dest_f;
				for (unsigned int k = (sq_dimension>>3)*8; k < sq_dimension; k++) {
					sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
															sq_matrix_2_tmp[j*sq_dimension + k];
				}
			}
		}
	else
		#pragma omp parallel for if(sq_dimension > 20)
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

/* version 4
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
		#pragma omp parallel for if(sq_dimension > 200)
		for (unsigned int i = 0; i < sq_dimension; i++) 
		{
			for(unsigned int j = 0; j < sq_dimension; j++) 
			{       
				__m128 dest = (__m128)_mm_setzero_si128();
				sq_matrix_result[i*sq_dimension + j] = 0;
				for (unsigned int k = 0; k < (sq_dimension>>2)*4; k+=4) {
					__m128* src1 = (__m128*)&sq_matrix_1[i*sq_dimension + k];
					__m128* src2 = (__m128*)&sq_matrix_2_tmp[j*sq_dimension + k];
					dest = _mm_add_ps(dest, _mm_mul_ps(*src1, *src2));
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
	else
		#pragma omp parallel for if(sq_dimension > 200)
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
*/

/* version 3
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
		#pragma omp parallel for if(sq_dimension > 200)
		for (unsigned int i = 0; i < sq_dimension; i++) 
		{
			for(unsigned int j = 0; j < sq_dimension; j++) 
			{       
				__m128 dest = (__m128)_mm_setzero_si128();
				sq_matrix_result[i*sq_dimension + j] = 0;
				for (unsigned int k = 0; k < (sq_dimension>>2)*4; k+=4) {
					__m128* src1 = (__m128*)&sq_matrix_1[i*sq_dimension + k];
					__m128* src2 = (__m128*)&sq_matrix_2_tmp[j*sq_dimension + k];
					dest = _mm_add_ps(dest, _mm_mul_ps(*src1, *src2));
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
	else
		#pragma omp parallel for if(sq_dimension > 200)
		for (unsigned int i = 0; i < sq_dimension; i++) 
		{
			for(unsigned int j = 0; j < sq_dimension; j++) 
			{       
				sq_matrix_result[i*sq_dimension + j] = 0;
				for (unsigned int k = 0; k < sq_dimension; k++)
					sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
															sq_matrix_2_tmp[j*sq_dimension + k];
			}
		}
}
  
} //namespace omp
*/

/* version 2
#include <omp.h>
#include <iostream>
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
    
#pragma omp parallel for
	for (unsigned int i = 0; i < sq_dimension; i++) 
    {
		for(unsigned int j = 0; j < sq_dimension; j++) 
		{       
			sq_matrix_result[i*sq_dimension + j] = 0;
			for (unsigned int k = 0; k < sq_dimension; k++)
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
														sq_matrix_2_tmp[j*sq_dimension + k];
		}
	}
}
  
} //namespace omp

*/

/* version 1
#include <omp.h>
#include <iostream>
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
    
	#pragma omp sections
	{
		#pragma omp section 
		{
			for (unsigned int i = 0; i < (sq_dimension>>2); i++) 
			{
				for(unsigned int j = 0; j < sq_dimension; j++) 
				{       
					sq_matrix_result[i*sq_dimension + j] = 0;
					for (unsigned int k = 0; k < sq_dimension; k++)
						sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
																sq_matrix_2_tmp[j*sq_dimension + k];
				}
			}	
		}
		#pragma omp section 
		{
			for (unsigned int i = (sq_dimension>>2); i < (sq_dimension>>1); i++) 
			{
				for(unsigned int j = 0; j < sq_dimension; j++) 
				{       
					sq_matrix_result[i*sq_dimension + j] = 0;
					for (unsigned int k = 0; k < sq_dimension; k++)
						sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
																sq_matrix_2_tmp[j*sq_dimension + k];
				}
			}
		}
		#pragma omp section 
		{
			for (unsigned int i = (sq_dimension>>1); i < (sq_dimension>>2)+(sq_dimension>>1); i++) 
			{
				for(unsigned int j = 0; j < sq_dimension; j++) 
				{       
					sq_matrix_result[i*sq_dimension + j] = 0;
					for (unsigned int k = 0; k < sq_dimension; k++)
						sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
																sq_matrix_2_tmp[j*sq_dimension + k];
				}
			}
		}
		#pragma omp section 
		{
			for (unsigned int i = (sq_dimension>>2)+(sq_dimension>>1); i < sq_dimension; i++) 
			{
				for(unsigned int j = 0; j < sq_dimension; j++) 
				{       
					sq_matrix_result[i*sq_dimension + j] = 0;
					for (unsigned int k = 0; k < sq_dimension; k++)
						sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
																sq_matrix_2_tmp[j*sq_dimension + k];
				}
			}
		}
	}
}
  
} //namespace omp
*/


/* ԭʼ�汾
#include <omp.h>
#include "matrix_mul.h"

namespace omp
{
  void
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
  {
#pragma omp parallel for
    for (unsigned int i = 0; i < sq_dimension; i++) 
      {
	for(unsigned int j = 0; j < sq_dimension; j++) 
	  {       
	    sq_matrix_result[i*sq_dimension + j] = 0;
	    for (unsigned int k = 0; k < sq_dimension; k++)
	      sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
	  }
      }// End of parallel region
  }
  
} //namespace omp
*/
