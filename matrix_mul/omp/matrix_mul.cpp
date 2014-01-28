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
    
#pragma omp parallel for if(sq_dimension > 200)
	for (unsigned int i = 0; i < sq_dimension; i++) 
    {
		for(unsigned int j = 0; j < sq_dimension; j++) 
		{       
			sq_matrix_result[i*sq_dimension + j] = 0;
			for (unsigned int k = 0; k < (sq_dimension>>4)*16; k+=16) {
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
														sq_matrix_2_tmp[j*sq_dimension + k];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+1] * 
														sq_matrix_2_tmp[j*sq_dimension + k+1];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+2] * 
														sq_matrix_2_tmp[j*sq_dimension + k+2];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+3] * 
														sq_matrix_2_tmp[j*sq_dimension + k+3];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+4] * 
														sq_matrix_2_tmp[j*sq_dimension + k+4];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+5] * 
														sq_matrix_2_tmp[j*sq_dimension + k+5];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+6] * 
														sq_matrix_2_tmp[j*sq_dimension + k+6];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+7] * 
														sq_matrix_2_tmp[j*sq_dimension + k+7];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+8] * 
														sq_matrix_2_tmp[j*sq_dimension + k+8];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+9] * 
														sq_matrix_2_tmp[j*sq_dimension + k+9];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+10] * 
														sq_matrix_2_tmp[j*sq_dimension + k+10];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+11] * 
														sq_matrix_2_tmp[j*sq_dimension + k+11];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+12] * 
														sq_matrix_2_tmp[j*sq_dimension + k+12];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+13] * 
														sq_matrix_2_tmp[j*sq_dimension + k+13];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+14] * 
														sq_matrix_2_tmp[j*sq_dimension + k+14];
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k+15] * 
														sq_matrix_2_tmp[j*sq_dimension + k+15];

			}
			for (unsigned int k = (sq_dimension>>4)*16; k < sq_dimension; k++) {
				sq_matrix_result[i*sq_dimension + j] += sq_matrix_1[i*sq_dimension + k] * 
														sq_matrix_2_tmp[j*sq_dimension + k];
			}
		}
	}
}
  
} //namespace omp

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
