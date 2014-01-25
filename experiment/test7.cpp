#include <stdio.h>
#include <omp.h>

int main()
{
  int idx = 100;
  
  //#pragma omp parallel private(idx)
  #pragma omp parallel firstprivate(idx)
  {
    printf("In thread %d idx = %d\n", omp_get_thread_num(), idx);
  }
}