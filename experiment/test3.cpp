#include <omp.h>
#include <iostream>
#include <time.h>
using namespace std;
int main()
{
  clock_t start, end;
  start = clock();
  omp_set_num_threads(5); 
  #pragma omp parallel 
  {
    cout << "Hello World!\n";
  }
  end = clock();
  cout << end << " " << start << endl;
}