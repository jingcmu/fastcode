#include <iostream>
#include <time.h>
using namespace std;
int main()
{
  clock_t start, end;
  start = clock();
  #pragma omp parallel num_threads(5) 
  {
    cout << "Hello World!\n";
  }
  end = clock();
  cout << end << " " << start << endl;
}