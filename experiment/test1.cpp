#include <iostream>
#include <time.h>
using namespace std;
int main()
{
  clock_t start, end;
  start = clock();
  //#pragma omp parallel
  {
    cout << "Hello World!\n";
  }
  end = clock();
  cout << end << " " << start << " " << (double)(end - start)/CLOCKS_PER_SEC << endl;
}