//partial sums of the harmonic series
//computed up and down to show differing precisions
//Summer Study group, 2015

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;

static void show_usage(std::string name)
{
  std::cerr << "Usage: " << name << " <N>" << "\n"
	    << "See comments, partial sum of harmonic series"
	    << "\n";
}

double difference(long N)
{
  double differ=0;
  double sumU=0;
  double sumD=0;
  for(int i=1; i<N+1; i++)
    {
      sumU+=1.0/i;
    }
  for(int i=N; i>0; i--)
    {
      sumD+=1.0/i;
    }
  differ=1.0/2.0*fabs((sumU-sumD))/(fabs(sumU)+fabs(sumD));
  return differ;
}

int main(int argc, char* argv[])
{
  clock_t tStart = clock();
  if (argc != 2)
    {
      show_usage(argv[0]);
      return 2;
    }
  double sumdown=0;
  double sumup=0;
  char *endptr;
  long int N = strtol(argv[1], &endptr, 10);
  if (!*argv[1] || *endptr)
    {
      cerr << "Invalid number " << argv[1] << '\n';
      return 2;
    }
  for(int i=1; i<N+1; i++)
    {
      sumup+=1.0/i;
    }
  for(int i=N; i>0; i--)
    {
      sumdown+=1.0/i;
    }
  cout << "sum down (" << N << ") = " << sumdown << "\n";
  cout << "sum up (" << N << ") = " << sumup << "\n";
  //Plot part
  //Gnuplot gp2;
  //std::vector<boost::tuple<double, double> > xy_pts_A1;
  for(double i=0; i<4; i++)
    {
      cout << "pow=" << i << " : " << difference(N*pow(10.0,i)) << "\n";
    }
  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  return 0;
}
	 
