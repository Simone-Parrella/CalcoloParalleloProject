#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double* create_rand_nums(const int num_elements) {
  double *rand_nums = (double *)malloc(sizeof(double) * num_elements);
  assert(rand_nums != NULL);
  for (int i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

int main(int argc, char* argv){
  int num_clusters = 60;
  int tot_nums = 648000*2;
  double* random_nums = create_rand_nums(tot_nums);
  FILE* output=fopen("input.txt","w");
  fprintf(output, "%d\n", num_clusters);
  fprintf(output, "%d\n", tot_nums/2);
  for(int i=0; i<tot_nums; i=i+2){
    fprintf(output,"%lf,",random_nums[i]);
    fprintf(output,"%lf\n", random_nums[i+1]);
  }
}