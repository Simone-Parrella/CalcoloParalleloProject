/*
  PARALLELIZZAZIONE DELL'ALGORITMO DI GAUSS-SEIDEL
*/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define MAX_ITER 100   // numero massimo di iterazioni
#define MAX 100        // valore massimo degli elementi della matrice
#define TOL 0.000001   // tolleranza

int np, myrank, communication, cnt_iter = 0;  // numero di processi e rank
float *vector;

double tstartdiffdone, tfinishdiffdone, TotalTimediffdone;
double tstartrow, tfinishrow, TotalTimerow;

void solver(int n, int num_elems) {
    float diff = 0, temp, difftemp = 0;
    int done = 0, cnt_iter = 0, myrank;
    int factor = num_elems - (2 * n);

    while (!done && (cnt_iter < MAX_ITER)) {
        diff = 0;

        // Neither the first row nor the last row are solved
        // (that's why it starts at "n" and it goes up to "num_elems - 2n")
        for (int i = n; i < factor; i++) {

            // Additionally, neither the first nor last column are solved
            // (that's why the first and last positions of "rows" are skipped)
            if ((i % n == 0) || ((i + 1) % n == 0)) {
                continue;
            }

            int pos_up = i - n;
            int pos_do = i + n;
            int pos_le = i - 1;
            int pos_ri = i + 1;

            temp = vector[i];
            vector[i] = 0.2 * (vector[i] + vector[pos_le] + vector[pos_up] + vector[pos_ri] + vector[pos_do]);
            diff += fabs(vector[i] - temp);
        }

        if (diff * 1 / n * 1 / n < TOL) {
            done = 1;
        }
        cnt_iter++;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Communicate diff and done
    float tstartdiffdone, tfinishdiffdone;
    tstartdiffdone = MPI_Wtime();

    switch (communication) {
        case 0: { // Point-to-Point communication
            if (myrank != 0) {
                MPI_Send(&vector[n], factor - n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                MPI_Recv(&vector[num_elems - n], factor - n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (myrank == 0) {
                int l;
                for (l = 1; l < np; l++) {
                    MPI_Recv(&vector[(l * factor) / np], factor / np, MPI_FLOAT, l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    diff += compute_diff(vector, (l * factor) / np, ((l + 1) * factor) / np, n);
                }

                // Check if we are done
                if (diff / n / n < TOL) done = 1;

                // Send done to all
                for (l = 1; l < np; l++)
                    MPI_Send(&done, 1, MPI_INT, l, 1, MPI_COMM_WORLD);
            }
            break;
        }

        case 1: { // Collective communication
            MPI_Reduce(&diff, &difftemp, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

            // Check if we are done
            if (difftemp / n / n < TOL) done = 1;

            MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        }
    }

    tfinishdiffdone = MPI_Wtime();
    TotalTimediffdone += tfinishdiffdone - tstartdiffdone;

    if (done) {
        printf("Node %d: Solver converged after %d iterations\n", myrank, cnt_iter);
    } else {
        printf("Node %d: Solver not converged after %d iterations\n", myrank, cnt_iter);
    }
}


// Funzione principale
int main(int argc, char *argv[]) {
  int n, tam;
  double tstart, tfinish, TotalTime;
  double tstartscatter, tfinishscatter, TotalTimescatter;
  double tstartgather, tfinishgather, TotalTimegather;
  double tstartsolver, tfinishsolver, TotalTimesolver;
  double tfinishMPI, TotalTimeMPI;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // ID del processo
  MPI_Comm_size(MPI_COMM_WORLD, &np);     // Numero di processi per il comunicatore COMM_WORLD

  tstart = MPI_Wtime();

  // Verifica se tutto è a posto
  if (argc < 3 || np < 2) {
    if (myrank == 0) {
      printf("\nRicorda gli argomenti o crea più di UN processo per favore!\n\n");
    }

    MPI_Finalize();
    exit(1);
  }

  // Ricevi gli argomenti del programma
  n = atoi(argv[1]);              // numero di elementi nell'array
  communication =  atoi(argv[2]);  // tipo di comunicazione

  // Il processo 0 crea la matrice
  if (myrank == 0)
  {
    int i, j;
    vector = (float *)malloc(n * n * sizeof(float));
    for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        vector[i * n + j] = ((float)rand() / (float)(RAND_MAX)) * MAX;
    }
  }

  // Gli altri processi (tranne il 0) riservano la memoria
  tam = n / np + 2; // righe che avrà ogni processo
  if (myrank != 0 && myrank != np - 1)
  {
    vector = (float *)malloc(tam * n * sizeof(float));
  }

  if (myrank == np - 1)
  {
    vector = (float *)malloc((tam - 1) * n * sizeof(float));
  }

  tfinishMPI = MPI_Wtime();
  TotalTimeMPI = tfinishMPI - tstart;

  // Scatter della matrice
  tstartscatter = MPI_Wtime();
  switch (communication)
  {
  case 0:
  { // Comunicazione P2P asincrona per lo scatter
    MPI_Request requests[np];
    MPI_Status statuses[np];
    if (myrank == 0)
    {
      int e = tam - 2, proc = 1, i; // totale elementi da inviare per processo (conteggio delle righe) e numero di processo
      for (i = e; i < n; i += e)
      {
        if (proc != np - 1)
          MPI_Isend(&vector[i * n - n], n * tam, MPI_FLOAT, proc, 1, MPI_COMM_WORLD, &requests[proc]);
        else
          MPI_Isend(&vector[i * n - n], n * tam - n, MPI_FLOAT, proc, 1, MPI_COMM_WORLD, &requests[proc]);
        proc++;
      }
    }

    if (myrank != 0 && myrank != np - 1)
      MPI_Irecv(&vector[0], n * tam, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &requests[myrank]);

    if (myrank == np - 1)
      MPI_Irecv(&vector[0], n * tam - n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &requests[myrank]);

  }
    case 1: { // Comunicazione collettiva asincrona per lo scatter
      int sendcount[np], displ[np];
      int e = tam - 2, proc, i;
      sendcount[0] = 0;
      displ[0] = 0;

      for (i = e, proc = 1; i < n; i += e, proc++) {
        displ[proc] = (i - 1) * n;
      }

      for (i = 1; i < np - 1; i++)
        sendcount[i] = n * tam;
      sendcount[np - 1] = n * tam - n;

      MPI_Request request;
      MPI_Status status;
      MPI_Iscatterv(vector, sendcount, displ, MPI_FLOAT, vector, tam * n, MPI_FLOAT, 0, MPI_COMM_WORLD, &request);
      break;
    }
  }
  tfinishscatter = MPI_Wtime();
  TotalTimescatter = tfinishscatter - tstartscatter;

  // Risolve la matrice
  tstartsolver = MPI_Wtime();

  if (myrank == 0)
  {
    solver(tam - 1, n);
  }
  else if (myrank != 0 && myrank != np - 1)
  {
    solver(tam, n);
  }
  else if (myrank == np - 1)
  {
    solver(tam - 1, n);
  }

  tfinishsolver = MPI_Wtime();
  TotalTimesolver = tfinishsolver - tstartsolver;

  // Gather dei risultati
  tstartgather = MPI_Wtime();
  switch (communication) {
    case 0: { // Comunicazione P2P asincrona per il gather
      MPI_Request request;
      MPI_Status status;
      if(myrank != 0 && myrank != np-1)
        MPI_Isend(&vector[n], n*(tam-2), MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request); 
              
      if(myrank == np-1)
        MPI_Isend(&vector[n], n*(tam-3), MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request); 

      if(myrank == 0) {
        int e = tam - 2, proc = 1, i;
        for(i = e; i < n; i += e) {
          if(proc != np-1)
            MPI_Irecv(&vector[i*n], n*(tam-2), MPI_FLOAT, proc, 1, MPI_COMM_WORLD, &request);  
          else          
            MPI_Irecv(&vector[i*n], n*(tam-3), MPI_FLOAT, proc, 1, MPI_COMM_WORLD, &request);
          proc++;
        }
      }
      break;
    }

    case 1: { // Comunicazione collettiva asincrona per il gather
      int recvcount[np], displ[np];
      int e = tam - 2, proc, i;
      displ[0] = 0;

      for(i = e, proc = 1; i < n; i += e, proc++) {
        displ[proc] = i*n;          
      }

      for(i = 1; i < np; i++)
        recvcount[i] = n*(tam-2);
      recvcount[0] = 0;

      MPI_Request request;
      MPI_Status status;
      MPI_Igatherv(vector+n, n*(tam-2), MPI_FLOAT, vector, recvcount, displ, MPI_FLOAT, 0, MPI_COMM_WORLD, &request);

      break;
    }
  }
  tfinishgather = MPI_Wtime();
  TotalTimegather = tfinishgather - tstartgather;

  // Concludiamo
  tfinish = MPI_Wtime();
  TotalTime = tfinish - tstart;

  if(myrank == 0) {
    printf("\n\n * * * * * * * * * * * RECORD * * * * * * * * * * * * * \n");
    printf("\n - - - - -Dimensione matrice               : %d", n); 
    printf("\n - - - - -Comunicazione                   : %d", communication);
    printf("\n - - - - -# processi                      : %d", np);
    printf("\n - - - - -Tempo soluzione                 : %lf s", TotalTime);
    printf("\n - - - - -Tempo memoria e MPI             : %lf s", TotalTimeMPI);
    printf("\n - - - - -Tempo scatter matrice           : %lf s", TotalTimescatter);
    printf("\n - - - - -Tempo gather matrice            : %lf s", TotalTimegather);
    printf("\n - - - - -Tempo in solver                 : %lf s", TotalTimesolver);
    printf("\n - - - - -Tempo totale comunicazione diff e done : %lf s", TotalTimediffdone);
    printf("\n - - - - -Tempo totale comunicazione tra righe vicine : %lf s", TotalTimerow);
    printf("\n - - - - -Iterazioni                      : %i ", cnt_iter);
    printf("\n - - - - -Tempo medio per iterazione      : %lf s", TotalTimesolver/cnt_iter);
    printf("\n - - - - -Tempo medio comunicazione diff e done   : %lf s", TotalTimediffdone/cnt_iter);
    printf("\n - - - - -Tempo medio comunicazione tra righe vicine : %lf s", TotalTimerow/cnt_iter);
    printf("\n\n * * * * * * * * * *  FINE RECORD * * * * * * * * * * * * \n");
  }

  MPI_Finalize();
  return 0;
}
