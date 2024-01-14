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

// Genera numeri casuali in virgola mobile
float rand_float(int max) {
  return ((float)rand()/(float)(RAND_MAX)) * max;
}

// Inizializza un vettore con valori casuali
void allocate_init_vector(int n, int m) {
  int i, j;
  vector = (float *) malloc(n * m * sizeof(float));

  for(i = 0; i < n; i++) {
    for (j = 0; j < m; j++)
      vector[i*m+j] = rand_float(MAX);
  }
}

// Alloca memoria per un vettore
void allocate_mem(int n, int m) {
  vector = (float *) malloc(n * m * sizeof(float));
}

// Stampa un vettore
void printVector(int n, int m) {
  int i, j;
  for (i = 0; i < n; i++) {
    printf("\n");
    for (j = 0; j < m; j++)
      printf(" %.2f", vector[i*m+j]);
  }
  printf("\n");
}

// Risolve la matrice usando l'algoritmo di Gauss-Seidel
void solver(int n, int m) {
  float diff = 0, temp, difftemp = 0;
  int done = 0,  i, j;

  while (!done && (cnt_iter < MAX_ITER)) {
    diff = 0;

    // Aggiorna gli elementi della matrice
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < m - 1; j++) {
        temp = vector[i*m+j];
        vector[i*m+j] = 0.2 * (vector[i*m + j] + vector[i*m + j-1] + vector[(i-1)*m + j] + vector[i*m + j+1] + vector[(i+1)*m + j]);
        diff += abs(vector[i*m + j] - temp); // Calcola la differenza
      }
    }

    // Comunicazione tra le righe vicine
    tstartrow = MPI_Wtime();

    if(myrank % 2 == 0) { // I ranks pari inviano e poi ricevono
      if(myrank != np-1) {
        MPI_Send(&vector[(n-2)*m], m, MPI_FLOAT, myrank+1, 1, MPI_COMM_WORLD); // invia penultima
        MPI_Recv(&vector[(n-1)*m], m, MPI_FLOAT, myrank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // ricevi in ultima
      }

      if(myrank != 0) {
        MPI_Send(&vector[m], m, MPI_FLOAT, myrank-1, 1, MPI_COMM_WORLD); // invia seconda
        MPI_Recv(&vector[0], m, MPI_FLOAT, myrank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // ricevi in prima
      }
    }

    if(myrank % 2 == 1) { // I ranks dispari ricevono e poi inviano
      MPI_Recv(&vector[0], m, MPI_FLOAT, myrank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // ricevi in prima
      MPI_Send(&vector[m], m, MPI_FLOAT, myrank-1, 1, MPI_COMM_WORLD); // invia seconda

      if(myrank != np-1) {
        MPI_Recv(&vector[(n-1)*m], m, MPI_FLOAT, myrank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // ricevi in ultima
        MPI_Send(&vector[(n-2)*m], m, MPI_FLOAT, myrank+1, 1, MPI_COMM_WORLD); // invia penultima
      }
    }

    tfinishrow = MPI_Wtime();
    TotalTimerow += tfinishrow - tstartrow;

    // Comunicazione diff e done
    tstartdiffdone = MPI_Wtime();

    switch(communication) {
      case 0: { // Chiamate P2P
        if(myrank != 0) {
          MPI_Send(&diff, 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
          MPI_Recv(&done, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if(myrank == 0) {
          int l;
          for(l = 1; l < np; l++) {
            MPI_Recv(&difftemp, 1, MPI_FLOAT, l, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            diff += difftemp;
          }

          // Calcola se abbiamo terminato
          if (diff/n/n < TOL)  done = 1;

          // Invia done a tutti
          for(l = 1; l < np; l++) 
            MPI_Send(&done, 1, MPI_INT, l, 1, MPI_COMM_WORLD); 
        }
        break;
      }

      case 1: { // Chiamate collettive
        MPI_Reduce(&diff, &difftemp, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Calcola se abbiamo terminato
        if (difftemp/n/n < TOL)  done = 1;

        MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
        break;
      }
    }

    tfinishdiffdone = MPI_Wtime();
    TotalTimediffdone += tfinishdiffdone - tstartdiffdone;

    // Conta l'iterazione
    cnt_iter++;
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
  if(myrank == 0)
    allocate_init_vector(n, n);

  // Gli altri processi (tranne il 0) riservano la memoria
  tam = n/np + 2; // righe che avrà ogni processo
  if(myrank != 0 && myrank != np-1)
    allocate_mem(tam, n); // tutti gli altri riservano lo spazio per ricevere i dati

  if(myrank == np-1)
    allocate_mem(tam-1, n); // l'ultimo processo non ha bisogno di tante righe secondo l'analisi (ne vale tam-1)

  tfinishMPI = MPI_Wtime();
  TotalTimeMPI = tfinishMPI - tstart;

  // Scatter della matrice
  tstartscatter = MPI_Wtime();
  switch (communication) {
    case 0: { // Comunicazione P2P per lo scatter
      if(myrank == 0) {
        int e = tam - 2, proc = 1, i; // totale elementi da inviare per processo (conteggio delle righe) e numero di processo
        for(i = e; i < n; i += e) {
          if(proc != np-1)
            MPI_Send(&vector[i*n-n], n*tam, MPI_FLOAT, proc, 1, MPI_COMM_WORLD);  
          else          
            MPI_Send(&vector[i*n-n], n*tam-n, MPI_FLOAT, proc, 1, MPI_COMM_WORLD);
          proc++;
        }
      }
      
      if(myrank != 0 && myrank != np-1)
        MPI_Recv(&vector[0], n*tam, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
      if(myrank == np-1)
        MPI_Recv(&vector[0], n*tam-n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
      break;
    }

    case 1: { // Comunicazione collettiva per lo scatter
      int sendcount[np], displ[np];
      int e = tam - 2, proc, i;
      sendcount[0] = 0;
      displ[0] = 0;

      for(i = e, proc = 1; i < n; i += e, proc++) {
        displ[proc] = (i-1)*n;          
      }

      for(i = 1; i < np-1; i++)
        sendcount[i] = n*tam;
      sendcount[np-1] = n*tam-n;

      MPI_Scatterv(vector, sendcount, displ, MPI_FLOAT, vector, tam*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
      break;
    }
  }
  tfinishscatter = MPI_Wtime();
  TotalTimescatter = tfinishscatter - tstartscatter;

  // Risolve la matrice
  tstartsolver = MPI_Wtime();
  if(myrank == 0) solver(tam-1, n);
  if(myrank != 0 && myrank != np-1) solver(tam, n);
  if(myrank == np-1) solver(tam-1, n);
  tfinishsolver = MPI_Wtime();
  TotalTimesolver = tfinishsolver - tstartsolver;

  // Gather dei risultati
  tstartgather = MPI_Wtime();
  switch (communication) {
    case 0: { // Comunicazione P2P per il gather
      if(myrank != 0 && myrank != np-1)
        MPI_Send(&vector[n], n*(tam-2), MPI_FLOAT, 0, 1, MPI_COMM_WORLD); 
              
      if(myrank == np-1)
        MPI_Send(&vector[n], n*(tam-3), MPI_FLOAT, 0, 1, MPI_COMM_WORLD); 

      if(myrank == 0) {
        int e = tam - 2, proc = 1, i;
        for(i = e; i < n; i += e) {
          if(proc != np-1)
            MPI_Recv(&vector[i*n], n*(tam-2), MPI_FLOAT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
          else          
            MPI_Recv(&vector[i*n], n*(tam-3), MPI_FLOAT, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          proc++;
        }
      }
      break;
    }

    case 1: { // Comunicazione collettiva per il gather
      int recvcount[np], displ[np];
      int e = tam - 2, proc, i;
      displ[0] = 0;

      for(i = e, proc = 1; i < n; i += e, proc++) {
        displ[proc] = i*n;          
      }

      for(i = 1; i < np; i++)
        recvcount[i] = n*(tam-2);
      recvcount[0] = 0;

      MPI_Gatherv(vector+n, n*(tam-2), MPI_FLOAT, vector, recvcount, displ, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
