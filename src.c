#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define HALO_SIZE 1 // Assuming a 1-cell wide halo region
int d, myrank;
void PrintTemp(double temp[][d], int d, int myrank){
    printf("Rank: %d Size: %d\n", myrank, d);
    for(int i = 0; i < d; i++)
    {
        for(int j = 0; j < d; j++)
        {
            printf("%f ", temp[i][j]);
        }
        printf("\n");
    }
}

      // defining calculate fn
void Calculate(double data[][d], int i, int j, int d, int stencil, double *left, double *right, double *up, double *down, int lflag, int rflag, int uflag, int dflag, double *newSum, int *pntsCnt){

    for(int k = 1; k <= (stencil - 1)/4; k++)
    {
        if(i - k < 0)
        {
            if(uflag)
            {
                *newSum += up[j + (k - i - 1)*d];
                *pntsCnt = *pntsCnt + 1;
            }
        }
        else
        {
            *newSum += data[i - k][j];
            *pntsCnt = *pntsCnt + 1;

        }

        if(i + k > d - 1)
        {
            if(dflag)
            {
                *newSum += down[j + (i + k - d)*d];
                *pntsCnt = *pntsCnt + 1;

            }
        }
        else
        {
            *newSum += data[i + k][j];
            *pntsCnt = *pntsCnt + 1;
        }


        if(j - k < 0)
        {
            if(lflag)
            {
                *newSum += left[i + (k - j - 1)*d];
                *pntsCnt = *pntsCnt + 1;
            }
        }
        else
        {
            *newSum += data[i][j - k];
            *pntsCnt = *pntsCnt + 1;
        }


        if(j + k > d - 1)
        {
            if(rflag)
            {
                *newSum += right[i + (j + k - d)*d];
                *pntsCnt = *pntsCnt + 1;
            }
        }
        else
        {
            *newSum += data[i][j + k];
            *pntsCnt = *pntsCnt + 1;
        }
    }
}



int main(int argc, char *argv[]) {
    int  P;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int Px = atoi(argv[1]);
    int N = atoi(argv[2]);
    int num_time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]);

    int Py = P / Px;
    d = sqrt(N);

    double data[d][d];
    double temp[d][d];

    // populating the data
    // srand(seed * (myrank + 10));
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            // data[i][j] = abs(rand() + (i * rand() + j * myrank)) / 100;
            data[i][j] = i*d + j;
        }
    }

    if(myrank == 5)
        PrintTemp(data, d, myrank);


    // flag signify if grid is present in that direction
    int lflag = 0, rflag = 0, uflag = 0, dflag = 0;

    // to send/receive data
    double sendleft[2*d], sendright[2*d], sendup[2*d], senddown[2*d];
    double recvleft[2*d], recvright[2*d], recvup[2*d], recvdown[2*d];

    // will work as a buffer to store the halo data
    double left[2*d],right[2*d],up[2*d],down[2*d];

    int sends=0;
    // position of the processes
    int Pi = myrank / Px, Pj = myrank % Px;

    if (Pi > 0) uflag = 1;
    if (Pi < Py - 1) dflag = 1;
    if (Pj > 0) lflag = 1;
    if (Pj < Px - 1) rflag = 1;

    double sTime, eTime, totalTime;

    sTime = MPI_Wtime();

    // Halo exchange and stencil computation for num_time_steps
    for (int t = 0; t < num_time_steps; t++) {
        // Non-blocking communication
        MPI_Request req[4];
        MPI_Status status[4];
        int req_idx = 0;
        int status_idx = 0;

        // Send halo regions to neighbors
        if (lflag) {
            int pack_pos = 0;
            // double send_buf[d];
            for (int i = 0; i < d; i++) {
                MPI_Pack(&data[i][0], 1, MPI_DOUBLE, sendleft, 2 * d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack left halo region
                if(stencil == 9) 
                    MPI_Pack(&data[i][1], 1, MPI_DOUBLE, sendleft, 2 * d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack left halo region
            }
            MPI_Isend(sendleft, pack_pos, MPI_PACKED, myrank - 1, myrank-1, MPI_COMM_WORLD, &req[req_idx++]);
        }
        
        if (rflag) {
            int pack_pos = 0;
            // double send_buf[d];
            for (int i = 0; i < d; i++) {
                MPI_Pack(&data[i][d - 1], 1, MPI_DOUBLE, sendright, 2*d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack right halo region
                if(stencil == 9) 
                    MPI_Pack(&data[i][d - 2], 1, MPI_DOUBLE, sendright, 2*d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack right halo region
            }
            MPI_Isend(sendright, pack_pos, MPI_PACKED, myrank + 1, myrank+1, MPI_COMM_WORLD, &req[req_idx++]);
        }

        if (uflag) {
            int pack_pos = 0;
            // double send_buf[d];
            for (int i = 0; i < d; i++) {
                MPI_Pack(&data[0][i], 1, MPI_DOUBLE, sendup, 2*d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack upper halo region
                if(stencil == 9) 
                    MPI_Pack(&data[1][i], 1, MPI_DOUBLE, sendup, 2*d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack upper halo region
            }
            MPI_Isend(sendup, pack_pos, MPI_PACKED, myrank - Px, myrank-Px, MPI_COMM_WORLD, &req[req_idx++]);
        }

        if (dflag) {
            int pack_pos = 0;
            // double send_buf[d];
            for (int i = 0; i < d; i++) {
                // int MPI_Pack (const void *inbuf, int incount, MPI_Datatype datatype,void *outbuf, int outsize, int *position, MPI_Comm comm)
                MPI_Pack(&data[d - 1][i], 1, MPI_DOUBLE, senddown, 2*d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack lower halo region
                if(stencil == 9) 
                    MPI_Pack(&data[d - 2][i], 1, MPI_DOUBLE, senddown, 2*d * sizeof(double), &pack_pos, MPI_COMM_WORLD); // Pack lower halo region
            }
            MPI_Isend(senddown, pack_pos, MPI_PACKED, myrank + Px, myrank+Px, MPI_COMM_WORLD, &req[req_idx++]);
        }


        // Receive halo regions from neighbors
        if (lflag) {
            // double recv_buf[d * d];
            int pack_pos = 0;
            MPI_Recv(recvleft,2* d  * sizeof(double), MPI_PACKED, myrank - 1, myrank, MPI_COMM_WORLD, &status[status_idx++]);
            // Unpack and use recv_buf

            for(int k=0; k<d; k++){
                // int MPI_Unpack (const void *inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm)
                MPI_Unpack(recvleft, 2*d, &pack_pos, &left[k], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                if(stencil == 9) 
                    MPI_Unpack(recvleft, 2*d, &pack_pos, &left[k+d], 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

        if (rflag) {
            // double recv_buf[d * d];
            int pack_pos = 0;
            MPI_Recv(recvright, 2*d  * sizeof(double), MPI_PACKED, myrank + 1, myrank, MPI_COMM_WORLD, &status[status_idx++]);
            // Unpack and use recv_buf
            
            for(int k=0; k<d; k++){
                MPI_Unpack(recvright, 2*d, &pack_pos, &right[k], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                if(stencil == 9) 
                    MPI_Unpack(recvright, 2*d, &pack_pos, &right[k+d], 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

        if (uflag) {
            // double recv_buf[d * d];
            int pack_pos = 0;
            MPI_Recv(recvup, 2*d  * sizeof(double), MPI_PACKED, myrank - Px, myrank, MPI_COMM_WORLD, &status[status_idx++]);
            // Unpack and use recv_buf

            for(int k=0; k<d; k++){
                MPI_Unpack(recvup, 2*d, &pack_pos, &up[k], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                if(stencil == 9) 
                    MPI_Unpack(recvup, 2*d, &pack_pos, &up[k+d], 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

        if (dflag) {
            // double recv_buf[d * d];
            int pack_pos = 0;
            MPI_Recv(recvdown, 2*d  * sizeof(double), MPI_PACKED, myrank + Px, myrank, MPI_COMM_WORLD, &status[status_idx++]);
            // Unpack and use recv_buf

            for(int k=0; k<d; k++){
                MPI_Unpack(recvdown, 2*d, &pack_pos, &down[k], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                if(stencil == 9) 
                    MPI_Unpack(recvdown, 2*d, &pack_pos, &down[k+d], 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }

        MPI_Waitall(req_idx, req, status);
        // Perform stencil computation here using the received halo data

        for(int i = 0; i < d; i++)
        {
            for(int j = 0; j < d; j++)
            {
                double newSum = data[i][j];
                int pntsCnt = 1;
                Calculate(data, i, j, d, stencil, left, right, up, down, lflag, rflag, uflag, dflag, &newSum, &pntsCnt);
                temp[i][j] = newSum / pntsCnt;
            }
        }
        if(myrank == 5)
        PrintTemp(temp, d, myrank);

        for(int i =0 ; i< d; i++){
            for(int j =0; j<d ; j++){
                data[i][j]=temp[i][j];
            }
        }
    }

    eTime = MPI_Wtime();
    double time = eTime - sTime;
    double maxTime;
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myrank==0){
		printf("%lf\n",maxTime);    	
    }

    MPI_Finalize();
    return 0;
}
