#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define HALO_SIZE 1 // Assuming a 1-cell wide halo region

int main(int argc, char *argv[]) {
    int myrank, P;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int Px = atoi(argv[1]);
    int N = atoi(argv[2]);
    int num_time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]);

    int Py = P / Px;
    int d = sqrt(N);

    double data[d][d];
    double temp[d][d];

    srand(seed * (myrank + 10));
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            data[i][j] = abs(rand() + (i * rand() + j * myrank)) / 100;
        }
    }

    int lflag = 0, rflag = 0, uflag = 0, dflag = 0;
    double sendleft[2*d], sendright[2*d], sendup[2*d], senddown[2*d];
    double recvleft[2*d], recvright[2*d], recvup[2*d], recvdown[2*d];
    double left[2*d],right[2*d],up[2*d],down[2*d];

    int sends=0;
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

        if(stencil == 5){
            for (int i = 1; i < d-1; i++) {
                for (int j = 1; j < d-1; j++) {
                    temp[i][j] = (data[i][j] + data[i-1][j] + data[i+1][j] + data[i][j-1] + data[i][j+1]) / 5;
                }
            }
          
            if (Pj==0) {
                for (int i = 1; i < d-1; i++) {
                    temp[i][0] = (data[i][0] + data[i-1][0] + data[i+1][0] + data[i][1]) / 4;
                }
            }
            if (Pj==Px-1) {
                for (int i = 1; i < d-1; i++) {
                    temp[i][d-1] = (data[i][d-1] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2]) / 4;
                }
            }



            if(Pi > 0 && Pj > 0 && Pi < Py-1 && Pj < Px-1){
                for (int i = 1 ; i< d-1 ; i++ ){
                    temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1]) / 5;
                }
                for (int i = 1 ; i< d-1 ; i++ ){
                    temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2]) / 5;
                }
                for (int i = 1 ; i< d-1 ; i++ ){
                    temp[0][i] = (data[0][i] + up[i] + data[0][i-1] + data[0][i+1] + data[1][i]) / 5;
                }

                for (int i = 1 ; i< d-1 ; i++ ){
                    temp[d-1][i] = (data[d-1][i] + down[i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i]) / 5;
                }
                temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0] + left[0]) / 5;
                temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1] + right[0]) / 5;
                temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + left[d-1]) / 5;
                temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + right[d-1]) / 5;

            }

            if(Pi == 0){

                for (int i = 1; i < d-1; i++) {
                    temp[0][i] = (data[0][i] + data[0][i-1] + data[0][i+1] + data[1][i]) / 4;
                }
                if(Pj > 0 && Pj < Px-1){
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1]) / 5;
                    }
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2]) / 5;
                    }

                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[d-1][i] = (data[d-1][i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i] + down[i]) / 5;
                    }

                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + left[d-1]) / 5;
                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + right[d-1]) / 5;                    
                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + left[0]) / 4;
                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + right[0]) / 4;

               }

               if(Pj == 0){
                    for(int i=1 ; i<d ; i++){
                        temp[i][d-1] = (data[i][d-1] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + right[i]) / 5;
                    }
                    for(int i=1 ; i<d ; i++){
                        temp[d-1][i] = (data[d-1][i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i] + down[i]) / 5;
                    }
                    
                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + right[d-1]) / 5;
                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0]) / 4;
                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + right[0]) / 4;
                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0]) / 3;
                    
               }

                if(Pj == Px-1){
                    for(int i=1 ; i<d ; i++){
                        temp[i][0] = (data[i][0] + data[i-1][0] + data[i+1][0] + data[i][1] + left[i]) / 5;
                    }
                    for(int i=1 ; i<d ; i++){
                        temp[d-1][i] = (data[0][i] + data[0][i-1] + data[0][i+1] + data[1][i] + down[i]) / 5;
                    }
                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + left[d-1]) / 5;
                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + left[0]) / 4;
                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1]) / 4;
                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1]) / 3;


                }

                    
            }

            if(Pi == Py-1){

                for (int i = 1; i < d-1; i++) {
                    temp[d-1][i] = (data[d-1][i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i]) / 4;
                }
                if(Pj > 0 && Pj < Px-1){
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1]) / 5;
                    }
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2]) / 5;
                    }

                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[0][i] = (data[0][i] + data[0][i-1] + data[0][i+1] + data[1][i] + up[i]) / 5;
                    }

                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0] + left[0]) / 5;
                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1] + right[0]) / 5;
                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + left[d-1]) / 4;
                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + right[d-1]) / 4;

               }
               if(Pj == 0){
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[0][i] = (data[0][i] + data[0][i-1] + data[0][i+1] + data[1][i] + up[i]) / 5;
                    }
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2]) / 5;
                    }

                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0]) / 3;
                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0]) / 4;
                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1]+ right[d-1]) / 4;
                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1]) / 4;

                    

               }

               if(Pj == Px-1) {
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[0][i] = (data[0][i] + data[0][i-1] + data[0][i+1] + data[1][i] + up[i]) / 5;
                    }
                    for (int i = 1 ; i< d-1 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1]) / 5;
                    }

                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1]) / 3;
                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0]) / 4;
                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0]+ left[d-1]) / 4;
                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1]) / 4;

               
               }



            }

                   
        }

        else if(stencil == 9){
            for (int i = 2 ; i< d-2 ; i++ ){
                for (int j = 2 ; j< d-2 ; j++ ){
                    temp[i][j] = (data[i][j] + data[i-1][j] + data[i+1][j] + data[i][j-1] + data[i][j+1] + data[i-2][j] + data[i+2][j] + data[i][j-2] + data[i][j+2]) / 9;
                }
            }

            if (Pj==0) {
                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[i][0] = (data[i][0] + data[i-1][0] + data[i+1][0] + data[i][1] + data[i-2][0] + data[i+2][0]+ data[i][2])  / 7;
                    temp[i][1] = (data[i][1] + data[i-1][1] + data[i+1][1] + data[i][2] + data[i-2][1] + data[i+2][1] + data[i][0] + data[i][3] ) / 8;
                }
            }

            if (Pj==Px-1) {
                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[i][d-1] = (data[i][d-1] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + data[i-2][d-1] + data[i+2][d-1] + data[i][d-3]) / 7;
                    temp[i][d-2] = (data[i][d-2] + data[i-1][d-2] + data[i+1][d-2] + data[i][d-3] + data[i-2][d-2] + data[i+2][d-2] + data[i][d-1] + data[i][d-4]) / 8;
                }
            }

            if(Pi > 0 && Pj > 0 && Pi < Py-1 && Pj < Px-1){
                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1] + left[i+d] + data[i-2][0] + data[i+2][0] + data[i][2]) / 9;
                    temp[i][1] = (data[i][1] + data[i-1][1] + data[i+1][1] + data[i][2] + data[i][0] + left[i] + data[i-2][1] + data[i+2][1] + data[i][3] ) / 9;  
                }
                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + right[i+d] + data[i-2][d-1] + data[i+2][d-1] + data[i][d-3]) / 9;
                    temp[i][d-2] = (data[i][d-2] + data[i-1][d-2] + data[i+1][d-2] + data[i][d-3] + data[i][d-1] + right[i] + data[i-2][d-2] + data[i+2][d-2] + data[i][d-4]) / 9;
                }
                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[0][i] = (data[0][i] + up[i] + data[0][i-1] + data[0][i+1] + data[1][i] + up[i+d] + data[2][i] + data[0][i-2] + data[0][i+2]) / 9;
                    temp[1][i] = (data[1][i] + data[0][i] + data[2][i] + data[1][i-1] + data[1][i+1] + up[i] + data[3][i] + data[1][i-2] + data[1][i+2]) / 9;
                }
                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[d-1][i] = (data[d-1][i] + down[i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i] + down[i+d] + data[d-3][i] + data[d-1][i-2] + data[d-1][i+2]) / 9;
                    temp[d-2][i] = (data[d-2][i] + data[d-1][i] + data[d-3][i] + data[d-2][i-1] + data[d-2][i+1] + down[i] + data[d-4][i] + data[d-2][i-2] + data[d-2][i+2]) / 9;
                }

                temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0] + left[0] + up[d] + left[d] + data[2][0] + data[0][2]) / 9;
                temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + up[1] + data[1][1] + left[0]+ data[0][2]+ up[d+1] + data[2][1]) / 9;
                temp[1][0] = (data[1][0] + left[1] + data[1][1] + data[0][0] + data[2][0] + left[1+d] + data[1][2] + up[0] + data[3][0]) / 9;
                temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + left[1] + data[1][3] + up[1] + data[3][1]) / 9;

                temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1] + right[0] + up[d-1+d] + right[d-1] + data[2][d-1] + data[0][d-3]) / 9;
                temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + up[d-2] + data[1][d-2] + right[0] + data[0][d-3] + up[d-2+d] + data[2][d-2]) / 9;
                temp[1][d-1] = (data[1][d-1] + right[1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + right[1+d] + data[1][d-3] + up[d-1] + data[3][d-1]) / 9;
                temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + right[1] + data[1][d-4] + up[d-2] + data[3][d-2]) / 9;

                temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + left[d-1] + down[0+d] + left[d-1+d] + data[d-3][0] + data[d-1][2]) / 9;
                temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + down[1] + data[d-2][1] + left[d-1] + data[d-1][2] + down[1+d] + data[d-3][1]) / 9;
                temp[d-2][0] = (data[d-2][0] + left[d-2] + data[d-2][1] + data[d-1][0] + data[d-3][0] + left[d-2+d] + data[d-2][2] + down[d-2] + data[d-4][0]) / 9;
                temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + left[d-2] + data[d-2][3] + down[d-2] + data[d-4][1]) / 9;

                temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + right[d-1] + down[d-1+d] + right[d-1+d] + data[d-3][d-1] + data[d-1][d-3]) / 9;
                temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + down[d-2] + data[d-2][d-2] + right[d-1] + data[d-1][d-3] + down[d-2+d] + data[d-3][d-2]) / 9;
                temp[d-2][d-1] = (data[d-2][d-1] + right[d-2] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + right[d-2+d] + data[d-2][d-3] + down[d-2] + data[d-4][d-1]) / 9;
                temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + right[d-2] + data[d-2][d-4] + down[d-2] + data[d-4][d-2]) / 9;



            }

            if(Pi == 0){

                for (int i = 2; i < d-2; i++) {
                    temp[0][i] = (data[0][i] + data[0][i-1] + data[0][i+1] + data[1][i] + data[0][i-2] + data[0][i+2] + data[2][i]) / 7;
                    temp[1][i] = (data[1][i] + data[0][i] + data[2][i] + data[1][i-1] + data[1][i+1]+ data[3][i] + data[1][i-2] + data[1][i+2] ) / 8;
                }

                for (int i = 2; i< d-2 ; i++){
                    temp[d-1][i]= (data[d-1][i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i] + down[i] + data[d-1][i-2] + data[d-1][i+2] + data[d-3][i] + down[i+d])/9;
                    temp[d-2][i] = (data[d-2][i] + data[d-2][i-1] + data[d-2][i+1] + data[d-3][i] + down[i] + data[d-2][i-2] + data[d-2][i+2]+  data[d-4][i] + data[d-1][i])/9;
                }

                if(Pj > 0 && Pj < Px-1){
                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1] + left[i+d] + data[i-2][0] + data[i+2][0] + data[i][2]) / 9;
                        temp[i][1] = (data[i][1] + data[i-1][1] + data[i+1][1] + data[i][2] + data[i][0] + left[i] + data[i-2][1] + data[i+2][1] + data[i][3]) / 9;  
                    }
                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + right[i+d] + data[i-2][d-1] + data[i+2][d-1] + data[i][d-3]) / 9;
                        temp[i][d-2] = (data[i][d-2] + data[i-1][d-2] + data[i+1][d-2] + data[i][d-3] + data[i][d-1] + right[i] + data[i-2][d-2] + data[i+2][d-2] + data[i][d-4]) / 9;
                    }


                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + left[0] + left[d] + data[2][0] + data[0][2]) / 7;
                    temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + data[1][1] + left[0]+ data[0][2] + data[2][1]) / 7;
                    temp[1][0] = (data[1][0] + left[1] + data[1][1] + data[0][0] + data[2][0] + left[1+d] + data[1][2] + data[3][0]) / 8;
                    temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + left[1] + data[1][3] + data[3][1]) / 8;

                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + right[0] + right[d-1] + data[2][d-1] + data[0][d-3]) / 7;
                    temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + data[1][d-2] + right[0] + data[0][d-3] + data[2][d-2]) / 7;
                    temp[1][d-1] = (data[1][d-1] + right[1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + right[1+d] + data[1][d-3] + data[3][d-1]) / 8;
                    temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + right[1] + data[1][d-4] + data[3][d-2]) / 8;


                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + left[d-1] + down[0+d] + left[d-1+d] + data[d-3][0] + data[d-1][2]) / 9;
                    temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + down[1] + data[d-2][1] + left[d-1] + data[d-1][2] + down[1+d] + data[d-3][1]) / 9;
                    temp[d-2][0] = (data[d-2][0] + left[d-2] + data[d-2][1] + data[d-1][0] + data[d-3][0] + left[d-2+d] + data[d-2][2] + down[d-2] + data[d-4][0]) / 9;
                    temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + left[d-2] + data[d-2][3] + down[d-2] + data[d-4][1]) / 9;

                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + right[d-1] + down[d-1+d] + right[d-1+d] + data[d-3][d-1] + data[d-1][d-3]) / 9;
                    temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + down[d-2] + data[d-2][d-2] + right[d-1] + data[d-1][d-3] + down[d-2+d] + data[d-3][d-2]) / 9;
                    temp[d-2][d-1] = (data[d-2][d-1] + right[d-2] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + right[d-2+d] + data[d-2][d-3] + down[d-2] + data[d-4][d-1]) / 9;
                    temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + right[d-2] + data[d-2][d-4] + down[d-2] + data[d-4][d-2]) / 9;



                    


                }


                if(Pj==0){

                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + right[i+d] + data[i-2][d-1] + data[i+2][d-1] + data[i][d-3]) / 9;
                        temp[i][d-2] = (data[i][d-2] + data[i-1][d-2] + data[i+1][d-2] + data[i][d-3] + data[i][d-1] + right[i] + data[i-2][d-2] + data[i+2][d-2] + data[i][d-4]) / 9;
                    }

                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + data[2][0] + data[0][2]) / 5;
                    temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + data[1][1] + data[0][2] + data[2][1]) / 6;
                    temp[1][0] = (data[1][0] + data[1][1] + data[0][0] + data[2][0] + data[1][2] + data[3][0]) / 6;
                    temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + data[1][3] + data[3][1]) / 7;

                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + right[0] + right[d-1] + data[2][d-1] + data[0][d-3]) / 7;
                    temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + data[1][d-2] + right[0] + data[0][d-3] + data[2][d-2]) / 7;
                    temp[1][d-1] = (data[1][d-1] + right[1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + right[1+d] + data[1][d-3] + data[3][d-1]) / 8;
                    temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + right[1] + data[1][d-4] + data[3][d-2]) / 8;


                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + down[0+d] + data[d-3][0] + data[d-1][2]) / 7;
                    temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + down[1] + data[d-2][1] + data[d-1][2] + down[1+d] + data[d-3][1]) / 8;
                    temp[d-2][0] = (data[d-2][0] + data[d-2][1] + data[d-1][0] + data[d-3][0] + data[d-2][2] + down[d-2] + data[d-4][0]) / 7;
                    temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + data[d-2][3] + down[d-2] + data[d-4][1]) / 8;

                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + right[d-1] + down[d-1+d] + right[d-1+d] + data[d-3][d-1] + data[d-1][d-3]) / 9;
                    temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + down[d-2] + data[d-2][d-2] + right[d-1] + data[d-1][d-3] + down[d-2+d] + data[d-3][d-2]) / 9;
                    temp[d-2][d-1] = (data[d-2][d-1] + right[d-2] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + right[d-2+d] + data[d-2][d-3] + down[d-2] + data[d-4][d-1]) / 9;
                    temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + right[d-2] + data[d-2][d-4] + down[d-2] + data[d-4][d-2]) / 9;


                }

                if(Pj == Px-1){
                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1] + left[i+d] + data[i-2][0] + data[i+2][0] + data[i][2]) / 9;
                        temp[i][1] = (data[i][1] + data[i-1][1] + data[i+1][1] + data[i][2] + data[i][0] + left[i] + data[i-2][1] + data[i+2][1] + data[i][3]) / 9;  
                    }


                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + left[0] + left[d] + data[2][0] + data[0][2]) / 7;
                    temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + data[1][1] + left[0]+ data[0][2] + data[2][1]) / 7;
                    temp[1][0] = (data[1][0] + left[1] + data[1][1] + data[0][0] + data[2][0] + left[1+d] + data[1][2] + data[3][0]) / 8;
                    temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + left[1] + data[1][3] + data[3][1]) / 8;

                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + data[2][d-1] + data[0][d-3]) / 5;
                    temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + data[1][d-2] + data[0][d-3] + data[2][d-2]) / 6;
                    temp[1][d-1] = (data[1][d-1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + data[1][d-3] + data[3][d-1]) / 6;
                    temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + data[1][d-4] + data[3][d-2]) / 7;


                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + down[0] + left[d-1] + down[0+d] + left[d-1+d] + data[d-3][0] + data[d-1][2]) / 9;
                    temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + down[1] + data[d-2][1] + left[d-1] + data[d-1][2] + down[1+d] + data[d-3][1]) / 9;
                    temp[d-2][0] = (data[d-2][0] + left[d-2] + data[d-2][1] + data[d-1][0] + data[d-3][0] + left[d-2+d] + data[d-2][2] + down[d-2] + data[d-4][0]) / 9;
                    temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + left[d-2] + data[d-2][3] + down[d-2] + data[d-4][1]) / 9;

                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + down[d-1] + down[d-1+d] + data[d-3][d-1] + data[d-1][d-3]) / 7;
                    temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + down[d-2] + data[d-2][d-2] + data[d-1][d-3] + down[d-2+d] + data[d-3][d-2]) / 8;
                    temp[d-2][d-1] = (data[d-2][d-1] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + data[d-2][d-3] + down[d-2] + data[d-4][d-1]) / 7;
                    temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + data[d-2][d-4] + down[d-2] + data[d-4][d-2]) / 8;

                }
            }


            if(Pi==Py-1){

                for (int i = 2 ; i< d-2 ; i++ ){
                    temp[0][i] = (data[0][i] + up[i] + data[0][i-1] + data[0][i+1] + data[1][i] + up[i+d] + data[2][i] + data[0][i-2] + data[0][i+2]) / 9;
                    temp[1][i] = (data[1][i] + data[0][i] + data[2][i] + data[1][i-1] + data[1][i+1] + up[i] + data[3][i] + data[1][i-2] + data[1][i+2]) / 9;
                }
                for (int i = 2; i< d-2 ; i++){
                    temp[d-1][i]= (data[d-1][i] + data[d-1][i-1] + data[d-1][i+1] + data[d-2][i] + data[d-1][i-2] + data[d-1][i+2] + data[d-3][i])/7;
                    temp[d-2][i] = (data[d-2][i] + data[d-2][i-1] + data[d-2][i+1] + data[d-3][i] + data[d-2][i-2] + data[d-2][i+2] + data[d-4][i] + data[d-1][i])/8;
                }

                if(Pj>0 && Pj < Px-1){
                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1] + left[i+d] + data[i-2][0] + data[i+2][0] + data[i][2]) / 9;
                        temp[i][1] = (data[i][1] + data[i-1][1] + data[i+1][1] + data[i][2] + data[i][0] + left[i] + data[i-2][1] + data[i+2][1] + data[i][3]) / 9;  
                    }
                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + right[i+d] + data[i-2][d-1] + data[i+2][d-1] + data[i][d-3]) / 9;
                        temp[i][d-2] = (data[i][d-2] + data[i-1][d-2] + data[i+1][d-2] + data[i][d-3] + data[i][d-1] + right[i] + data[i-2][d-2] + data[i+2][d-2] + data[i][d-4]) / 9;
                    }


                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0] + left[0] + up[d] + left[d] + data[2][0] + data[0][2]) / 9;
                    temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + up[1] + data[1][1] + left[0]+ data[0][2]+ up[d+1] + data[2][1]) / 9;
                    temp[1][0] = (data[1][0] + left[1] + data[1][1] + data[0][0] + data[2][0] + left[1+d] + data[1][2] + up[0] + data[3][0]) / 9;
                    temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + left[1] + data[1][3] + up[1] + data[3][1]) / 9;

                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1] + right[0] + up[d-1+d] + right[d-1] + data[2][d-1] + data[0][d-3]) / 9;
                    temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + up[d-2] + data[1][d-2] + right[0] + data[0][d-3] + up[d-2+d] + data[2][d-2]) / 9;
                    temp[1][d-1] = (data[1][d-1] + right[1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + right[1+d] + data[1][d-3] + up[d-1] + data[3][d-1]) / 9;
                    temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + right[1] + data[1][d-4] + up[d-2] + data[3][d-2]) / 9;


                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + left[d-1] + left[d-1+d] + data[d-3][0] + data[d-1][2]) / 7;
                    temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + data[d-2][1] + left[d-1] + data[d-1][2] + data[d-3][1]) / 7;
                    temp[d-2][0] = (data[d-2][0] + left[d-2] + data[d-2][1] + data[d-1][0] + data[d-3][0] + left[d-2+d] + data[d-2][2] + data[d-4][0]) / 8;
                    temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + left[d-2] + data[d-2][3] + data[d-4][1]) / 8;

                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + right[d-1] + right[d-1+d] + data[d-3][d-1] + data[d-1][d-3]) / 7;
                    temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + data[d-2][d-2] + right[d-1] + data[d-1][d-3] + data[d-3][d-2]) / 7;
                    temp[d-2][d-1] = (data[d-2][d-1] + right[d-2] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + right[d-2+d] + data[d-2][d-3] + data[d-4][d-1]) / 8;
                    temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + right[d-2] + data[d-2][d-4] + data[d-4][d-2]) / 8;



                }


                if(Pj==0){
                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][d-1] = (data[i][d-1] + right[i] + data[i-1][d-1] + data[i+1][d-1] + data[i][d-2] + right[i+d] + data[i-2][d-1] + data[i+2][d-1] + data[i][d-3]) / 9;
                        temp[i][d-2] = (data[i][d-2] + data[i-1][d-2] + data[i+1][d-2] + data[i][d-3] + data[i][d-1] + right[i] + data[i-2][d-2] + data[i+2][d-2] + data[i][d-4]) / 9;
                    }

                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0] + up[d] + data[2][0] + data[0][2]) / 7;
                    temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + up[1] + data[1][1] + data[0][2]+ up[d+1] + data[2][1]) / 8;
                    temp[1][0] = (data[1][0] + data[1][1] + data[0][0] + data[2][0] + data[1][2] + up[0] + data[3][0]) / 7;
                    temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + data[1][3] + up[1] + data[3][1]) / 8;

                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1] + right[0] + up[d-1+d] + right[d-1] + data[2][d-1] + data[0][d-3]) / 9;
                    temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + up[d-2] + data[1][d-2] + right[0] + data[0][d-3] + up[d-2+d] + data[2][d-2]) / 9;
                    temp[1][d-1] = (data[1][d-1] + right[1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + right[1+d] + data[1][d-3] + up[d-1] + data[3][d-1]) / 9;
                    temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + right[1] + data[1][d-4] + up[d-2] + data[3][d-2]) / 9;


                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + data[d-3][0] + data[d-1][2]) / 5;
                    temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + data[d-2][1] + data[d-1][2] + data[d-3][1]) / 6;
                    temp[d-2][0] = (data[d-2][0] + data[d-2][1] + data[d-1][0] + data[d-3][0] + data[d-2][2] + data[d-4][0]) / 6;
                    temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + data[d-2][3] + data[d-4][1]) / 7;

                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + right[d-1] + right[d-1+d] + data[d-3][d-1] + data[d-1][d-3]) / 7;
                    temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + data[d-2][d-2] + right[d-1] + data[d-1][d-3] + data[d-3][d-2]) / 7;
                    temp[d-2][d-1] = (data[d-2][d-1] + right[d-2] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + right[d-2+d] + data[d-2][d-3] + data[d-4][d-1]) / 8;
                    temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + right[d-2] + data[d-2][d-4] + data[d-4][d-2]) / 8;


                }


                if(Pj == Px-1){

                    for (int i = 2 ; i< d-2 ; i++ ){
                        temp[i][0] = (data[i][0] + left[i] + data[i-1][0] + data[i+1][0] + data[i][1] + left[i+d] + data[i-2][0] + data[i+2][0] + data[i][2]) / 9;
                        temp[i][1] = (data[i][1] + data[i-1][1] + data[i+1][1] + data[i][2] + data[i][0] + left[i] + data[i-2][1] + data[i+2][1] + data[i][3]) / 9;  
                    }

                    temp[0][0] = (data[0][0] + data[0][1] + data[1][0] + up[0] + left[0] + up[d] + left[d] + data[2][0] + data[0][2]) / 9;
                    temp[0][1] = (data[0][1] + data[0][0] + data[0][2] + up[1] + data[1][1] + left[0]+ data[0][2]+ up[d+1] + data[2][1]) / 9;
                    temp[1][0] = (data[1][0] + left[1] + data[1][1] + data[0][0] + data[2][0] + left[1+d] + data[1][2] + up[0] + data[3][0]) / 9;
                    temp[1][1] = (data[1][1] + data[0][1] + data[2][1] + data[1][0] + data[1][2] + left[1] + data[1][3] + up[1] + data[3][1]) / 9;

                    temp[0][d-1] = (data[0][d-1] + data[0][d-2] + data[1][d-1] + up[d-1] + up[d-1+d] + data[2][d-1] + data[0][d-3]) / 7;
                    temp[0][d-2] = (data[0][d-2] + data[0][d-1] + data[0][d-3] + up[d-2] + data[1][d-2] + data[0][d-3] + up[d-2+d] + data[2][d-2]) / 8;
                    temp[1][d-1] = (data[1][d-1] + data[1][d-2] + data[0][d-1] + data[2][d-1] + data[1][d-3] + up[d-1] + data[3][d-1]) / 7;
                    temp[1][d-2] = (data[1][d-2] + data[0][d-2] + data[2][d-2] + data[1][d-1] + data[1][d-3] + data[1][d-4] + up[d-2] + data[3][d-2]) / 8;


                    temp[d-1][0] = (data[d-1][0] + data[d-1][1] + data[d-2][0] + left[d-1] + left[d-1+d] + data[d-3][0] + data[d-1][2]) / 7;
                    temp[d-1][1] = (data[d-1][1] + data[d-1][0] + data[d-1][2] + data[d-2][1] + left[d-1] + data[d-1][2] + data[d-3][1]) / 7;
                    temp[d-2][0] = (data[d-2][0] + left[d-2] + data[d-2][1] + data[d-1][0] + data[d-3][0] + left[d-2+d] + data[d-2][2] + data[d-4][0]) / 8;
                    temp[d-2][1] = (data[d-2][1] + data[d-1][1] + data[d-3][1] + data[d-2][0] + data[d-2][2] + left[d-2] + data[d-2][3] + data[d-4][1]) / 8;


                    temp[d-1][d-1] = (data[d-1][d-1] + data[d-1][d-2] + data[d-2][d-1] + data[d-3][d-1] + data[d-1][d-3]) / 5;
                    temp[d-1][d-2] = (data[d-1][d-2] + data[d-1][d-1] + data[d-1][d-3] + data[d-2][d-2] + data[d-1][d-3] + data[d-3][d-2]) / 6;
                    temp[d-2][d-1] = (data[d-2][d-1] + data[d-2][d-2] + data[d-1][d-1] + data[d-3][d-1] + data[d-2][d-3] + data[d-4][d-1]) / 6;
                    temp[d-2][d-2] = (data[d-2][d-2] + data[d-1][d-2] + data[d-3][d-2] + data[d-2][d-1] + data[d-2][d-3] + data[d-2][d-4] + data[d-4][d-2]) / 7;


                }



            }



           

        }


        for(int i =0 ; i< d; i++){
            for(int j =0; j<d ; j++){
                data[i][j]=temp[i][j];
            }
        }
    }

    eTime = MPI_Wtime();
    double time = eTime-sTime;
    double maxTime;
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(myrank==0){
		printf("%lf\n",maxTime);    	
    }

    MPI_Finalize();
    return 0;
}
