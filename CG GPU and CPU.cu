#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "time.h"

#define IDX2C(i,j,ld) (((j)*(ld))+( i ))
void MultMatVec(float* S,float *A1D,float* P);

int im=151, jm=151,N; 
cudaError_t        cudaStat;
cublasStatus_t    stat;
cublasHandle_t    handle;

int main(int argc, char* argv[]) {



  float **A,*B,*X;
  
  int  i=0,j=0,m=0,n=0;
  N=(im+1)*(jm+1);

  // Allocate
  A=(float**)calloc(N,sizeof(float *));
  for(i=0;i<N;i++){
    A[i]=(float*)calloc(N,sizeof(float));
  }
  X=(float *)calloc(N,sizeof(float));
  B=(float *)calloc(N,sizeof(float));
 
//INITIALIZE CUDA EVENTS
cudaEvent_t start,stop;
float elapsedTime;

//CREATING EVENTS
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);

  // Form Matrix A
  float dx2=1./im,dy2=1./jm;
  for (m = 0; m < N; m++){
    j=(int)(m/(im+1)); // For bounday condition
    i=m-(j*(im+1));
    
    if(i==0 || j==0 || i==im || j==jm){
      n=m;
      A[m][n]=1.0;
    }else{
      n=m;
      A[m][n] =-2*((1/dx2)+(1/dy2));

      n=m+1;
      A[m][n]=1./dx2;
      
      n=m-1;
      A[m][n]=1./dx2;
    
      n=m+im+1;
      A[m][n]=1./dy2;
  
      n=m-(im+1);
      A[m][n]=1./dy2;
    }
  }//if not on boundary

  // Form RHS
  float XL=1.,XR=2.,XD=2.,XU=1.;

  for (m = 0; m < N; m++){
    j=(int)(m/(im+1)); 
    i=m-(j*(im+1));
    if(i==0){
           B[m]=XL;
    }
    if(i==im){
          B[m]=XR;
    }
    if(j==0){
          B[m]=XD;
    }
    if(j==jm){
         B[m]=XU;
    }
    if(i==0 && j==0){
          B[m]=(XL+XD)/2;
    }
    if(i==0 && j==jm){
          B[m]=(XL+XU)/2;
    }
    if(i==im && j==0){
         B[m]=(XR+XD)/2;
    }
    if(i==im && j==jm){
          B[m]=(XR+XU)/2;
    }
  }

  //Conjugate gradient
  int k=0,kmax=10;
  float tol=10e-3,res=1.,Beta,Alpha;
  float *r,*P,*S,*Z,*A1D;
  
   r=(float *)calloc(N,sizeof(float));
   P=(float *)calloc(N,sizeof(float));
   S=(float *)calloc(N,sizeof(float));
   Z=(float *)calloc(N,sizeof(float));
   A1D=(float *)calloc(N*N,sizeof(float));

for(j=0;j<N;j++){
    for(i=0;i<N;i++){
        A1D[IDX2C(i,j,N)]=A[i][j];
    }
}


   for (m = 0; m < N; m++){
       X[m]=0.;
       r[m]=B[m];
       P[m]=r[m];
   }
      
   do{
     k+=1;

     
MultMatVec(S,A1D,P);
// printf("K: %d, S[10]: %le \n",S[10]);
 
     float PDr=0.0;
     for (m = 0; m < N; m++){
       PDr+=P[m]*r[m];
     }

     float PDS=0.0;
     for (m = 0; m < N; m++){
       PDS+=P[m]*S[m];
     }
     Alpha=PDr/PDS;

     for (m = 0; m < N; m++){
       X[m]=X[m]+Alpha*P[m];
       r[m]=r[m]-Alpha*S[m];
     }

     MultMatVec(Z,A1D,r);
  
     float PDZ=0.0;
     for (m = 0; m < N; m++){
       PDZ+=P[m]*Z[m];
     }
     Beta=-PDZ/PDS;
    
     for (m = 0; m < N; m++){
       P[m]=r[m]+Beta*P[m];
     }

     float res=0.0;
     for (m = 0; m < N; m++){
       res+=r[m]*r[m];
     }
     res=sqrt(res);

     
     printf("K: %d, X[4]: %le , res:%le , Alpha: %le , Beta : %le \n",k,X[4],res,Alpha,Beta);
     
   } while (abs(res)>tol && k<kmax);

   
	 //FINISH RECORDING
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

//CALCULATE ELAPSED TIME
cudaEventElapsedTime(&elapsedTime,start,stop);

//DISPLAY COMPUTATION TIME
//cout<<"\n\nElapsed Time = "<<elapsedTime<<" ms";
    printf("CPUtime: %le  \n",elapsedTime); 

   free(r);
   free(P);
   free(A);
   free(X);
   free(B);

 
   return 0;
} 

//----------------------------------------------------------------
void MultMatVec(float* S,float *A1D,float* P) {
 


float* d_a;
float* d_x;
float* d_y;

cudaStat=cudaMalloc((void**)&d_a,N*N*sizeof(*A1D));
cudaStat=cudaMalloc((void**)&d_x,N*sizeof(*P));
cudaStat=cudaMalloc((void**)&d_y,N*sizeof(*S));

stat=cublasCreate(&handle);
stat=cublasSetMatrix(N,N,sizeof(*A1D),A1D,N,d_a,N);
stat=cublasSetVector(N,sizeof(*P),P,1,d_x,1);
stat=cublasSetVector(N,sizeof(*S),S,1,d_y,1);

float al=1.0f;
float bet=0.0f;

stat=cublasSgemv(handle,CUBLAS_OP_N,N,N,&al,d_a,N,d_x,1,&bet,d_y,1);

stat=cublasGetVector(N,sizeof(*S),d_y,1,S,1);


}
//----------------------------------------------------------------
