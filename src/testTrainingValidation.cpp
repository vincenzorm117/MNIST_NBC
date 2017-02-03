//
//  main.cpp
//  NaiveBayesClassifier
//
//  Created by Vincenzo on 9/10/14.
//  Copyright (c) 2014 MachineLearning. All rights reserved.
//

#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cmath>
#include <tgmath.h>
#include <cfloat>




// Constants
#define MAX_MAP 16
#define E     1E-3
#define TEST  10000
#define VALD  10000
#define TRAIN 40000
#define COMP  784
#define CLASS 10


using namespace std;

int main(int argc, const char * argv[]) {
   
   
   double count[10];
   double prior[10][MAX_MAP];
   
   double *training   = (double *)calloc(CLASS*COMP*2,  sizeof(double));
   double *validation = (double *)calloc(VALD*(COMP+1), sizeof(double));
   double *test       = (double *)calloc(TEST*(COMP+1), sizeof(double));
   
   bzero(count, sizeof(double)*10);
   
   
   
   //READ File
   FILE *IN = fopen("mnist", "r");
   
   int i,j,digit,index,value;
   for(i = 0; i < TEST; i++){
       fscanf(IN, "%d", &digit);
       
       test[i*(COMP+1) + COMP] = digit;
       
       while(2 == fscanf(IN, " %d:%d",&index,&value))
           test[i*(COMP+1) + index] = (double)value / 255;
       
       
       fseek(IN, -1, SEEK_CUR);
   }
   
   for(i = 0; i < TEST; i++){
       fscanf(IN, "%d", &digit);
       
       validation[i*(COMP+1) + COMP] = digit;
       
       while(2 == fscanf(IN, " %d:%d",&index,&value))
           validation[i*(COMP+1) + index] = (double)value / 255;
       
       
       fseek(IN, -1, SEEK_CUR);
   }
   
   for(i = 0; i < TRAIN; i++){
       fscanf(IN, "%d", &digit);
       count[digit] += 1;
       
       while(2 == fscanf(IN, " %d:%d",&index,&value)){
           training[digit*COMP*2 + index] += (double)(value) / 255.0;
           training[digit*COMP*2 + COMP + index] += (double)(value)*(double)(value) / (255.0*255.0);
       }
       fseek(IN, -1, SEEK_CUR);
       
   }
   
  
   
   
   fclose(IN);
   // Done Reading File
   
   // Calculate Distribution values for each digit
   double mean, variance,sum, sqrSum, n;
   for(i = 0; i < CLASS; i++){
       n = count[i];
       for(j = 0; j < COMP; j++){
           sum    = training[i*COMP*2 + j];
           sqrSum = training[i*COMP*2 + COMP + j];
           mean = sum / n;
           variance = (sqrSum - (sum*sum/n)) / n;
           
           training[i*COMP*2 + j] = mean;
           training[i*COMP*2 + COMP + j] = variance;
       }
   }
   
   //Calculate Prior for both MAP and MLE
   for(i = 0; i < 10; i++)
       prior[i][0] = log(count[i]/TRAIN);
   
   for(i = 0; i < CLASS; i++)
       for(j = 0; j < MAX_MAP-1; j++)
           prior[i][j+1] = (count[i] + pow(2, j)/(100)*TRAIN) / (TRAIN*(1+pow(2, j)/10*10.0));
   
   
   
   
   
   printf("Alpha Error MinAlpha\n");
   
   double err = 0.0;
   sum = 0;
   double error[2];
   double alpha[2];
   int d,p,a;
   
   alpha[0] = 100;
   alpha[1] = DBL_MAX;
   
   for(a = 0; a < MAX_MAP; a++){
       err = 0;
       
       
       for(i = 0; i < VALD; i++){
           error[1] = -DBL_MAX;
           
           for(d = 0; d < CLASS; d++){
               sum = 0;
               
               for(p = 0; p < COMP; p++){
                   
                   if(training[d*COMP*2 + COMP + p] != 0)
                       sum += -(validation[i*(COMP+1) + p] - training[d*COMP*2 + p])*(validation[i*(COMP+1) + p] - training[d*COMP*2 + p]) / (2*training[d*COMP*2 + COMP + p])
                       - 0.5*log(2*M_PI*training[d*COMP*2 + COMP + p]);
               }
               
               sum += prior[d][a];
               
               if(error[1] < sum){
                   error[0] = d;
                   error[1] = sum;
               }
           }
           
           if((int)error[0] != (int)validation[i*(COMP+1) + COMP])
               err += 1.0;
       }
       
       if(err < alpha[1]){
           alpha[1] = err;
           alpha[0] = a;
       }
       
       
       fprintf(stdout, "%9.5lf %3.5lf %3.5lf\n", pow(2,a-1)/100, err/10000.0, pow(2,alpha[0]-1)/100);
   }
   
   
   
   for(a = 0; a < MAX_MAP; a++){
       err = 0;
       for(i = 0; i < TEST; i++){
           error[1] = -DBL_MAX;
           for(d = 0; d < CLASS; d++){
               sum = 0;
               for(p = 0; p < COMP; p++){
                   
                   if(training[d*COMP*2 + COMP + p] != 0)
                       sum += -(test[i*(COMP+1) + p] - training[d*COMP*2 + p])*(test[i*(COMP+1) + p] - training[d*COMP*2 + p]) / (2*training[d*COMP*2 + COMP + p])
                       - 0.5*log(2*M_PI*training[d*COMP*2 + COMP + p]);
                   
               }
               //                sum += log(count[d]) - log(50000.0);
               sum += prior[d][a];
               
               if( error[1] < sum){
                   error[0] = d;
                   error[1] = sum;
               }
               
           }
           
           if((int)error[0] != (int)test[i*(COMP+1) + COMP]){
               err += 1.0;
           }
           
           // Determine the argmax over d
           // Add to error buffer
       }
       
       fprintf(stdout, "%lf %lf\n", pow(2,a-1)/100, err/10000.0);
   }
   
   
   
   
   free(training);
   free(test);
}
















