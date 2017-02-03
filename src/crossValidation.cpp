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
#define ALL   60000
#define COMP  784
#define CLASS 10
#define FOLD  10
#define NORM  255


typedef struct {
    int digit;
    double pixels[COMP];
}pic;

typedef struct{
    double count;
    double mean[COMP];
    double variance[COMP];
    double prior[MAX_MAP];
}params;

using namespace std;

int main(int argc, const char * argv[]) {
    
    pic *all = (pic *)calloc(ALL, sizeof(pic));
    
    
    int a,f,d,j,p,i,digit,index,value;
    
    
    //READ IN FILE
    FILE *IN = fopen("mnist", "r");
    FILE *OUT = fopen("OUT.txt", "w");
    
    for(i = 0; i < ALL; i++){
        
        fscanf(IN, "%d",&digit);
        
        all[i].digit = digit;
        
        while(2 == fscanf(IN, " %d:%d",&index,&value))
            all[i].pixels[index] = (double)value / (double)NORM ;
        
        fseek(IN, -1, SEEK_CUR);
    }
    
    fclose(IN);
    //Done Reading file
    int CURR_FOLD,FOLD_SIZE = ALL / FOLD;
    double variance,mean,count, trainSize, error[2],sum, errCount;
    
    pic *validation;
    params *train;

    
    for(a = 0; a < MAX_MAP; a++) {
        for(f = 0; f < FOLD; f++){
            
            
            // Place ALL/FOLD pictures in validation starting at f*ALL/FOLD
            validation = (pic *)calloc(FOLD_SIZE, sizeof(pic));
            
            CURR_FOLD = f*FOLD_SIZE;
            
            for(i = 0; i < FOLD_SIZE; i++)
                validation[i] = all[CURR_FOLD+i];
            
            // Calculate mu and variance for other (FOLD - 1)*ALL/FOLD pictures
            train = (params *)calloc(CLASS, sizeof(params));
            
            for(i = 0; i < ALL; i++){
                if(CURR_FOLD <= i && i < (CURR_FOLD  + FOLD))
                    continue;
                
                digit = all[i].digit;
                train[digit].count += 1.0;
                
                for(p = 0; p < COMP; p++){
                    train[digit].mean[p]     += all[i].pixels[p];
                    train[digit].variance[p] += all[i].pixels[p]*all[i].pixels[p];
                }
            }
            
            
            for(i = 0; i < CLASS; i++){

                count = train[i].count;
                trainSize = ALL - FOLD_SIZE;
                
                // Calculate Priors
                train[i].prior[0] = log(count / trainSize );
                
                for(j = 0; j < MAX_MAP-1; j++)
                    train[i].prior[j+1] = log((count + pow(2,j) * trainSize / 100) / (trainSize + pow(2,j) / 100 * trainSize * CLASS));
                
                // Calculate parameters for guassian distribution
                for(p = 0; p < COMP; p++){
                   
                    mean = train[i].mean[p];
                    variance = train[i].variance[p];

                    train[i].mean[p] = mean / count;
                    train[i].variance[p] = (variance - (mean*mean) / count ) / count;
                    
                }
            }
            
            
            errCount = 0;
            // Run Naive Gaussian on current validation
            for(i = 0; i < FOLD_SIZE; i++){
                error[1] = -DBL_MAX;
                
                for(d = 0; d < CLASS; d++){
                    sum = 0;
                    
                    for(p = 0; p < COMP; p++){
                        if(.02 < train[d].mean[p])
                            sum += -(train[d].mean[p] - validation[i].pixels[p])
                                   *(train[d].mean[p] - validation[i].pixels[p])
                                   /(2*train[d].variance[p])
                            - 0.5*log(2*M_PI*train[d].variance[p]);
                    }
                    
                    sum += train[d].prior[a];
                    
                    if(error[1] < sum){
                        error[0] = d;
                        error[1] = sum;
                    }
                }
                
                if((int)error[0] != validation[i].digit)
                    errCount += 1.0;
            }
            
            fprintf(OUT, "%lf %d %lf\n", pow(2,a-1)/100 ,f,errCount/FOLD_SIZE);
            fprintf(stdout, "%lf %d %lf\n", pow(2,a-1)/100 ,f,errCount/FOLD_SIZE);
            
            
            // Reset Data
            free(validation);
            free(train);
            
        }

        fflush(OUT);
        fflush(stdout);
    }
    
    fclose(OUT);

}




























