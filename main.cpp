/*
ITML C++
Cheng Zhang, chengz@kth.se 

(C) Copyright 2015, Cheng Zhang 

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License.

The GNU General Public License does not permit this software to be redistributed in proprietary programs.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/
#include <iostream>
#include "itml.h"
#include "crossKNN.h"

//define the cross validation folds
#define N_fold 4

int main(){
    InfoTheoreticMetricLearning metriclearningAlg;
    metriclearningAlg.loadDatafromFile("iris/iris.mtx", "iris/iris.truth");   
    metriclearningAlg.setU(1.0);
    metriclearningAlg.setL(50.0);
    const int numGammas = 4;
    double  gammaAll[numGammas]={0.001,  0.1, 10.0, 1000.0};
    for(int i=0; i<numGammas; ++i){
        double gamma=gammaAll[i];
        std::cout<<"gamma:"<<gamma<<std::endl;
        metriclearningAlg.setGamma(gamma);
        double sumrate=0.0;
        for ( int f=1; f< N_fold+1; ++f){
            std::cout<<"the "<<f<<"fold"<<std::endl;
            metriclearningAlg.setNfoldTrainingdata(f, N_fold);
            metriclearningAlg.metricLearning();      
            metriclearningAlg.computeMahalanobisDis();       
            std::vector<double> EstRes;    
            KNN(metriclearningAlg.getD(), metriclearningAlg.getYtrain(),  EstRes, 5);
            int N_test= metriclearningAlg.getXtestrow();
            auto Groundtruth=metriclearningAlg.getYgroundTruth();
            int correct =0;
            for(int n=0; n< N_test; ++n){
                if(EstRes[n]==Groundtruth[n]){
                    correct ++;
                }
            }            
            double rate=(double) correct/N_test;
            sumrate+=rate;
            std::cout<<"correct rate"<<rate<<std::endl;
        }
        double avgRate=sumrate/N_fold;
        std::cout<<"average rate:"<<avgRate<<std::endl;
    }
    return 0; 
}
