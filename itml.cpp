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
#include <fstream>
#include <sstream>
#include <stdlib.h> 
#include <iterator>
#include <algorithm>
#include <math.h>
#include <unistd.h>
#include "itml.h"

void InfoTheoreticMetricLearning::setU(double pU)
	{this->u=pU;}

void InfoTheoreticMetricLearning::setL(double pL)
	{this->l=pL;}

void InfoTheoreticMetricLearning::setGamma(double pGamma)
{this->gamma=pGamma;}

std::vector<std::vector<double> > InfoTheoreticMetricLearning::getXtrain(){
        return Xtrain;
    }

std::vector<double> InfoTheoreticMetricLearning::getYtrain(){
        return Ytrain;
    }

std::vector<std::vector<double> > InfoTheoreticMetricLearning::getXtest(){
        return Xtest;
    }

std::vector<double> InfoTheoreticMetricLearning::getYgroundTruth(){
        return YgroundTruth;
    }

std::vector<std::vector<double> > InfoTheoreticMetricLearning::getA(){
        return A;
    }

std::vector<std::vector<double> > InfoTheoreticMetricLearning::getD(){
        return D;
    }

int  InfoTheoreticMetricLearning::getXtrainrow(){
    return this->Xtrain.size();
}

int  InfoTheoreticMetricLearning::getXtraincol(){
    return this->Xtrain[0].size();
}

int  InfoTheoreticMetricLearning::getYtrainrow(){
    return this->Ytrain.size();
}

int  InfoTheoreticMetricLearning::getXtestrow(){
    return this->Xtest.size();
}

double InfoTheoreticMetricLearning::Xtrainat(int r, int c){
    return Xtrain[r][c];
}

double InfoTheoreticMetricLearning::Ytrainat(int r){
    return Ytrain[r];
}

void InfoTheoreticMetricLearning::loadDatafromFile(std::string filenameX, std::string filenameY){
    //load X
    std::cout<<"load data from "<<filenameX<< ". . . " <<std::flush;
    std::ifstream Xfile;
    Xfile.open(filenameX.c_str());
    std::string line;
    if(Xfile.is_open()){
        while(std::getline(Xfile, line)){
            std::vector<std::string> xStr;
            std::vector<double> x;
            std::istringstream iss(line);
            copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), back_inserter(xStr));
            
            for(auto it=xStr.begin(); it !=xStr.end(); ++it){
                x.push_back(atof((*it).c_str()));
            }

            this->X.push_back(x);
        }
        
    }else {
        std::cout<<"Error opening "<< filenameX<<std::endl;
        abort();
    }

    Xfile.close();
    std::cout<<"done"<<std::endl;

    std::cout<<"X:"<<X.size()<<"x"<<X[0].size()<<std::endl;
    
    //load Y
    std::cout<<"load data from "<<filenameY<< ". . . " <<std::flush;
    std::ifstream Yfile;
    Yfile.open(filenameY.c_str());
    if(Yfile.is_open()){
        while(std::getline(Yfile, line)){
            double y = atof(line.c_str()); 
            this->Y.push_back(y);
        }
        
    }else {
        std::cout<<"Error opening "<< filenameY<<std::endl;
        abort();
    }

    Yfile.close();
    std::cout<<"done"<<std::endl;

    std::cout<<"Y:"<<Y.size()<<std::endl;
}

void 	InfoTheoreticMetricLearning::setNfoldTrainingdata(const int f, const int N_fold ){
    int oneFold = this->X.size()/N_fold;
    this->Xtest.clear();
    this->YgroundTruth.clear();
    std::copy(this->Y.end()-f*oneFold ,this->Y.end()-(f-1)*oneFold ,std::back_inserter(this->YgroundTruth));
    std::copy(this->X.end()-f*oneFold ,this->X.end()-(f-1)*oneFold ,std::back_inserter(this->Xtest));
    //std::cout<<"Ytest size:"<<this->YgroundTruth.size()<<std::endl;
    //std::cout<<"Xtest size:"<<this->Xtest.size()<<" "<<this->Xtest[0].size()<<std::endl;
    
    this->Xtrain.clear();
    this->Ytrain.clear();
    std::copy(this->Y.begin() ,this->Y.end()-f*oneFold ,std::back_inserter(this->Ytrain));
    std::copy(this->X.begin() ,this->X.end()-f*oneFold ,std::back_inserter(this->Xtrain));
    std::copy(this->Y.end()-(f-1)*oneFold, this->Y.end() ,std::back_inserter(this->Ytrain));
    std::copy(this->X.end() - (f-1)*oneFold ,this->X.end() ,std::back_inserter(this->Xtrain));
    //std::cout<<"Ytrain size:"<<this->Ytrain.size()<<std::endl;
    //std::cout<<"Xtrain size:"<<this->Xtrain.size()<<" "<<this->Xtrain[0].size()<<std::endl;
}

void 	InfoTheoreticMetricLearning::metricLearning(){
    //TODO dunno whether it worth  to use Eigen, brutal implementation for now
   if (Xtrain.size()!= 0 && Ytrain.size()!=0){
        int d=this->Xtrain[0].size(); // the dimention of data
        int n=this->Xtrain.size(); // num of data points

        for(int i=0; i<d; ++i){ //inital A0 with identity matrix
            std::vector<double> ARow(d, 0.0);
            ARow[i]=1;
            this->A.push_back(ARow);
        }

        std::vector<std::vector< double> > lambda;
        std::vector<std::vector< double> > xi;
        for(int i=0; i<n; ++i){ //inital lambda with zeros
            std::vector<double> LRow(n, 0.0);
            lambda.push_back(LRow);

            std::vector<double> XiRow(n);
            for(int j=0; j<n; ++j){
                if(this->Ytrain[i] == this->Ytrain[j]){ //similar pair
                    XiRow[j] = this->u;
                }else{ //dissimilar pairs
                    XiRow[j] = this->l;
                }
            }
            xi.push_back(XiRow);
        }

        int iter=0;
        while(iter++<MAX_ITER){
            for( int i=0; i< n; ++i ){

                for( int j=0; j< n; ++j ){
                    std::vector<double> diffX(d); //x_i-x_j
                    for(int dd=0; dd<d; ++dd){
                        diffX[dd]=this->Xtrain[i][dd]-this->Xtrain[j][dd];
                    }
                    //compute p = (xi-xj)A(xi-xj)
                    double p=0.0;
                    for(int r=0; r<d; ++r){
                        for(int c=0; c<d; ++c){
                            p+=diffX[r]*this->A[r][c]*diffX[c];
                            if(std::isnan(this->A[r][c])){
                                abort();
                            }
                        }
                    }
                    //std::cout<<"p:"<<p<<std::endl;
                    double delta;                 
                    if(this->Ytrain[i] == this->Ytrain[j]){ //similar pair
                        delta =1.0;
                    }else{ //dissimilar pairs
                        delta=-1.0;
                    }

                    //std::cout<<"delta:"<<delta<<std::endl;
                    double alpha = std::min(lambda[i][j],
                                     0.5*delta*(1/p - this->gamma/xi[i][j] ));
                    
                   // std::cout<<"alpha"<<alpha<<std::endl;
                    double beta = delta*alpha/(1-delta*alpha*p);

                    //std::cout<<"beta"<<beta<<std::endl;

                    xi[i][j] = this->gamma*xi[i][j]/(this->gamma + delta*alpha*xi[i][j]);

                    lambda[i][j] -=alpha;


                    std::vector<std::vector< double> > T; //A(xi-xj)(xi-xj)
                    for(int i=0; i<d; ++i){ //inital A0 with identity matrix
                        std::vector<double> TRow(d, 0.0);
                        T.push_back(TRow);
                    }

                    for(int r=0; r<d; ++r){
                        for(int c=0; c<d; ++c){
                            for(int t=0;t<d; ++t){
                                T[r][c]+= this->A[r][t]*diffX[t]*diffX[c];
                            }
                        }
                    }
                            
                    auto A_tmp= this->A;
                    for(int r=0; r<d; ++r){
                        for(int c=0; c<d; ++c){
                            for(int t=0;t<d; ++t){
                                this->A[r][c]+=beta* T[r][t]*A_tmp[t][c];
                            }
                        }
                    }

                }
            }
        }
    }else{
        std::cout<<"Error:Need to load data first"<<std::endl;
    }
}

void  InfoTheoreticMetricLearning::computeMahalanobisDis(){
    int N_train= getXtrainrow();
    int N_test = this->Xtest.size();
    int d= getXtraincol();
    this->D.clear(); //N_test * N_train
    //std::cout<<"N train:"<<N_train<< " N test:" <<N_test<<std::endl;
    for(int r=0; r<N_test; ++r){
        std::vector<double> DRow(N_train,0.0);
        for(int c=0; c< N_train; ++c){

            std::vector<double> diffX(d); //x_i-x_j
            for(int dd=0; dd<d; ++dd){
                diffX[dd]=this->Xtest[r][dd]-this->Xtrain[c][dd];
            }

            double dis=0.0;

            for(int r=0; r<d; ++r){
                for(int c=0; c<d; ++c){
                    dis+=diffX[r]*this->A[r][c]*diffX[c];
                }
            }
            
            DRow[c] = dis;
        }
        this->D.push_back(DRow);
    }

}
