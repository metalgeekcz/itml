#include <map>
#include <vector>
#include <algorithm>
#include "crossKNN.h"


void KNN(const std::vector<std::vector<double> > & DistanceMat, const std::vector<double> & trainY, std::vector<double> & EstRes, const int K){
    
    int N_test = DistanceMat.size(); // num of test data
    int N_train = DistanceMat[0].size(); // num of training data

    std::vector<unsigned> ind_score;
    for(unsigned ii=0; ii<N_train; ++ii){
        ind_score.push_back(ii);
    }

    for(int n=0; n< N_test; ++n){
        if(n>0){
            std::sort(ind_score.begin(), ind_score.end());
        }

        std::sort(ind_score.begin(), ind_score.end(), index_cmp(DistanceMat[n]));

        std::map<double, int> topKlabel;
        
        for(int k=0; k<K; ++k){
            auto it= topKlabel.find(trainY[ind_score[k]]);
            if(it == topKlabel.end()){
            topKlabel.insert(std::pair<double, int>(trainY[ind_score[k]], 1 ));
            }else{
                it->second ++;
            }
        }

        //find the max vote
        double est;
        int maxCount=0;
        for(auto it=topKlabel.begin(); it!=topKlabel.end(); ++it){
            if(it->second > maxCount){
                est=it->first;
            }
        }

        EstRes.push_back(est);

    }

}
