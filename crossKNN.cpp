/*
ITML C++
Cheng Zhang, chengz@kth.se 

(C) Copyright 2015, Cheng Zhang 

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License.

The GNU General Public License does not permit this software to be redistributed in proprietary programs.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/
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
