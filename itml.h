/*
ITML C++
Cheng Zhang, chengz@kth.se 

(C) Copyright 2015, Cheng Zhang 

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License.

The GNU General Public License does not permit this software to be redistributed in proprietary programs.

This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/

#ifndef ITMLHEADERFILE_H
#define ITMLHEADERFILE_H 
#include <iostream>
#include <vector>

#define MAX_ITER  5

class InfoTheoreticMetricLearning{
    private:
    std::vector<std::vector<double> > X;
	std::vector<double> Y; // this can be label or distance
    std::vector<std::vector<double> > Xtrain;
	std::vector<double> Ytrain; 
    std::vector<std::vector<double> > Xtest;
	std::vector<double> YgroundTruth; 
	double u;
	double l;
	double gamma;
    std::vector<std::vector<double> > A; // the learned matric
    std::vector<std::vector<double> > D;

	public:
	void setU(double pU);
	void setL(double pL);
    std::vector<std::vector<double> > getXtrain();
	std::vector<double> getYtrain(); 
    std::vector<std::vector<double> > getXtest();
	std::vector<double> getYgroundTruth(); 
    std::vector<std::vector<double> > getA();
    std::vector<std::vector<double> > getD();
    int  getXtrainrow();
    int  getXtraincol();
    int  getYtrainrow();
    int  getXtestrow();
    double Xtrainat(int r, int c);
    double Ytrainat(int r);
	void setGamma(double pGamma);
	void loadDatafromFile(std::string filenameX, std::string filenameY);
	void setNfoldTrainingdata(const int f,const int N_fold);
    void metricLearning();
	void metricLearningOnline();
    void computeMahalanobisDis();

};
#endif /* ITMLHEADERFILE_H */
