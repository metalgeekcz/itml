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
#include <vector>
struct index_cmp
{
    index_cmp(const std::vector<double> &pArr)
        :arr(pArr)
    {}

    bool operator()(const size_t a, const size_t b) const
    {
        return arr[a] < arr[b];
    }

    const std::vector<double> &arr;
};
void KNN(const std::vector<std::vector<double> > & DistanceMat, const std::vector<double> & trainY, std::vector<double> & EstRes, int K);

