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

