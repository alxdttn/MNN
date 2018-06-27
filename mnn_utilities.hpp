#ifndef MNN_UTILITIES_H
#define MNN_UTILITIES_H

#include <cmath>

namespace mnn {

/* Common Activation Functions */
class Sigmoid {
public:
    constexpr static double f(double x){
        return 1./(1.+exp(-1.*x));
    }
    constexpr static double Df(double x){
        return f(x)*(1-f(x));
    }
    constexpr static double Df_f(double x){
        return x*(1.-x);
    }
};

class ReLU {
public:
    constexpr static double f(double x){
        return (x > 0) ? x : 0;
    }
    constexpr static double Df(double x){
        return (x > 0) ? 1 : 0;
    }
    constexpr static double Df_f(double x){
        return (x > 0) ? 1 : 0;
    }
};

class Tanh {
public:
    constexpr static double f(double x){
        return 2./(1+exp(-2*x)) - 1;
    }
    constexpr static double Df(double x){
        return pow((1+f(x)), 2)*exp(-2.*x);
    }
    constexpr static double Df_f(double x){
        return 1.-pow(x, 2);
    }
};

class Linear {
public:
    constexpr static double f(double x){
        return x;
    }
    constexpr static double Df(double x){
        return 1;
    }
    constexpr static double Df_f(double x){
        return 1;
    }
};

class MNNException: public std::runtime_error
{
    MNNException() : std::runtime_error("MNNException") { }
    MNNException(string str) :
        std::runtime_error("MNNException: " + str) { }
};

double get_rand_weight(double min = 0., max = 1.){
    return static_cast<double>(rand()*(max-min))/static_cast<double>(RAND_MAX);
}

}

#endif
