#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "nodes.h"
using namespace std;

Net_Params NET_PARAMS{false, false};

double Node::get_weight(Node* node){
    auto it = find(inputs.begin(), inputs.end(), node);
    if (it != inputs.end()){
        size_t index = distance(inputs.begin(), it);
        return weights[index];
    }
    return 0.;
}

void Node::add_input(Node* node){
    inputs.push_back(node);
    weights.push_back(get_rand(0.25));
}

double Node::get_result(){
    return result;
}

void Node::calculate(){
    //For cycle detection to resume and skip.
    //If this isn't the first run, it simply takes
    //the old value of the problematic connection
    //and uses that. If it is the first run, it just
    //ignores the connection and continues without it.
    //This causes some error during back-propagation
    //where weight is adjusted regardless of the fact
    //that the end result was not calculated, but I 
    //believe this to be acceptable in the domain of
    //large epochs.
    //Cycles otherwise behave like a Recurrent Cell
    size_t i;
    if (!waiting){
        i = 0;
        result = 0;
    }
    else {
        i = waiting_on + 1;
        result = saved_result;
        waiting = false;
        if (!NET_PARAMS.first_run){
            result += inputs[i-1]->get_result()*weights[i];
        }
    }

    for(; i < inputs.size(); ++i){
        if (!inputs[i]->done_calculating){
            waiting = true;
            waiting_on = i;
            saved_result = result;
            inputs[i]->calculate();
        }
        result += inputs[i]->get_result()*weights[i];
    }
    result += bias;
    result = 1./(1.+exp(-1.*result));
    done_calculating = true;
}

void Output_Node::update(double epsilon)  {
    double dE_dO = result - target;
    double dO_dN = result*(1.0 - result);
    for (size_t i = 0; i < inputs.size(); ++i){
        double dN_dWi = inputs[i]->get_result();
        double delta  = dE_dO*dO_dN;
        inputs[i]->give_backprop(delta*weights[i]);
        weights[i] -= epsilon*delta*dN_dWi;
    }
    done_calculating = false;
}

void Output_Node::give_backprop(double input)  {
    target = input;
}

void Hidden_Node::update(double epsilon) {
    double dE_dO = accumulate(dEn_dO.begin(), dEn_dO.end(), 0.);
    dEn_dO.clear();
    double dO_dN = result*(1.-result);
    for (size_t i = 0; i < inputs.size(); ++i){
        double dN_dWi = inputs[i]->get_result();
        double delta = dE_dO*dO_dN;
        inputs[i]->give_backprop(delta*weights[i]);
        weights[i] -= epsilon*delta*dN_dWi;
    }
    done_calculating = false;
}

void Hidden_Node::give_backprop(double input)  {
    dEn_dO.push_back(input);

}

void Input_Node::update(double epsilon)  { }

void Input_Node::give_backprop(double input)  { }

void Input_Node::calculate(){ }

void Input_Node::set_result(double r) { result = r; done_calculating = true; }
