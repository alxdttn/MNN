#ifndef OUTPUT_NODE_H
#define OUTPUT_NODE_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "mnn_utilities.hpp"
#include "node_object.hpp"

namespace mnn {

template <typename Activator = Sigmoid>
class OutputNode : Node {
   private:
    double target;
   protected:
    /*
     * NeuralObj::name         :     string
     * NeuralObj::inputs       :     NeuralObj_ptr[]
     * NeuralObj::weights      :     double[]
     * NeuralObj::is_loop_flag :     bool[]
     * NeuralObj::is_waiting   :     bool
     * NeuralObj::waiting_on   :     int
     * NeuralObj::handoff      :     double
     * NeuralObj::forward_hand :    map<NeuralObj_ptr, double>
     *
     * NodeObj::bias           :     double
     * NodeObj::result         :     result
     * NodeObj::saved_result   :     saved_result
     *
     */
   public:
    /* NeuralObj::done_calcuating : bool */
   private:
   protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) override;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) override;

    virtual double calculate_dE_dO() override;
    virtual double get_responce(NeuralObj_ptr &) override;
   public:
    OutputNode() : name("output_") {}
};

template <typename Activator>
void OutputNode<Activator>::recieve_backprop_handoff(NeuralObj_ptr &n_ptr,
                                                     double r) {
    target = r;
}

template <typename Activator>
void OutputNode<Activator>::give_forwardprop_handoff(NeuralObj_ptr &n_ptr,
                                                     double r) {
    this.forward_handoff_map[n_ptr] = r;
}

template <typename Activator>
double OutputNode<Activator>::calculate_dE_dO() {
    return result - target;
}

template <typename Activator>
double OutputNode<Activator>::get_responce(NeuralObj_ptr &n_ptr) {
    assert(forward_handoff_map.find(n_ptr) != forward_handoff_map.end());
    return forward_handoff_map[n_ptr];
}

}