#ifndef HIDDEN_NODE_H
#define HIDDEN_NODE_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "node.hpp"

namespace mnn {

template <typename Activator = Sigmoid>
class HiddenNode : Node {
   private:
    vector<double> dEn_dO;

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
     */

   public:
    /*
     * NeuralObj::done_calcuating : bool
     */
   private:
   protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) override;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) override;

    virtual double calculate_dE_dO() override;
    virtual double get_responce(NeuralObj_ptr &) override;

   public:
    HiddenNode() : name("hidden_") {}

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;
};

template <typename Activator>
void HiddenNode<Activator>::recieve_backprop_handoff(NeuralObj_ptr &n_ptr,
                                                     double r) {
    this.dEn_dO.push_back(r);
}

template <typename Activator>
void HiddenNode<Activator>::give_forwardprop_handoff(NeuralObj_ptr &n_ptr,
                                                     double r) {
    this.forward_handoff_map[n_ptr] = r;
}

template <typename Activator>
double HiddenNode<Activator>::calculate_dE_dO() {
    return std::accumulate(dEn_dO.begin(), dEn_dO.end(), 0.);
}

template <typename Activator>
double HiddenNode<Activator>::get_responce(NeuralObj_ptr &n_ptr) {
    assert(forward_handoff_map.find(n_ptr) != forward_handoff_map.end());
    return forward_handoff_map[n_ptr];
}

}  // namespace mnn

#endif
