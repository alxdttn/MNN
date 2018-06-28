#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <vector>
#include "neural_object.hpp"

namespace mnn {

class Layer : NeuralObj {
   private:
    vector<NeuralObj_ptr> children;

   protected:
    /*
     * NeuralObj::name         :    string
     * NeuralObj::inputs       :    NeuralObj_ptr&[]
     * NeuralObj::weights      :    double[]
     * NeuralObj::is_loop_flag :    bool[]
     * NeuralObj::is_waiting   :    bool
     * NeuralObj::waiting_on   :    int
     * NeuralObj::forward_hand :    map<NeuralObj_ptr, double>
     */
   public:
   private:
   protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) override;
    virtual void request_forwardprop_handoff(NeuralObj_ptr &) override;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) override;

    /*Eventually make this thread-safe*/
    double get_forwardprop_responce(NeuralObj_ptr &n) {
        assert(forward_handoff_map.find(n) != forward_handoff_map.end());
        return forward_handoff_map[n];
    }

   public:
    Layer() : {}

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;

    virtual void calculate() override;
    virtual void update(double) override;
};

void Layer::recieve_backprop_handoff(NeuralObj_ptr &n_ptr, double r) { 
    vector<double> node_multiplier = node_map[n_ptr];
    int i = 0;
    for_each(children.begin(), children.end(), [&](auto n){
        n->recieve_backprop_handoff(shared_from_this(), r*node_multiplier[i]);
        ++i;
    }
 }
 
void Layer::request_forwardprop_handoff(NeuralObj_ptr &) { return; }
void Layer::give_forwardprop_handoff(NeuralObj_ptr &, double) { return; }
void Layer::add_input(NeuralObj_ptr &) { return; }
void Layer::remove_input(NeuralObj_ptr &) { return; }
void Layer::calculate() { return; }
void Layer::update(double) { return; }

}  // namespace mnn