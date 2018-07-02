#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <vector>
#include "neural_object.hpp"

namespace mnn {

class Layer : NeuralObj {
   private:
    std::vector<NeuralObj_ptr> children;
    std::unordered_map<NeuralObj_ptr, std::vector<double>> node_map;

    /* Anonymous class to allow for hiding internal nodes from external nodes
     * on the input side. It only does one thing, and that is aggregate
     * forward propogation responces and distribute them to the internal
     * nodes when requested.
     * This is the sole input to internal nodes
     */
    class : NeuralObj {
       private:
        void recieve_backprop_handoff(NeuralObj_ptr &, double) { };
        void give_forwardprop_handoff(NeuralObj_ptr &, double) { };
        void add_input(NeuralObj_ptr &) { };
        void remove_input(NeuralObj_ptr &) { };
        void calculate() { };
        void update(double) { };
        void connect(NeuralObj_ptr &) { };
       protected:
        void request_forwardprop_handoff(NeuralObj_ptr &n_ptr) {
            n_ptr->give_forwardprop_handoff(shared_from_this(),
                std::accumulate(inputs_to_layer.begin(), inputs_to_layer.end(), 0.);
        }
       public:
        std::vector<double> inputs_to_layer;
    } input_interfacer;

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

   public:
    Layer() : {}

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;

    virtual void calculate() override;
    virtual void update(double) override;

    virtual void connect(NeuralObj_ptr &) override;
};

void Layer::recieve_backprop_handoff(NeuralObj_ptr &n_ptr, double r) { 
    std::vector<double> node_multiplier = node_map[n_ptr];
    int i = 0;
    std::for_each(children.begin(), children.end(), [&](auto n){
        n->recieve_backprop_handoff(shared_from_this(), r*node_multiplier[i]);
        ++i;
    }
 }
 
void Layer::request_forwardprop_handoff(NeuralObj_ptr &n_ptr) { 
    std::vector<double> node_multiplier = node_map[n_ptr];
    double base = n_ptr->weights[
        std::distance(n_ptr->inputs.begin(),
            std::find(n_ptr->inputs.begin(),
                      n_ptr->inputs.end(),
                      shared_from_this()
            )
        )
    ];

    std::vector<double> scaled;
    scaled.reserve(node_multiplier.size());
    std::transform(node_multiplier.begin(), node_multiplier.end(),
                   children.begin(), std::back_inserter(scaled),
                   [&base](double multi, NeuralObj_ptr& node){
                       node->request_forwardprop_handoff(shared_from_this());
                       double value = get_responce(node);
                       return multi * value * (1./base);
                   });
    n_ptr->give_forwardprop_handoff(shared_from_this(),
            std::accumulate(scaled.begin(), scaled.end(), 0.));
}

void Layer::give_forwardprop_handoff(NeuralObj_ptr &n_ptr, double r) {
    /*
    std::for_each(children.begin(), children.end(), [](auto child){
        child->give_forwardprop_handoff(shared_from_this(), r);
    });
    */
    input_interfacer.inputs_to_layer.push_back(r);
}

void Layer::add_input(NeuralObj_ptr &n_ptr) { 
    assert(std::find(inputs.begin(), inputs.end(), node) == inputs.end());
    inputs.push_back(n_ptr);
    weights.push_back(get_rand_weight(0, 0.5));
    is_loop_flag.push_back(false);
    node->connect(std::shared_from_this());
}

void Layer::remove_input(NeuralObj_ptr &n_ptr) {
    auto it = std::find(inputs.begin(), inputs.end(), node);
    assert(it != inputs.end());
    int i = std::distance(inputs.begin(), it);
    inputs.erase(inputs.begin() + i);
    weights.erase(weights.begin() + i);
}

void Layer::calculate() { 
    for_each(children.begin(), children.end(), mem_fun(&NueralObj::calculate));
}

void Layer::update(double r) {
    for_each(children.begin(), children.end(), std::bind2nd(mem_fun_ref(&NeuralObj::update), r));
    input_interface.inputs_to_layer.clear();
}

void Layer::connect(NeuralObj_ptr &n_ptr) {
    node_multiplier[n_ptr] = get_rand_weight(0.5, 1.);
}

}  // namespace mnn

#endif
