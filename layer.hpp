#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <vector>
#include "neural_object.hpp"

namespace mnn {

class Layer : NeuralObj {
   private:
    vector<NeuralObj_ptr> children;
    class {  // Interfacer Anonymous Class
        /* TODO */
    } input, output;

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
    NeuralObj() : is_waiting(false), waiting_on(0), done_calculating(false) {}

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;

    virtual void calculate() override;
    virtual void update(double) override;
};

void Layer::recieve_backprop_handoff(NeuralObj_ptr &, double){
    return; 
}
void Layer::request_forwardprop_handoff(NeuralObj_ptr &){
    return; 
}
void Layer::give_forwardprop_handoff(NeuralObj_ptr &, double){
    return; 
}
void Layer::add_input(NeuralObj_ptr &){
    return; 
}
void Layer::remove_input(NeuralObj_ptr &){
    return; 
}
void Layer::calculate(){
    return; 
}
void Layer::update(double){
    return; 
}

}  // namespace mnn