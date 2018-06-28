#ifndef NODE_WRAPPER_H
#define NODE_WRAPPER_H

#include <algorithm>
#include <memory>
#include "mnn_utilities.hpp"
#include "node_object.hpp"

namespace mnn {

using NodeObj_ptr = std::shared_ptr<NodeObj>;

class NodeWrapper : NodeObj {
   private:
    NodeObj_ptr wrappee;

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
   protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) override;
    virtual void request_forwardprop_handoff(NeuralObj_ptr &) override;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) override;

   public:
    NodeWrapper(NodeObj_ptr in_ptr) : wrappee(in_ptr), name("decorator") {}

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;
    virtual void calculate() override;
    virtual void update(double) override;
};

void NodeWrapper::recieve_backprop_handoff(NeuralObj_ptr &p, double d) {
    wrappee->recieve_backprop_handoff(p, d);
}
void NodeWrapper::request_forwardprop_handoff(NeuralObj_ptr &p) {
    wrappee->request_forwardprop_handoff(p);
}
void NodeWrapper::give_forwardprop_handoff(NeuralObj_ptr &p, double d) {
    wrappee->give_forwardprop_handoff(p, d);
}
void NodeWrapper::add_input(NeuralObj_ptr &p) { wrappee->add_input(p); }
void NodeWrapper::remove_input(NeuralObj_ptr &p) { wrappee->remove_input(p); }
void NodeWrapper::calculate() { wrappee->calculate(); }
void NodeWrapper::update(double d) { wrappee->update(d); }

}  // namespace mnn