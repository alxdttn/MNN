#ifndef NODE_TYPES_H
#define NODE_TYPES_H

#include "mnn_utilities.hpp"
#include "node.hpp"

namespace mnn {

template <typename Activator = Sigmoid>
class InputNode : Node<Activator> {
   private:
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
    /* NeuralObj::done_calcuating : bool */
   private:
   protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) override;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) override;

    virtual double calculate_dE_dO() override;
    virtual double get_responce(NeuralObj_ptr &) override { return -1.; };

   public:
    InputNode() : name("input_") {}
    InputNode(double r) : name("input_"), result(r) {}
    void set_input(double r);

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;

    virtual void calculate() override;
    virtual void update(double) override;
};

template <typename Activator>
void InputNode<Activator>::recieve_backprop_handoff(NeuralObj_ptr &, double) {
    return;
}

template <typename Activator>
void InputNode<Activator>::give_forwardprop_handoff(NeuralObj_ptr &, double) {
    throw MNNException("Cannot forward propogate into InputNode");
}

template <typename Activator>
void InputNode<Activator>::set_input(double r) {
    result = r;
}

template <typename Activator>
void InputNode<Activator>::add_input(NeuralObj_ptr &) {
    throw MNNException("Cannot add further inputs to InputNode");
}

template <typename Activator>
void InputNode<Activator>::remove_input(NeuralObj_ptr &) {
    throw MNNException("InputNode has no further inputs to remove");
}

template <typename Activator>
void InputNode<Activator>::calculate() {}

template <typename Activator>
void InputNode<Activator>::update(double) {}

template <typename Activator>
double InputNode<Activator>::calculate_dE_dO() {
    return -1.;
}

}  // namespace mnn
