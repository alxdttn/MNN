#ifndef NODE_TYPES_H
#define NODE_TYPES_H

#include "node.hpp"
#include "mnn_utilities.hpp"

namespace mnn
{

template <typename Activator = Sigmoid>
class InputNode : Node<Activator>
{
private:
protected:
    void recieve_backprop_handoff(NeuralObj_ptr&, double) final override;
    void give_forwardprop_handoff(NeuralObj_ptr&, double) final override;
    double calculate_dE_dO() final override;
public:
    InputNode() : name("input_") { }
    InputNode(double r) : name("input_"), result(r) { }
    void set_input(double r);

    virtual void add_input(NeuralObj_ptr&) override;
    virtual void remove_input(NeuralObj_ptr&) override;

    virtual void calculate() override;
    virtual void update(double) override;
};

template <typename Activator>
void InputNode<Activator>::recieve_backprop_handoff(NeuralObj_ptr&, double){
    return;
}

template <typename Activator>
void InputNode<Activator>::give_forwardprop_handoff(NeuralObj_ptr&, double){
    throw MNNException("Cannot forward propogate into InputNode");
}

template <typename Activator>
void InputNode<Activator>::set_input(double r){ result = r; }

template <typename Activator>
void InputNode<Activator>::add_input(NeuralObj_ptr&){
    throw MNNException("Cannot add further inputs to InputNode");
}

template <typename Activator>
void InputNode<Activator>::remove_input(NeuralObj_ptr&){
    throw MNNException("InputNode has no further inputs to remove");
}

template <typename Activator>
void InputNode<Activator>::calculate(){ }

template <typename Activator>
void InputNode<Activator>::update(double){ }

template <typename Activator>
double InputNode<Activator>::calculate_dE_dO(){ return -1.; }

} // namespace mnn

