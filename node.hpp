#ifndef NODE_H
#define NODE_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "node.hpp"
#include "node_object.hpp"
#include "mnn_utilities.hpp"

namespace mnn
{

template <typename Activator = Sigmoid>
class Node : NodeObj
{
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
        *
        * NodeObj::bias           :     double
        * NodeObj::result         :     result
        * NodeObj::saved_result   :     saved_result
        */

  public:
    /* NeuralObj::done_calcuating : bool */
  private:
  protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) = 0;

    virtual void request_forwardprop_handoff(NeuralObj_ptr &) override;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) = 0;

    virtual double calculate_dE_dO() = 0;

  public:
    Node()
    {
        name += "node" + to_string(objects_made);
    }

    virtual void add_input(NeuralObj_ptr &) override;
    virtual void remove_input(NeuralObj_ptr &) override;

    virtual void calculate() override;
    virtual void update(double) override;
};

template <typename Activator = Sigmoid>
void Node<Activator>::request_forwardprop_handoff(NeuralObj_ptr n_ptr)
{
    n_ptr->give_forwardprop_handoff(this.result);
}

template <typename Activator = Sigmoid>
void Node<Activator>::add_input(NeuralObj_ptr node)
{
    assert(std::find(inputs.begin(), inputs.end(), node) == inputs.end());
    inputs.push_back(node);
    weights.push_back(get_rand_weight(0, 0.5));
    is_loop_flag.push_back(false);
}

template <typename Activator = Sigmoid>
void Node<Activator>::remove_input(NeuralObj_ptr node)
{
    auto it = std::find(inputs.begin(), inputs.end(), node);
    assert(it != inputs.end());
    int i = std::distance(inputs.begin(), it);
    inputs.erase(inputs.begin() + i);
    weights.erase(weights.begin() + i);
}

/* During Forward Propogation, there are times when one object
 * may not yet be began/finished calculating before being requested by
 * another, or there could even be a cyclic dependancy in the case
 * of Recurrent Nodes or Systems. This function handles such by
 * calling the stalled process' calculate recursively.
 * If this ends up finding a process that is already waiting, it 
 * decides whether to ignore the dependancy, or whether to use an old value
 * and continue on. This behaviour is consistent with Recurrent Cells.
 */
template <typename Activator = Sigmoid>
void Node<Activator>::calculate()
{
    size_t i;
    if (!waiting)
    {
        i = 0;
        result = 0;
    }
    else
    {
        is_loop_flag[waiting_on] = true;
        i = waiting_on;
        result = saved_result;
        if (!NET_PARAMS.first_run)
        {
            inputs[i]->request_forwardprop_handoff(shared_from_this());
            result += get_responce(inputs[i]) * weights[i];
        }
        ++i;
    }
    for (; i < inputs.size(); ++i)
    {
        if (!inputs[i]->done_calculating && !is_loop_flag[i])
        {
            waiting = true;
            waiting_on = i;
            saved_result = result;
            inputs[i]->calculate();
            waiting = false;
        }
        inputs[i]->request_forwardprop_handoff(shared_from_this());
        result += get_responce(inputs[i]) * weights[i];
    }
    result += bias;
    result = Activator::f(result);
    done_calculating = true;
}

template <typename Activator = Sigmoid>
void Node<Activator>::update(double epsilon)
{
    double dE_dO = calculate_dE_dO();

    double dO_dN = Activator::Df_f(result);
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inputs[i]->request_forwardprop_handoff(shared_from_this());
        double dN_dWi = get_responce(inputs[i]);
        double delta = dE_dO * dO_dN;
        inputs[i]->recieve_backprop_handoff(delta * weights[i]);
        weights[i] -= epsilon * delta * dN_dWi;
    }
    done_calculating = false;
}

} // namespace mnn
