#ifndef NODE_OBJ_H
#define NODE_OBJ_H

#include "neural_object.hpp"
#include "mnn_utilities.hpp"

namespace mnn
{

class NodeObj : public NeuralObj
{
  private:
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
    double bias;
    double result;

    double saved_result;

  public:
  private:
  protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) = 0;
    virtual void request_forwardprop_handoff(NeuralObj_ptr &) = 0;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) = 0;

  public:
    NodeObj() : bias(get_rand_weight(0., 0.5)),
                result(0)
    {
    }

    virtual void add_input(NeuralObj_ptr &) = 0;
    virtual void remove_input(NeuralObj_ptr &) = 0;

    virtual void calculate() = 0;
    virtual void update(double) = 0;
};

} // namespace mnn
