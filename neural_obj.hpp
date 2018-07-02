#ifndef NEURAL_OBJ_H
#define NEURAL_OBJ_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tracked_object.hpp"

namespace mnn {

class NeuralObj;
using NeuralObj_ptr = std::shared_ptr<NeuralObj>;

class NeuralObj : CounterObj<NeuralObj>,
                  std::enable_shared_from_this<NeuralObj> {
    /* Member Variables */
   private:
   protected:
    std::string name;
    std::vector<NeuralObj_ptr> inputs;
    std::vector<double> weights;
    std::vector<bool> is_loop_flag;

    bool is_waiting;
    int waiting_on;

    std::unordered_map<NeuralObj_ptr, double> forward_handoff_map;

   public:
    bool done_calculating;

    /* Member Methods */
   private:
   protected:
    virtual void recieve_backprop_handoff(NeuralObj_ptr &, double) = 0;

    virtual void request_forwardprop_handoff(NeuralObj_ptr &) = 0;
    virtual void give_forwardprop_handoff(NeuralObj_ptr &, double) = 0;

    /*Eventually make this thread-safe*/
    double get_forwardprop_responce(NeuralObj_ptr &n) {
        assert(forward_handoff_map.find(n) != forward_handoff_map.end());
        return forward_handoff_map[n];
    }

    //Called at the end of add_input on the added object, passing shared_ptr<this>
    virtual void connect(NeuralObj_ptr &) = 0;

   public:
    NeuralObj() : is_waiting(false), waiting_on(0), done_calculating(false) {}

    virtual void add_input(NeuralObj_ptr &) = 0;
    virtual void remove_input(NeuralObj_ptr &) = 0;

    virtual void calculate() = 0;
    virtual void update(double) = 0;
};

}  // namespace mnn

#endif
