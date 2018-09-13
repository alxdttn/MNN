#ifndef NODES_H
#define NODES_H 

#include <vector>
using namespace std;

typedef struct Net_Params {
    bool first_run = false;
    bool print_output = false;
} Net_Params;

extern Net_Params NET_PARAMS;

class Node {
private:
    double get_rand(double max){
        return static_cast<double>(rand()*max)/static_cast<double>(RAND_MAX);
    }
protected:
    vector<Node*> inputs;
    vector<double> weights;
    vector<bool> is_loop;
    double bias;
    double result;

    bool waiting;
    size_t waiting_on;
    double saved_result;
public:
    bool done_calculating;

public:
    Node() :
        bias(get_rand(0.5)),
        result(0.f),
        waiting(false),
        done_calculating(false)
        { }
    double get_weight(Node* node);
    void add_input(Node* node);
    double get_result();
    virtual void set_result(double input) { };
    virtual void calculate();
    virtual void update(double epsilon) = 0;
    virtual void give_backprop(double input) = 0;
};

class Output_Node : public Node {
private:
    double target;
public:
    void update(double epsilon) override;
    void give_backprop(double input) override; 
};

class Hidden_Node : public Node {
private:
    vector<double> dEn_dO;
public:
    void update(double epsilon) override;
    void give_backprop(double input) override;
};

class Input_Node : public Node {
private:
public:
    void update(double epsilon) override;
    void give_backprop(double input) override;
    void calculate() override;
    void set_result(double r);
};

#endif
