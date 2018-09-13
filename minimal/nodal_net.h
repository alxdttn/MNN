#ifndef NODAL_NET_H
#define NODAL_NET_H

#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;
using Matrix = vector<vector<double>>;

class Node;

class Net {
private:
    vector<Node*> nodes;
    size_t input_size;
    size_t output_size;
    size_t n_hidden_nodes;

    double epsilon;

public:
    Net(size_t x_size, size_t y_size);
    void add_node(Node* n);
    void connect_nodes(Node* giver, Node* reciever);
    void make_input_reciever(Node* node);
    void make_output_giver(Node* node);

    void train(const Matrix& x, const Matrix& y, double epsilon, size_t epochs);
    vector<double> predict(const vector<double>& x);

    void print();
};

#endif
