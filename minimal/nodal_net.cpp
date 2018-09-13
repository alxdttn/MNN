#include <cassert>
#include <ctime>
#include <typeinfo>
#include <iomanip>
#include <iostream>
#include "nodal_net.h"
#include "nodes.h"
using namespace std;


Net::Net(size_t x_size, size_t y_size) :
input_size(x_size), output_size(y_size), n_hidden_nodes(0){
    for (size_t i = 0; i < input_size; ++i){
        nodes.push_back(new Input_Node());
    }
    for (size_t i = 0; i < output_size; ++i){
        nodes.push_back(new Output_Node());
    }
}

void Net::add_node(Node* n){
    nodes.insert(nodes.begin()+input_size+n_hidden_nodes, n);
    ++n_hidden_nodes;
}

void Net::connect_nodes(Node* giver, Node* reciever){
    assert(typeid(*giver) != typeid(Output_Node));
    assert(typeid(*reciever) != typeid(Input_Node));
    assert(reciever->get_weight(giver) == 0.);
    reciever->add_input(giver);
}

void Net::make_input_reciever(Node* node){
    assert(typeid(*node) != typeid(Input_Node));
    auto it = nodes.begin();
    for_each(it, it+input_size, [node](auto& i){
            node->add_input(i);
        });
}

void Net::make_output_giver(Node* node){
    assert(typeid(*node) != typeid(Output_Node));
    auto it = nodes.begin()+nodes.size()-output_size;
    for_each(it, nodes.end(), [node](auto& i){
        i->add_input(node);
    });
}

void Net::train(const Matrix& x, const Matrix& y, double epsilon, size_t epochs){
    assert(x.size() == y.size());

    double total_error = 1.;
    NET_PARAMS.first_run = true;
    clock_t begin = clock();
    clock_t end;
    for(size_t i = 0; i < epochs; ++i){
        for (size_t j = 0; j < x.size(); ++j){
            vector<double> x_j = x[j];
            vector<double> y_j = y[j];

            assert(x_j.size() == input_size);
            assert(y_j.size() == output_size);

            for (size_t k = 0; k < input_size; ++k){
                nodes[k]->set_result(x_j[k]);
            }

            //Forward Prop
            for (auto& node : nodes){
                //occasionally in cyclic graphs this can occur
                if (node->done_calculating) continue;
                node->calculate();
            }

            total_error = 0.;
            for (size_t k = 0; k < output_size; ++k){
                double output = nodes[nodes.size()-output_size+k]->get_result();
                double error = 0.5*((y_j[k]-output)*(y_j[k]-output));
                nodes[nodes.size()-output_size+k]->give_backprop(error);
                total_error += error;
            }

            //Backwards Prop
            for (auto it = nodes.rbegin(); it != nodes.rend(); ++it){
                (*it)->update(epsilon);
            }

            NET_PARAMS.first_run = false;
        }
        if (i%1000 == 0){
            end = clock();
            cout << "Epoch: " << i
                 << "\tError: " << total_error
                 << "\tTime Elapsed: " << double(end - begin)/ CLOCKS_PER_SEC << endl;
            begin = clock();
        }
        if (total_error < 0.001) break;
    }
}




vector<double> Net::predict(const vector<double>& x){
    assert(x.size() == input_size);

    for (size_t i = 0; i < input_size; ++i){
        nodes[i]->set_result(x[i]);
    }

    //Run twice to ensure full calculations of the graph
    for (size_t i = 0; i < 2; ++i){
        for (auto& node: nodes){
            node->calculate();
        }
    }

    vector<double> output;
    for (size_t i = 0; i < output_size; ++i){
        output.push_back(nodes[nodes.size()-output_size+i]->get_result());
    }

    return output;

}

void Net::print(){
    vector<string> names;

    cout << "*\t";
    for (size_t i = 0; i < input_size; ++i){
        names.push_back("i"+to_string(i+1));
        cout << "i" << i+1 << "\t";
    }
    for (size_t i = 0; i < n_hidden_nodes; ++i){
        names.push_back("h"+to_string(i+1));
        cout << "h" << i+1 << "\t";
    }
    for (size_t i = 0; i < output_size; ++i){
        names.push_back("o"+to_string(i+1));
        cout << "o" << i+1 << "\t";
    }
    cout << "\n";
    cout << setprecision(3);

    for (size_t i = input_size; i < nodes.size(); ++i){
        cout << names[i] + "\t";
        Node* node_ptr = nodes[i];
        for (size_t j = 0; j < nodes.size(); ++j){
            if (i == j) cout << "*\t";
            else cout << node_ptr->get_weight(nodes[j]) << "\t";
        }
        cout << "\n";
    }
}
