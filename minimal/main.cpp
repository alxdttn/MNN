#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "nodal_net.h"
#include "nodes.h"
using namespace std;

vector<vector<double>> read_CSV(istream& str){
    vector<vector<double>> table;
    string line;
    while(getline(str, line)){
        stringstream lineStream(line);
        string cell;

        vector<double> values;
        while(getline(lineStream, cell, ',')){
            values.push_back(stod(cell));
        }
        table.push_back(values);
    }
    return table;
}

vector<vector<double>> extract_y(vector<vector<double>>& table, size_t size){
    vector<vector<double>> y;
    for (auto& row: table){
        vector<double> y_row;
        for (auto i = 0; i < size; ++i){
            y_row.push_back(row.back());
            row.pop_back();
        }
        y.push_back(y_row);
    }
    return y;
}

int main(){
    Net net(8, 1);
    Node* node1 = new Hidden_Node();
    Node* node2 = new Hidden_Node();
    Node* node3 = new Hidden_Node();
    Node* node4 = new Hidden_Node();
    Node* node5 = new Hidden_Node();
    Node* node6 = new Hidden_Node();
    Node* node7 = new Hidden_Node();
    Node* node8 = new Hidden_Node();
    Node* node9 = new Hidden_Node();

    net.add_node(node1);
    net.add_node(node2);
    net.add_node(node3);
    net.add_node(node4);
    net.add_node(node5);
    net.add_node(node6);
    net.add_node(node7);
    net.add_node(node8);
    net.add_node(node9);

    net.make_input_reciever(node1);
    net.make_input_reciever(node2);
    net.make_input_reciever(node3);

    net.make_output_giver(node7);
    net.make_output_giver(node8);
    net.make_output_giver(node9);

    net.connect_nodes(node1, node2);
    net.connect_nodes(node1, node3);
    net.connect_nodes(node1, node4);

    net.connect_nodes(node2, node3);
    net.connect_nodes(node3, node2);
    net.connect_nodes(node2, node4);
    net.connect_nodes(node2, node5);
    net.connect_nodes(node3, node4);
    net.connect_nodes(node3, node6);

    net.connect_nodes(node4, node5);
    net.connect_nodes(node4, node6);
    net.connect_nodes(node4, node7);
    net.connect_nodes(node4, node8);
    net.connect_nodes(node4, node9);

    net.connect_nodes(node5, node6);
    net.connect_nodes(node5, node8);
    net.connect_nodes(node6, node5);
    net.connect_nodes(node6, node9);

    net.connect_nodes(node8, node9);
    net.connect_nodes(node9, node8);

    net.print();

    ifstream file("pima-indians-diabetes.csv");
    vector<vector<double>> x = read_CSV(file);
    vector<vector<double>> y = extract_y(x, 1);

    net.train(x, y, 0.001, 50000);

    net.print();

    return 0;
}
