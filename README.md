# MNN: Modular Neural Net
MNN is a Neural Net construction library to allow for more fine grained control of inter-node connections and complex graphs. Currently MNN is planned to implement three degrees of control for Networks, with each higher level being able to be built from lower level pieces (and higher level pieces deconstructed into base components).

MNN is written and targetted for use in C\+\+1y and higher. It is not completed (or truly even started. Consider every claim in this document a lie until this project is completed)
## Nodes:
Nodes are the lowest level of control and will provide the basic functionality of a single cell in a Neural Net. Nodes come in three basic types:
* Input Node: A node whose value is set at construction and remains unchanged by Forward or Backpropogation
* Hidden Node: A traditional Hidden Node that can perform Forward and Backpropogation
* Output Node: A node whose value is compared to targets during training and performs the first step of Backpropogation

All Hidden Nodes can be directionally connected to any other Hidden Node. This explicitly allows for two or more nodes to be cyclically connected, or for a node to form a connection to itself. At runtime cycles will be detected and will behave similarly to a Recurrent Node in a RNN with a period equal to the depth of the cycle.
All Nodes can have their functionality extended through decorators, which provide various alterations to the output, input, or creation of a Node, including:
* Probablistic Dropout
* Recurrence
* Memory
* Spiking

## Layers:
Layers are the intermediate level of control, and act as collections of independant Nodes. Layers can be connected to either other Layers, or to individual Nodes. Actions on Layers operate the same as if they were a single Node (including the atomocity of their operations) and abstract away the actual content of itself from the rest of the program. 

Hidden Nodes within layers should not be inter-connected, but will usually not cause crashing in the event that they are. Cyclic connections in Layers WILL cause a crash and should not be done. Cyclic Connections **BETWEEN** Layers is supported.
