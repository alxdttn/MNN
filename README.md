# MNN: Modular Neural Net

- [About](#About)
    - [Nodes]
## About
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

As layers are treated exactly the same as individual Nodes, Layers of Layers are supported, but frankly not particularly useful, as they do not represent an interconnected mesh of layers like a traditional neural net, but rather a series of completely independant layers (remembering that Nodes within layers should not be inter-connected).

Creations of layers pre-initialized with Nodes of a particular type can be done through the LayerBuilder, which handles the creation through an interface. You can also create layers yourself by manually adding individual nodes. When layers are connected to outside Nodes (or other layers), to prevent external objects from needing to know the contents of the layer, it will create an interface such that when feedforward occurs, it maintains an internal map of weights to each node to each external connection which updates every pass and is given to requesting external nodes when needed. This has a decently steep memory cost in the domain of many fully-connected layers, but it simplifies the model in terms of complexity of data-flow and - because the connections are stored contiguously instead of across memory - can be slightly faster than direct querying due to cacheing.

## Nets:
Nets are the structures normally associated with a Neural Net. They are a collection of Nodes, Layers, and Inputs and Outputs that can be trained and used to predict results. However, Nets, just like layers and Nodes, can ALSO be made to behave like Nodes on a a super Net. This allows for some wild behavior like chaining two completely different models, and using a probablistic node to send data randomly to one or the other, or for independantly training two nets at the same time and then combining them during runtime for prediction. 

For ease of access, there are a number of pre-built net options that can be made using the NetBuilder class and supplying the number of inputs, number of outputs, type, and a few other parameters. This speeds up prototyping and manually connecting all of the objects yourself.

# Structure:
<img src="/Class_UML.svg" width="100%">

# Compiling:
TODO

# Running:
TODO

# Examples:
TODO
