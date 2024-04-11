import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphNetworkVisualizer:
    def __init__(self, layers, weights, input_data):
        self.layers = layers
        self.input_data = self.prepend_bias(input_data)
        self.graph = nx.Graph()
        self.weights = [np.array(w) for w in weights]

    def prepend_bias(self, input):
        return np.insert(input, 0, 1, axis=1)
    
    def visualize(self):
        count = 1
        for each_input_data in self.input_data:
          plt.figure(figsize=(12,6.75))
          G = self.graph
          previous_layer_nodes = []
          current_node_id = 0

          # pos = {}
          pos = nx.spring_layout(G)
          # Reset layer_nodes each iteration
          previous_layer_nodes = [] 
          current_layer_nodes = []

          for input_index, input_value in enumerate(each_input_data):
              # Bias value is orange-colored, skyblue-color for the rest of input
              color = 'orange' if input_index == 0 else 'skyblue'
              G.add_node(current_node_id, label=f'{input_value}', value=input_value, color=color)
              pos[current_node_id] = (0, -input_index * 2)
              previous_layer_nodes.append(current_node_id)
              current_node_id += 1

          for layer_index, (_, weight_matrix) in enumerate(zip(self.layers, self.weights)):
              current_layer_nodes = []
              current_node_id += 1
              for neuron_index in range(weight_matrix.shape[1]):
                  G.add_node(current_node_id, label=f'L{layer_index+1}N{neuron_index+1}', color='lightgreen')
                  pos[current_node_id] = (layer_index + 1, -neuron_index * 2)
                  current_layer_nodes.append(current_node_id)

                  for prev_node_id in previous_layer_nodes:
                      weight = weight_matrix[previous_layer_nodes.index(prev_node_id), neuron_index]
                      # print(f"Index: {prev_node_id}, neuron index: {neuron_index}, weight: {weight}")
                      G.add_edge(prev_node_id, current_node_id, weight=float(weight))

                  current_node_id += 1

              previous_layer_nodes = current_layer_nodes

          output_node_id = current_node_id
          G.add_node(output_node_id, label=f'Output', color='red')
          pos[output_node_id] = (len(self.layers) + 1, 0) 

          for prev_node_id in previous_layer_nodes:
              G.add_edge(prev_node_id, output_node_id)

          edge_labels = nx.get_edge_attributes(G, 'weight')
          colors = list(nx.get_node_attributes(G, 'color').values())
          labels = nx.get_node_attributes(G, 'label')

          nx.draw(G, pos,  labels=labels, with_labels=True, node_size=1200, node_color=colors, font_size=10, font_weight='bold')
          nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.8, font_size=8)
          G.clear
          plt.title("FFNN Visualization Batch "+str(count))
          plt.show()
          plt.clf()
          plt.close("all")
          count += 1
