import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arrow

def create_neuron_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define layer positions and neuron counts
    layers = [
        {"name": "Input\nSpectrogram", "neurons": 5, "x": 1, "type": "input"},
        {"name": "Conv1\n+ ReLU", "neurons": 8, "x": 3, "type": "conv"},
        {"name": "MaxPool1", "neurons": 8, "x": 5, "type": "pool"},
        {"name": "Conv2\n+ ReLU", "neurons": 12, "x": 7, "type": "conv"},
        {"name": "MaxPool2", "neurons": 12, "x": 9, "type": "pool"},
        {"name": "Conv3\n+ ReLU", "neurons": 16, "x": 11, "type": "conv"},
        {"name": "MaxPool3", "neurons": 16, "x": 13, "type": "pool"},
        {"name": "Flatten", "neurons": 1, "x": 15, "type": "flatten"},
        {"name": "FC1\n128 neurons", "neurons": 10, "x": 17, "type": "fc"},  # Showing sample of 10 neurons
        {"name": "Output\n3 classes", "neurons": 3, "x": 19, "type": "output"}
    ]
    
    # Draw layers and neurons
    max_neurons = max(layer["neurons"] for layer in layers)
    neuron_positions = {}
    
    for layer in layers:
        y_center = 5
        neuron_y_positions = np.linspace(y_center - 2, y_center + 2, layer["neurons"])
        
        # Draw layer background
        if layer["type"] == "input":
            color = "lightblue"
        elif layer["type"] == "conv":
            color = "lightcoral"
        elif layer["type"] == "pool":
            color = "lightgreen"
        elif layer["type"] == "flatten":
            color = "lightyellow"
        elif layer["type"] == "fc":
            color = "lightpink"
        else:  # output
            color = "lightseagreen"
            
        rect = Rectangle((layer["x"] - 0.4, y_center - 2.5), 0.8, 5, 
                        facecolor=color, alpha=0.7, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
        
        # Draw neurons
        neuron_positions[layer["name"]] = []
        for i, y in enumerate(neuron_y_positions):
            if layer["type"] in ["conv", "pool"]:
                # Represent convolutional layers as small rectangles (feature maps)
                neuron = Rectangle((layer["x"] - 0.2, y - 0.1), 0.4, 0.2, 
                                  facecolor="white", edgecolor="black")
            else:
                # Represent fully connected layers as circles
                neuron = Circle((layer["x"], y), 0.15, facecolor="white", edgecolor="black")
            
            ax.add_patch(neuron)
            neuron_positions[layer["name"]].append((layer["x"], y))
        
        # Add layer name
        ax.text(layer["x"], y_center + 3, layer["name"], 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw connections between layers (simplified)
    for i in range(len(layers) - 1):
        current_layer = layers[i]
        next_layer = layers[i + 1]
        
        # Only show connections between selected neurons for clarity
        if current_layer["type"] in ["conv", "pool"] and next_layer["type"] in ["conv", "pool"]:
            # Convolutional connections (simplified)
            for j in range(min(3, current_layer["neurons"])):
                for k in range(min(3, next_layer["neurons"])):
                    x1, y1 = neuron_positions[current_layer["name"]][j]
                    x2, y2 = neuron_positions[next_layer["name"]][k]
                    ax.plot([x1 + 0.2, x2 - 0.2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
        else:
            # Fully connected connections
            for j in range(min(5, current_layer["neurons"])):
                for k in range(min(5, next_layer["neurons"])):
                    x1, y1 = neuron_positions[current_layer["name"]][j]
                    x2, y2 = neuron_positions[next_layer["name"]][k]
                    ax.plot([x1 + 0.15, x2 - 0.15], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
    
    # Add title and adjust layout
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.set_title('CNN Architecture with Neuron Representation', fontsize=16, pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='Input Layer'),
        Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='black', label='Convolutional Layer'),
        Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', label='Pooling Layer'),
        Rectangle((0, 0), 1, 1, facecolor='lightyellow', edgecolor='black', label='Flatten Operation'),
        Rectangle((0, 0), 1, 1, facecolor='lightpink', edgecolor='black', label='Fully Connected Layer'),
        Rectangle((0, 0), 1, 1, facecolor='lightseagreen', edgecolor='black', label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig('neuron_cnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('neuron_cnn_architecture.svg', bbox_inches='tight')
    plt.show()

create_neuron_architecture_diagram()