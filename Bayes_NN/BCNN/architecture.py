import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

def create_bayesian_cnn_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Define architecture layout
    layers = [
        {"name": "Input\nSpectrogram (1 ch)", "neurons": 5, "x": 1, "type": "input"},
        {"name": "Conv1 (16)\n+ BN + ReLU", "neurons": 6, "x": 3, "type": "conv"},
        {"name": "MaxPool1", "neurons": 6, "x": 4, "type": "pool"},
        
        {"name": "Conv2 (32)\n+ BN + ReLU", "neurons": 8, "x": 6, "type": "conv"},
        {"name": "MaxPool2", "neurons": 8, "x": 7, "type": "pool"},
        
        {"name": "Conv3 (64)\n+ BN + ReLU", "neurons": 10, "x": 9, "type": "conv"},
        {"name": "MaxPool3", "neurons": 10, "x": 10, "type": "pool"},
        
        {"name": "Conv4 (128)\n+ BN + ReLU", "neurons": 12, "x": 12, "type": "conv"},
        {"name": "MaxPool4\n+ Dropout(0.2)", "neurons": 12, "x": 13, "type": "pool"},
        
        {"name": "Flatten", "neurons": 1, "x": 15, "type": "flatten"},
        
        {"name": "FC1 (512)\n+ Dropout(0.3)", "neurons": 8, "x": 17, "type": "fc"},
        {"name": "FC2 (256)\n+ Dropout(0.4)", "neurons": 6, "x": 18.5, "type": "fc"},
        {"name": "FC3 (128)", "neurons": 4, "x": 20, "type": "fc"},
        {"name": "Output\n(classes)", "neurons": 3, "x": 21.5, "type": "output"},
    ]
    
    neuron_positions = {}
    y_center = 5
    
    # Draw layers
    for layer in layers:
        neuron_y_positions = np.linspace(y_center - 2, y_center + 2, layer["neurons"])
        
        # Color scheme
        colors = {
            "input": "lightblue",
            "conv": "lightcoral",
            "pool": "lightgreen",
            "flatten": "lightyellow",
            "fc": "lightpink",
            "output": "lightseagreen"
        }
        color = colors.get(layer["type"], "lightgray")
        
        rect = Rectangle((layer["x"] - 0.4, y_center - 2.5), 0.8, 5, 
                         facecolor=color, alpha=0.7, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
        
        neuron_positions[layer["name"]] = []
        for y in neuron_y_positions:
            if layer["type"] in ["conv", "pool"]:
                patch = Rectangle((layer["x"] - 0.2, y - 0.1), 0.4, 0.2, facecolor="white", edgecolor="black")
            else:
                patch = Circle((layer["x"], y), 0.15, facecolor="white", edgecolor="black")
            ax.add_patch(patch)
            neuron_positions[layer["name"]].append((layer["x"], y))
        
        ax.text(layer["x"], y_center + 3.2, layer["name"], ha="center", va="center", fontweight="bold", fontsize=9)
    
    # Draw simplified connections
    for i in range(len(layers) - 1):
        src, dst = layers[i], layers[i + 1]
        for j in range(min(3, src["neurons"])):
            for k in range(min(3, dst["neurons"])):
                x1, y1 = neuron_positions[src["name"]][j]
                x2, y2 = neuron_positions[dst["name"]][k]
                ax.plot([x1 + 0.15, x2 - 0.15], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
    
    # Formatting
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 10)
    ax.set_title("Bayesian CNN Architecture with Dropout", fontsize=16, pad=20)
    ax.axis("off")
    
    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor="lightblue", edgecolor="black", label="Input Layer"),
        Rectangle((0, 0), 1, 1, facecolor="lightcoral", edgecolor="black", label="Conv + BN + ReLU"),
        Rectangle((0, 0), 1, 1, facecolor="lightgreen", edgecolor="black", label="Pooling/Dropout"),
        Rectangle((0, 0), 1, 1, facecolor="lightyellow", edgecolor="black", label="Flatten"),
        Rectangle((0, 0), 1, 1, facecolor="lightpink", edgecolor="black", label="Fully Connected"),
        Rectangle((0, 0), 1, 1, facecolor="lightseagreen", edgecolor="black", label="Output")
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig("bayesian_cnn_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig("bayesian_cnn_architecture.svg", bbox_inches="tight")
    plt.show()

create_bayesian_cnn_diagram()
