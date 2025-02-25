from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import numpy as np
import sys
import torch

class TrainingVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAE Training Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Plot windows
        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.activation_plot = pg.PlotWidget(title="Neuron Activations")
        
        # Hidden layer visualization
        graphics_layout = pg.GraphicsLayoutWidget()
        view = graphics_layout.addViewBox()
        self.hidden_layer_plot = pg.ImageItem()
        view.addItem(self.hidden_layer_plot)
        
        # Add widgets to layout
        layout.addWidget(self.loss_plot)
        layout.addWidget(self.activation_plot)
        layout.addWidget(graphics_layout)
        
        # Data storage
        self.losses = []
        self.activations = []
        
    def update_plots(self, loss, activations, hidden_weights):
        # Update loss curve
        self.losses.append(loss)
        self.loss_plot.plot(self.losses, clear=True)
        
        # Update activation patterns - detach before converting to numpy
        with torch.no_grad():
            activation_means = activations.mean(0).cpu().detach().numpy()
            hidden_weights_np = hidden_weights.cpu().detach().numpy()
        
        self.activation_plot.plot(activation_means, clear=True)
        self.hidden_layer_plot.setImage(hidden_weights_np)
