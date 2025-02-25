from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import numpy as np
import sys

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
        self.hidden_layer_plot = pg.ImageItem()
        
        layout.addWidget(self.loss_plot)
        layout.addWidget(self.activation_plot)
        layout.addWidget(pg.GraphicsLayoutWidget().addItem(self.hidden_layer_plot))
        
        # Data storage
        self.losses = []
        self.activations = []
        
    def update_plots(self, loss, activations, hidden_weights):
        # Update loss curve
        self.losses.append(loss)
        self.loss_plot.plot(self.losses, clear=True)
        
        # Update activation patterns
        self.activation_plot.plot(activations.mean(0).cpu().numpy(), clear=True)
        
        # Update hidden layer visualization
        self.hidden_layer_plot.setImage(hidden_weights.cpu().numpy())
