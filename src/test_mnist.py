import network
import weather_loader

training_data, test_data = weather_loader.get_x_y_train()

net = network.Network([4, 3, 2])
net.SGD(training_data, 30, 10, 0.1, test_data=test_data)