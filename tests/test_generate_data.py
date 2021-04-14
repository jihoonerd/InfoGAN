from infogan.data.generate_toy_example import generate_circle_toy_data
import matplotlib.pyplot as plt


def test_toy_circle_data():
    data = generate_circle_toy_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    