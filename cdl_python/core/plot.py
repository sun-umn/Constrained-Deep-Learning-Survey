# stdlib
import os
from typing import Dict, List

# third party
import matplotlib.pyplot as plt
import numpy as np


def plot_by_epoch(arrs, labels, y, title, fn, r_value=None):
    # Generating a colormap to differentiate the
    # arrays with unique colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(arrs)))

    plt.figure(figsize=(10, 6))
    for i, (array, color) in enumerate(zip(arrs, colors)):
        plt.plot(array, label=labels[i], color=color)
    if r_value is not None:
        plt.axhline(y=r, color='r', linestyle='--')

    # plt.title('Plot of 10 Arrays with Unique Colors')
    plt.xlabel('Epoch')
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(fn)
    plt.close()


if __name__ == '__main__':
    optiimizer_names = ['pygranso', 'tfco']
    plot_base_folder = '/home/jusun/zhan7867/Deep_Learning_NTR_CST/exp/figures'
    os.makedirs(plot_base_folder, exist_ok=True)
    base_folder = "/home/jusun/zhan7867/Deep_Learning_NTR_CST/exp/"
    folders = {
        'pygranso': 'PyGRANSO_adult_self_cleaned',
        'tfco': 'TFCO_adult_self_cleaned',
    }
    r_values = [0.1, 0.3]
    file_names = [f"{i:06}" for i in range(5)]

    for r in r_values:
        plot_folder = os.path.join(plot_base_folder, f'r_{r}')
        os.makedirs(plot_folder, exist_ok=True)
        data: Dict[str, List] = {'pygranso': [], 'tfco': []}
        for file in file_names:
            for opt_name in optiimizer_names:
                data[opt_name].append(
                    np.load(
                        os.path.join(
                            base_folder, f'r_{r}', folders[opt_name], file + '.npz'
                        )
                    )
                )
        y_ls = ['accuracy', 'accuracy_test', 'f', 'f_test', 'diff', 'diff_test']
        y_name_ls = [
            'accuracy',
            'accuracy_test',
            'obj',
            'obj_test',
            'constr',
            'constr_test',
        ]
        for y, y_name in zip(y_ls, y_name_ls):
            for opt_name in optiimizer_names:
                arr = [opt_data[y] for opt_data in data[opt_name]]
                labels = ['init_' + fn for fn in file_names]
                title = f'{opt_name} ({y_name},r={r})'

                filename = os.path.join(plot_folder, f'{y_name}_{opt_name}.png')
                if 'diff' in y:
                    plot_by_epoch(arr, labels, y, title, filename, r)
                else:
                    plot_by_epoch(arr, labels, y, title, filename)
        # for k, init_fn in enumerate(file_names):
        #    arr = [data[opt_name][k][y] for opt_name in optiimizer_names]
        #    labels = optiimizer_names
        #    title = f'{init_fn} {y}'
        #    filename = os.path.join(plot_folder,f'{y}_{init_fn}.png')
        #    plot_by_epoch(arr,labels,y,title,filename)

    # x_ls = []
